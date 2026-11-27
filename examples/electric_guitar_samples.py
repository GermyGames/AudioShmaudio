"""Generate realistic, DAW-ready electric guitar samples and SFZ patches via audio-shmaudio.

Synthesizes 26 sustained multisampled electric guitar notes across E2-E5 (13 minor-third
roots x 2 velocity layers: clean warm neck/mid and dug-in edge-of-breakup) plus 7
palm-muted bass/mid notes (E2-Bb3), all in 24-bit / 44.1 kHz mono WAV format.

Dogfoods `audio-shmaudio` instrument & foley primitives (`plucked_string`,
`resonant_lowpass`, `saturate`, `cabinet_filter`, `FoleyLayer`, `modal_filter`,
`StereoScene`, `measure_pitch`, `write_sfz`, and `SpectralCanvas`).

Includes:
- Multi-pass iterative refinement on the hero E2 (v2) note with spectrogram diagnostics
  and progression MP4 export (`output/electric_guitar/refinement/`).
- Cross-register prototype rendering (`--test-register`) for critical listening & tuning.
- Full DAW sample pack, ready-to-load SFZ instruments, manifest JSON, and a stereo
  musical demo groove built exclusively from the exported sample WAVs (`--full`).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, resample_poly, sosfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/audio-shmaudio/scripts"))
from spectral_foley import (  # noqa: E402
    SpectralCanvas,
    cabinet_filter,
    measure_pitch,
    modal_filter,
    peaking_biquad,
    plucked_string,
    resonant_lowpass,
    saturate,
    write_sf2,
    write_sfz,
)

SR = 44100
SUSTAIN_DURATION = 4.0
PALM_MUTE_DURATION = 1.2
L_SCALE_MM = 647.7  # 25.5" Stratocaster scale length
D_NECK_MM = 156.0
D_MID_MM = 100.0
D_BRIDGE_MM = 42.0


@dataclass(frozen=True)
class StringSpec:
    string_num: int
    open_name: str
    open_hz: float
    wound: bool
    b0: float
    t60_low: float
    t60_hi: float
    h_detune_cents: float
    pm_t60_low: float
    pm_t60_hi: float


@dataclass(frozen=True)
class NoteSpec:
    midi: int
    name: str
    freq_hz: float
    string_num: int
    fret: int


STRING_SPECS: Dict[int, StringSpec] = {
    6: StringSpec(6, "E2", 82.4069, True, 7.5e-5, 8.5, 1.45, 0.42, 0.52, 0.185),
    5: StringSpec(5, "A2", 110.0000, True, 5.5e-5, 8.0, 1.38, 0.48, 0.48, 0.172),
    4: StringSpec(4, "D3", 146.8324, True, 4.5e-5, 7.2, 1.28, 0.55, 0.42, 0.160),
    3: StringSpec(3, "G3", 195.9977, False, 8.0e-5, 6.2, 1.35, 0.62, 0.36, 0.148),
    2: StringSpec(2, "B3", 246.9417, False, 4.5e-5, 5.2, 1.45, 0.68, 0.32, 0.138),
    1: StringSpec(1, "E4", 329.6276, False, 3.0e-5, 3.9, 1.75, 0.72, 0.28, 0.130),
}

NOTE_SPECS: Tuple[NoteSpec, ...] = (
    NoteSpec(40, "E2", 82.4069, 6, 0),
    NoteSpec(43, "G2", 97.9989, 6, 3),
    NoteSpec(46, "Bb2", 116.5409, 5, 1),
    NoteSpec(49, "Db3", 138.5913, 5, 4),
    NoteSpec(52, "E3", 164.8138, 4, 2),
    NoteSpec(55, "G3", 195.9977, 3, 0),
    NoteSpec(58, "Bb3", 233.0819, 3, 3),
    NoteSpec(61, "Db4", 277.1826, 2, 2),
    NoteSpec(64, "E4", 329.6276, 1, 0),
    NoteSpec(67, "G4", 391.9954, 1, 3),
    NoteSpec(70, "Bb4", 466.1638, 1, 6),
    NoteSpec(73, "Db5", 554.3653, 1, 9),
    NoteSpec(76, "E5", 659.2551, 1, 12),
)

PALM_MUTE_MIDIS = {40, 43, 46, 49, 52, 55, 58}

# Solid-body alder/maple resonance modes (freq_hz, decay_sec, level)
SOLID_BODY_MODES: Tuple[Tuple[float, float, float], ...] = (
    (118.0, 0.065, 0.70),
    (236.0, 0.050, 0.90),
    (432.0, 0.038, 1.00),
    (775.0, 0.026, 0.65),
    (1320.0, 0.018, 0.35),
)


def _string_params(note: NoteSpec, spec: StringSpec, velocity_mode: str) -> dict:
    """Compute physical fretboard, plectrum, pickup, and tension parameters for a note."""
    f0 = note.freq_hz
    l_eff = L_SCALE_MM * (2.0 ** (-note.fret / 12.0))
    stiffness = min(spec.b0 * (2.0 ** (note.fret / 6.0)), 8.0e-4)

    mu_neck = float(np.clip(D_NECK_MM / l_eff, 0.08, 0.49))
    mu_mid = float(np.clip(D_MID_MM / l_eff, 0.08, 0.49))
    mu_bridge = float(np.clip(D_BRIDGE_MM / l_eff, 0.06, 0.45))

    if velocity_mode == "v1":
        d_pick = 132.0
        pick_lp_hz = float(np.clip(3300.0 + 2.4 * f0, 3300.0, 4900.0))
        noise_amount = 0.16
        pulse_ms = float(np.clip(0.30 + 0.08 * (200.0 / max(f0, 80.0)), 0.24, 0.46))
        vel_cutoff_hz = float(np.clip(1.65 * f0, 145.0, 440.0))
        pickup_pos = ((mu_neck, 0.54), (mu_mid, 0.46))
        wound_ripple = 0.07 if spec.wound else 0.0
        t60_low = spec.t60_low * (0.90 if note.fret > 0 else 1.0)
        t60_hi = spec.t60_hi * (0.92 if note.fret > 0 else 1.0)
        pitch_drop_cents = float(np.clip(5.5 - 0.07 * (note.midi - 40), 3.0, 6.5))
        pitch_drop_tau = 0.055
    elif velocity_mode == "v2":
        d_pick = 108.0
        pick_lp_hz = float(np.clip(5400.0 + 2.8 * f0, 5400.0, 7600.0))
        noise_amount = 0.28
        pulse_ms = float(np.clip(0.14 + 0.04 * (200.0 / max(f0, 80.0)), 0.12, 0.24))
        vel_cutoff_hz = float(np.clip(2.25 * f0, 195.0, 540.0))
        pickup_pos = ((mu_neck, 0.48), (mu_mid, 0.52))
        wound_ripple = 0.14 if spec.wound else 0.0
        t60_low = spec.t60_low * (0.90 if note.fret > 0 else 1.0)
        t60_hi = spec.t60_hi * (0.92 if note.fret > 0 else 1.0)
        pitch_drop_cents = float(np.clip(14.5 - 0.18 * (note.midi - 40), 7.5, 16.5))
        pitch_drop_tau = 0.068
    else:  # palm_mute
        d_pick = 95.0
        pick_lp_hz = float(np.clip(4000.0 + 2.2 * f0, 4000.0, 5600.0))
        noise_amount = 0.26
        pulse_ms = float(np.clip(0.15 + 0.05 * (200.0 / max(f0, 80.0)), 0.14, 0.26))
        vel_cutoff_hz = float(np.clip(2.40 * f0, 210.0, 500.0))
        pickup_pos = ((mu_mid, 0.50), (mu_bridge, 0.50))
        wound_ripple = 0.14 if spec.wound else 0.0
        t60_low = spec.pm_t60_low
        t60_hi = spec.pm_t60_hi
        pitch_drop_cents = 15.0 if note.midi <= 49 else 11.0
        pitch_drop_tau = 0.024

    return {
        "t60_low": t60_low,
        "t60_hi": t60_hi,
        "stiffness": stiffness,
        "pluck_pos": float(np.clip(d_pick / l_eff, 0.11, 0.40)),
        "pickup_pos": pickup_pos,
        "pickup_comb_depth": 0.76,
        "pulse_ms": pulse_ms,
        "vel_cutoff_hz": vel_cutoff_hz,
        "pick_lp_hz": pick_lp_hz,
        "noise_amount": noise_amount,
        "wound_ripple": wound_ripple,
        "dual_pol_detune_cents": spec.h_detune_cents,
        "bridge_coupling": 0.0012,
        "cross_pol_mix": 0.22,
        "pitch_drop_cents": pitch_drop_cents,
        "pitch_drop_tau": pitch_drop_tau,
        "flux_asymmetry": 0.032,
        "attack_time": 0.004,
    }


def add_guitar_mechanics(
    canvas: SpectralCanvas,
    pu_signal: np.ndarray,
    buf_v: np.ndarray,
    note: NoteSpec,
    spec: StringSpec,
    velocity_mode: str,
    rng: np.random.Generator,
    *,
    t_attack: float = 0.004,
) -> np.ndarray:
    """Layer FoleyLayer mechanical transients, solid-body wood resonance, sympathetic bloom, and fret buzz."""
    sr = canvas.sr
    sig = pu_signal / max(float(np.max(np.abs(pu_signal))), 1e-8)

    mech = canvas.new_layer()
    if velocity_mode == "v1":
        pick_gain = 0.022
        scrape_gain = 0.015 if spec.wound else 0.006
    elif velocity_mode == "v2":
        pick_gain = 0.045
        scrape_gain = 0.028 if spec.wound else 0.010
    else:
        pick_gain = 0.042
        scrape_gain = 0.024 if spec.wound else 0.010

    mech.friction(
        0.0012,
        0.0055 if spec.wound else 0.0038,
        band=(1900, 5600) if spec.wound else (2200, 5200),
        end_band=(1400, 4000),
        roughness=0.72 if spec.wound else 0.38,
        pressure=((0.0, 0.0), (0.35, 1.0), (1.0, 0.0)),
        gain=scrape_gain,
    )
    mech.contact(
        t_attack - 0.0002,
        0.0032,
        band=(1650, 5200),
        gain=pick_gain,
        modes=((2750.0, 0.0045, 0.28),),
    )
    if note.fret > 0:
        mech.contact(
            t_attack - 0.0008,
            0.0055,
            band=(135, 820),
            gain=0.026 if velocity_mode == "v2" else 0.016,
            modes=((240.0, 0.016, 0.45),),
        )
    if velocity_mode == "pm":
        mech.contact(
            t_attack - 0.0003,
            0.015,
            band=(85, 420),
            gain=0.075,
            modes=((132.0, 0.042, 0.55),),
        )

    mech.resonate(SOLID_BODY_MODES, dry=0.65, gain=1.0)

    combined = sig + mech.audio.astype(np.float64)
    combined = combined + 0.095 * modal_filter(combined, sr, SOLID_BODY_MODES, dry=0.0)

    if velocity_mode != "pm":
        symp_modes: List[Tuple[float, float, float]] = []
        for s_num, s_info in STRING_SPECS.items():
            if s_num == note.string_num:
                continue
            f_sym = s_info.open_hz
            symp_modes.append((f_sym, 1.35, 0.85))
            if f_sym * 2.0 < sr * 0.45:
                symp_modes.append((f_sym * 2.0, 0.85, 0.35))
        if symp_modes:
            combined = combined + 0.016 * modal_filter(sig, sr, symp_modes, dry=0.0)

    if velocity_mode == "v2" and note.midi <= 55:
        n_buzz = min(len(combined), round(0.14 * sr))
        v_norm = np.abs(buf_v[:n_buzz]) / max(float(np.max(np.abs(buf_v[:n_buzz]))), 1e-8)
        excess = np.maximum(v_norm - 0.56, 0.0)
        if np.any(excess > 0.0):
            buzz_noise = sosfilt(
                butter(2, [2100.0, 5400.0], btype="bandpass", fs=sr, output="sos"),
                rng.standard_normal(n_buzz),
            )
            buzz_noise /= max(float(np.max(np.abs(buzz_noise))), 1e-8)
            buzz_env = excess * np.exp(-np.arange(n_buzz) / (0.045 * sr))
            combined[:n_buzz] += 0.075 * buzz_env * buzz_noise

    return combined


def apply_pickup_amp_cab(
    pre_pickup: np.ndarray,
    note: NoteSpec,
    velocity_mode: str,
    sr: int,
) -> np.ndarray:
    """Process pickup signal through RLC resonance, tone stack, 4x 12AX7 saturation, and 1x12 cab IR."""
    pitch_trim = 0.94 + 0.34 * ((note.midi - 40) / 36.0)
    x = (pre_pickup / max(float(np.max(np.abs(pre_pickup))), 1e-8)) * pitch_trim

    if velocity_mode == "v1":
        f_res, q_res = 3150.0, 1.75
        drive, bias, pre_level = 1.28, 0.11, 0.56
    elif velocity_mode == "v2":
        f_res, q_res = 3350.0, 1.95
        drive, bias, pre_level = 3.15, 0.18, 1.18
    else:
        f_res, q_res = 3200.0, 1.85
        drive, bias, pre_level = 3.35, 0.17, 1.22

    x = resonant_lowpass(x, sr, f_res, q=q_res)
    x = sosfilt(butter(2, 60.0, btype="highpass", fs=sr, output="sos"), x)
    eq_curve = (
        ((180.0, 0.95, 2.2), (450.0, 0.90, -2.5), (2250.0, 1.05, 2.5))
        if velocity_mode == "pm"
        else ((155.0, 0.90, 1.4), (480.0, 0.85, -2.4), (2350.0, 1.00, 2.0))
    )
    for f_hz, q_v, g_db in eq_curve:
        b_eq, a_eq = peaking_biquad(f_hz, q_v, g_db, sr)
        x = lfilter(b_eq, a_eq, x)

    x = (x / max(float(np.max(np.abs(x))), 1e-8)) * pre_level
    y_amp = saturate(x, sr, drive=drive, bias=bias, sag=0.55, sag_release=0.12, oversample=4, dc_block_hz=45.0)
    y_cab = cabinet_filter(y_amp, sr, preset="guitar_1x12", dc_block_hz=35.0)

    fade_s = 0.18 if velocity_mode == "pm" else 0.42
    n_fade = min(round(fade_s * sr), len(y_cab) // 3)
    if n_fade >= 4:
        y_cab[-n_fade:] *= 0.5 + 0.5 * np.cos(np.pi * (np.arange(n_fade) + 1) / n_fade)
    return y_cab.astype(np.float64)


def target_peak_for_mode(velocity_mode: str) -> float:
    if velocity_mode == "v2":
        return -1.2
    if velocity_mode == "v1":
        return -4.2
    return -2.0


def synthesize_note(
    note: NoteSpec,
    velocity_mode: str,
    *,
    seed: int,
    stage: str = "full",
    sr: int = SR,
) -> Tuple[np.ndarray, SpectralCanvas]:
    """Synthesize a single electric guitar note at a chosen refinement stage."""
    duration = PALM_MUTE_DURATION if velocity_mode == "pm" else SUSTAIN_DURATION
    canvas = SpectralCanvas(duration=duration, sr=sr, seed=seed)
    rng = np.random.default_rng(seed)
    spec = STRING_SPECS[note.string_num]
    params = _string_params(note, spec, velocity_mode)

    if stage == "pass_01":
        params_p1 = {
            **params,
            "pulse_ms": 0.38,
            "noise_amount": 0.22,
            "wound_ripple": 0.0,
            "pitch_drop_cents": 0.0,
            "tail_fade": 0.35,
        }
        raw_pu = plucked_string(duration, sr, note.freq_hz, rng=rng, **params_p1)
        return raw_pu, canvas

    pu_signal, buf_v, _ = plucked_string(
        duration, sr, note.freq_hz, rng=rng, tail_fade=0.0, return_polarizations=True, **params
    )
    pre_amp = add_guitar_mechanics(canvas, pu_signal, buf_v, note, spec, velocity_mode, rng)
    if stage == "pass_02":
        n_fade = round(0.35 * sr)
        pre_amp[-n_fade:] *= 0.5 + 0.5 * np.cos(np.pi * (np.arange(n_fade) + 1) / n_fade)
        return pre_amp, canvas

    if stage == "pass_03":
        final = apply_pickup_amp_cab(pre_amp * 0.85, note, "v1", sr)
        return final, canvas

    final = apply_pickup_amp_cab(pre_amp, note, velocity_mode, sr)
    return final, canvas


def run_refinement_passes(out_dir: Path) -> None:
    """Render iterative Pass 1..4 progression on hero E2 (v2) note and export progression MP4."""
    ref_dir = out_dir / "refinement"
    ref_dir.mkdir(parents=True, exist_ok=True)
    hero_note = NOTE_SPECS[0]  # E2
    hero_seed = 4002

    canvas = SpectralCanvas(duration=SUSTAIN_DURATION, sr=SR, seed=hero_seed)

    p1_audio, _ = synthesize_note(hero_note, "v2", seed=hero_seed, stage="pass_01")
    d1 = canvas.render_mix(
        p1_audio, str(ref_dir / "pass_01.wav"), str(ref_dir / "pass_01.png"),
        "Pass 1: Dual-Pol Waveguide + Combs", target_peak_db=-1.2, subtype="PCM_24",
    )
    print("Pass 1:", json.dumps(d1))

    canvas.reset(seed=hero_seed)
    p2_audio, _ = synthesize_note(hero_note, "v2", seed=hero_seed, stage="pass_02")
    d2 = canvas.render_mix(
        p2_audio, str(ref_dir / "pass_02.wav"), str(ref_dir / "pass_02.png"),
        "Pass 2: Pick/Fret Foley, Body & Sympathetic Modes", target_peak_db=-1.2, subtype="PCM_24",
    )
    print("Pass 2:", json.dumps(d2))

    canvas.reset(seed=hero_seed)
    p3_audio, _ = synthesize_note(hero_note, "v2", seed=hero_seed, stage="pass_03")
    d3 = canvas.render_mix(
        p3_audio, str(ref_dir / "pass_03.wav"), str(ref_dir / "pass_03.png"),
        "Pass 3: Pickup RLC + Tube Amp + 1x12 Cab IR", target_peak_db=-1.2, subtype="PCM_24",
    )
    print("Pass 3:", json.dumps(d3))

    canvas.reset(seed=hero_seed)
    p4_audio, _ = synthesize_note(hero_note, "v2", seed=hero_seed, stage="full")
    d4 = canvas.render_mix(
        p4_audio, str(ref_dir / "pass_04.wav"), str(ref_dir / "pass_04.png"),
        "Pass 4: Final Tuned E2 (Edge-of-Breakup)", target_peak_db=-1.2, subtype="PCM_24",
    )
    print("Pass 4:", json.dumps(d4))

    if shutil.which("ffmpeg"):
        mp4_path = canvas.export_progression_video(str(ref_dir / "hero_E2_progression.mp4"))
        print("Exported progression MP4:", mp4_path)


def run_test_register(out_dir: Path) -> None:
    """Render prototype notes across low/mid/high registers and palm mute for listening inspection."""
    proto_dir = out_dir / "prototypes"
    proto_dir.mkdir(parents=True, exist_ok=True)
    by_name = {n.name: n for n in NOTE_SPECS}
    tests = [
        (by_name["E2"], "v1"),
        (by_name["E2"], "v2"),
        (by_name["G3"], "v2"),
        (by_name["E4"], "v2"),
        (by_name["E5"], "v1"),
        (by_name["E5"], "v2"),
        (by_name["E2"], "pm"),
    ]
    for idx, (note, mode) in enumerate(tests):
        seed = 1000 + note.midi * 10 + idx
        audio, canvas = synthesize_note(note, mode, seed=seed, stage="full")
        wav_path = proto_dir / f"proto_{note.name}_{mode}.wav"
        png_path = proto_dir / f"proto_{note.name}_{mode}_spec.png"
        diag = canvas.render_mix(
            audio, str(wav_path), str(png_path),
            f"Proto {note.name} ({mode})", target_peak_db=target_peak_for_mode(mode), subtype="PCM_24",
        )
        f_meas, cents = measure_pitch(audio, SR, note.freq_hz)
        print(
            f"{note.name:3s} [{mode:2s}] -> pitch={f_meas} Hz ({cents} cents), "
            f"centroid={diag['weighted_centroid_hz']} Hz, crest={diag['crest_factor_db']} dB, "
            f">7k={diag['band_energy_pct']['above_7000_hz']}%"
        )


def _resample_semitones(audio: np.ndarray, semitones: float) -> np.ndarray:
    """High-quality pitch shift of a mono sample by `semitones` via 4x upsampled cubic interpolation."""
    if abs(semitones) < 1e-6:
        return audio.astype(np.float64, copy=True)
    rate = 2.0 ** (semitones / 12.0)
    up = resample_poly(audio.astype(np.float64), 4, 1)
    new_len = max(1, int(math.floor(len(audio) / rate)))
    pos = np.arange(new_len, dtype=np.float64) * (4.0 * rate)
    m = np.floor(pos).astype(np.int64)
    valid = (m >= 1) & (m + 2 < len(up))
    out = np.zeros(new_len, dtype=np.float64)
    if np.any(valid):
        mv = m[valid]
        u = pos[valid] - mv
        c_m1 = -u * (u - 1.0) * (u - 2.0) / 6.0
        c_0 = (u + 1.0) * (u - 1.0) * (u - 2.0) / 2.0
        c_1 = -(u + 1.0) * u * (u - 2.0) / 2.0
        c_2 = (u + 1.0) * u * (u - 1.0) / 6.0
        out[valid] = (
            c_m1 * up[mv - 1] + c_0 * up[mv] + c_1 * up[mv + 1] + c_2 * up[mv + 2]
        )
    return out


def _write_sfz_patches(out_dir: Path) -> None:
    """Write DAW/sampler-ready SFZ instruments using `write_sfz`."""
    sust_regions = []
    for idx, note in enumerate(NOTE_SPECS):
        lokey = note.midi - 1 if idx > 0 else note.midi - 2
        hikey = note.midi + 1 if idx < len(NOTE_SPECS) - 1 else note.midi + 2
        for mode, lovel, hivel in (("v1", 1, 78), ("v2", 79, 127)):
            sust_regions.append({
                "sample": f"samples/sustained/egtr_sust_{mode}_{note.midi}_{note.name}.wav",
                "pitch_keycenter": note.midi,
                "lokey": lokey,
                "hikey": hikey,
                "lovel": lovel,
                "hivel": hivel,
            })
    write_sfz(
        out_dir / "electric_guitar_sustained.sfz",
        sust_regions,
        attack=0.001,
        release=0.14,
        amp_veltrack=45,
        header_comment="AudioShmaudio Electric Guitar - Sustained (26 samples, E2-E5, 2 velocity layers)",
    )

    pm_notes = [n for n in NOTE_SPECS if n.midi in PALM_MUTE_MIDIS]
    pm_regions = []
    for idx, note in enumerate(pm_notes):
        lokey = note.midi - 1 if idx > 0 else note.midi - 2
        hikey = note.midi + 1 if idx < len(pm_notes) - 1 else note.midi + 2
        pm_regions.append({
            "sample": f"samples/palm_mute/egtr_pm_{note.midi}_{note.name}.wav",
            "pitch_keycenter": note.midi,
            "lokey": lokey,
            "hikey": hikey,
            "lovel": 1,
            "hivel": 127,
        })
    write_sfz(
        out_dir / "electric_guitar_palm_mute.sfz",
        pm_regions,
        attack=0.001,
        release=0.08,
        amp_veltrack=65,
        header_comment="AudioShmaudio Electric Guitar - Palm Mute (7 samples, E2-Bb3)",
    )
    write_sf2(
        out_dir / "electric_guitar.sf2",
        [
            {
                "name": "Electric Guitar",
                "bank": 0,
                "program": 0,
                "regions": sust_regions,
                "attack": 0.001,
                "release": 0.14,
                "amp_veltrack": 45.0,
            },
            {
                "name": "Guitar Palm Mute",
                "bank": 0,
                "program": 1,
                "regions": pm_regions,
                "attack": 0.001,
                "release": 0.08,
                "amp_veltrack": 65.0,
            },
        ],
        bank_name="AudioShmaudio Electric Guitar",
        author="AudioShmaudio",
        comment="24-bit/44.1kHz physical-modeled electric guitar (Sustained v1/v2 + Palm Mute)",
        bit_depth=24,
        sample_root=out_dir,
    )


def build_demo_riff_from_samples(out_dir: Path) -> dict:
    """Assemble an 11.2-second stereo guitar demo EXCLUSIVELY from the exported sample WAVs."""
    demo_dir = out_dir / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    sust_dir = out_dir / "samples/sustained"
    pm_dir = out_dir / "samples/palm_mute"

    sust_bank: Dict[Tuple[str, int], np.ndarray] = {}
    for note in NOTE_SPECS:
        for mode in ("v1", "v2"):
            wav_p = sust_dir / f"egtr_sust_{mode}_{note.midi}_{note.name}.wav"
            data, file_sr = sf.read(wav_p, dtype="float64")
            assert file_sr == SR
            sust_bank[(mode, note.midi)] = data

    pm_bank: Dict[int, np.ndarray] = {}
    for note in NOTE_SPECS:
        if note.midi in PALM_MUTE_MIDIS:
            wav_p = pm_dir / f"egtr_pm_{note.midi}_{note.name}.wav"
            data, file_sr = sf.read(wav_p, dtype="float64")
            assert file_sr == SR
            pm_bank[note.midi] = data

    def _trigger_sample(
        layer_buf: np.ndarray,
        midi: int,
        art: str,
        t_sec: float,
        dur_sec: float,
        gain: float,
    ) -> None:
        pool = pm_bank if art == "pm" else {m: sust_bank[(art, m)] for (mode, m) in sust_bank if mode == art}
        nearest = min(pool.keys(), key=lambda m: abs(m - midi))
        shifted = _resample_semitones(pool[nearest], float(midi - nearest))
        max_len = min(len(shifted), round((dur_sec + 0.04) * SR))
        clip = shifted[:max_len].copy()
        fade_len = min(round(0.038 * SR), len(clip) // 2)
        if fade_len >= 4:
            clip[-fade_len:] *= 0.5 + 0.5 * np.cos(np.pi * (np.arange(fade_len) + 1) / fade_len)
        start = max(0, round((t_sec - 0.004) * SR))
        end = min(len(layer_buf), start + len(clip))
        if end > start:
            layer_buf[start:end] += gain * clip[: end - start]

    duration = 11.2
    canvas = SpectralCanvas(duration=duration, sr=SR, seed=777)
    demo_rng = np.random.default_rng(777)
    rhythm_pm = canvas.new_layer()
    arp_clean = canvas.new_layer()
    lead_edge = canvas.new_layer()

    bpm = 96.0
    beat = 60.0 / bpm
    t0 = 0.12

    pm_events = [
        (0.0, 40, 0.24, 0.90), (0.5, 40, 0.20, 0.65), (1.0, 40, 0.26, 0.85),
        (1.5, 47, 0.22, 0.72), (2.0, 40, 0.24, 0.88), (2.5, 40, 0.20, 0.68),
        (3.0, 40, 0.24, 0.85), (3.5, 43, 0.22, 0.78),
        (4.0, 48, 0.26, 0.90), (4.5, 48, 0.20, 0.65), (5.0, 48, 0.24, 0.85),
        (5.5, 55, 0.22, 0.72), (6.0, 48, 0.25, 0.88), (6.5, 48, 0.20, 0.66),
        (7.0, 50, 0.25, 0.85), (7.5, 55, 0.22, 0.76),
        (8.0, 43, 0.28, 0.90), (8.5, 43, 0.20, 0.65), (9.0, 47, 0.25, 0.80),
        (10.0, 42, 0.28, 0.88), (10.5, 42, 0.20, 0.65), (11.0, 50, 0.28, 0.82),
    ]
    for b_pos, midi, dur_b, vel in pm_events:
        jitter = float(demo_rng.uniform(-0.005, 0.005))
        _trigger_sample(rhythm_pm.audio, midi, "pm", t0 + b_pos * beat + jitter, dur_b * beat, vel * 0.88)

    arp_events = [
        (0.00, 40, 3.90, 0.68), (0.50, 52, 3.40, 0.56), (1.00, 55, 2.90, 0.60),
        (1.50, 59, 2.40, 0.62), (2.25, 66, 1.65, 0.65), (3.00, 64, 0.92, 0.58),
        (4.00, 48, 3.90, 0.68), (4.50, 55, 3.40, 0.56), (5.00, 60, 2.90, 0.60),
        (5.50, 62, 2.40, 0.62), (6.25, 67, 1.65, 0.65), (7.00, 64, 0.92, 0.58),
        (8.00, 43, 1.92, 0.68), (8.50, 55, 1.42, 0.58), (9.00, 59, 0.92, 0.60), (9.50, 62, 0.44, 0.62),
        (10.00, 50, 1.92, 0.68), (10.50, 57, 1.42, 0.58), (11.00, 62, 0.92, 0.62), (11.50, 66, 0.44, 0.64),
        (12.000, 40, 4.40, 0.72), (12.028, 47, 4.37, 0.62), (12.056, 52, 4.34, 0.64),
        (12.086, 55, 4.31, 0.62), (12.115, 59, 4.28, 0.64), (12.145, 66, 4.25, 0.68),
    ]
    for b_pos, midi, dur_b, vel in arp_events:
        jitter = float(demo_rng.uniform(-0.004, 0.004)) if b_pos < 12.0 else 0.0
        _trigger_sample(arp_clean.audio, midi, "v1", t0 + b_pos * beat + jitter, dur_b * beat, vel * 0.72)

    lead_events = [
        (1.75, 55, 0.24, 0.70), (2.00, 62, 0.46, 0.80), (2.50, 64, 0.44, 0.76), (3.00, 67, 0.88, 0.90),
        (5.75, 71, 0.24, 0.74), (6.00, 74, 0.46, 0.86), (6.50, 72, 0.44, 0.76), (7.00, 71, 0.86, 0.88),
        (8.75, 67, 0.45, 0.78), (9.25, 69, 0.45, 0.82), (9.75, 71, 0.68, 0.88), (10.50, 74, 0.68, 0.90),
        (11.25, 76, 3.80, 0.96),
    ]
    for b_pos, midi, dur_b, vel in lead_events:
        jitter = float(demo_rng.uniform(-0.004, 0.004))
        _trigger_sample(lead_edge.audio, midi, "v2", t0 + b_pos * beat + jitter, dur_b * beat, vel * 0.80)

    scene = canvas.new_scene()
    scene.add(rhythm_pm, pan=-0.26, distance=0.02, gain=1.0)
    scene.add(arp_clean, pan=-0.10, distance=0.04, gain=1.0)
    scene.add(lead_edge, pan=0.22, distance=0.01, gain=1.05)
    scene.apply_room(size=0.28, damping=0.58, wet=0.14, predelay=0.008, early_reflections=6)
    scene.soft_limit(ceiling_db=-1.0, drive=1.06)
    scene.fade_edges(fade_in=0.005, fade_out=0.85)

    demo_wav = demo_dir / "guitar_demo_riff.wav"
    demo_png = demo_dir / "guitar_demo_riff_spec.png"
    diag = canvas.render_mix(
        scene.audio, str(demo_wav), str(demo_png),
        "Electric Guitar Sample Pack Demo (Stereo)", subtype="PCM_24",
    )
    print("Demo Riff:", json.dumps(diag))
    return diag


def run_full_pack(out_dir: Path) -> None:
    """Render all 33 DAW WAV samples, representative spectrograms, SFZ files, manifest, and stereo demo."""
    sust_dir = out_dir / "samples/sustained"
    pm_dir = out_dir / "samples/palm_mute"
    demo_dir = out_dir / "demo"
    sust_dir.mkdir(parents=True, exist_ok=True)
    pm_dir.mkdir(parents=True, exist_ok=True)
    demo_dir.mkdir(parents=True, exist_ok=True)

    rep_specs = {
        ("E2", "v1"): "spec_E2_v1.png",
        ("E2", "v2"): "spec_E2_v2.png",
        ("G3", "v2"): "spec_G3_v2.png",
        ("E4", "v2"): "spec_E4_v2.png",
        ("E5", "v1"): "spec_E5_v1.png",
        ("E2", "pm"): "spec_E2_pm.png",
    }

    manifest_entries: List[dict] = []
    for note in NOTE_SPECS:
        modes = ["v1", "v2"] + (["pm"] if note.midi in PALM_MUTE_MIDIS else [])
        for mode in modes:
            seed = 2000 + note.midi * 17 + (1 if mode == "v1" else (2 if mode == "v2" else 3))
            audio, canvas = synthesize_note(note, mode, seed=seed, stage="full")
            wav_path = (
                pm_dir / f"egtr_pm_{note.midi}_{note.name}.wav"
                if mode == "pm"
                else sust_dir / f"egtr_sust_{mode}_{note.midi}_{note.name}.wav"
            )
            spec_name = rep_specs.get((note.name, mode))
            spec_path = str(demo_dir / spec_name) if spec_name else None
            diag = canvas.render_mix(
                audio,
                str(wav_path),
                spec_path,
                f"{note.name} ({mode})" if spec_path else None,
                target_peak_db=target_peak_for_mode(mode),
                subtype="PCM_24",
            )
            f_meas, cents = measure_pitch(audio, SR, note.freq_hz)
            entry = {
                "note": note.name,
                "midi": note.midi,
                "string": note.string_num,
                "fret": note.fret,
                "articulation": mode,
                "target_hz": note.freq_hz,
                "measured_hz": f_meas,
                "pitch_error_cents": cents,
                "wav_relpath": str(wav_path.relative_to(out_dir)),
                "diagnostics": diag,
            }
            manifest_entries.append(entry)
            print(
                f"Rendered {entry['wav_relpath']:46s} | err={str(cents):>6s} c | "
                f"centroid={diag['weighted_centroid_hz']:>6.1f} Hz | "
                f"crest={diag['crest_factor_db']:>5.2f} dB | tail={diag['tail_rms_db']:>6.1f} dB"
            )

    _write_sfz_patches(out_dir)
    demo_diag = build_demo_riff_from_samples(out_dir)

    manifest = {
        "sample_rate": SR,
        "subtype": "PCM_24",
        "channels": 1,
        "sustained_count": sum(1 for e in manifest_entries if e["articulation"] in ("v1", "v2")),
        "palm_mute_count": sum(1 for e in manifest_entries if e["articulation"] == "pm"),
        "sf2": "electric_guitar.sf2",
        "samples": manifest_entries,
        "demo": demo_diag,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Wrote manifest.json, SFZ patches, and electric_guitar.sf2.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize realistic electric guitar sample pack.")
    parser.add_argument("--refinement-only", action="store_true", help="Render E2 Pass 1-4 + MP4.")
    parser.add_argument("--test-register", action="store_true", help="Render cross-register prototypes.")
    parser.add_argument("--full", action="store_true", help="Render full 33-sample pack, SFZ, and demo.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output/electric_guitar",
        help="Output root directory.",
    )
    args = parser.parse_args()

    if args.refinement_only:
        run_refinement_passes(args.out_dir)
    if args.test_register:
        run_test_register(args.out_dir)
    if args.full or not (args.refinement_only or args.test_register):
        run_refinement_passes(args.out_dir)
        run_full_pack(args.out_dir)


if __name__ == "__main__":
    main()
