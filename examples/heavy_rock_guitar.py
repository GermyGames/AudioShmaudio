"""Generate a 7-string / 24-fret heavy rock electric guitar SF2 & SFZ instrument via audio-shmaudio.

Synthesizes 82 mono 24-bit / 44.1 kHz samples spanning nearly 5 octaves (A1 MIDI 33 to F#6 MIDI 90):
- 38 sustained samples (19 minor-third roots Bb1..E6 x 2 velocity layers: `v1` crunch, `v2` heavy rock roar)
- 22 palm-mute samples (11 riff roots Bb1..E4 x 2 mute depths: `pm1` tight chug, `pm2` half-mute chunk)
- 22 power-chord samples (11 riff roots Bb1..E4 x 2 articulations: `pc1` muted 5th, `pc2` sustained 5th+octave)
  summed *before* pickup RLC & amp saturation so true chord intermodulation is baked into every note.

Exports 4 SFZ patches + a single 24-bit SoundFont 2.04 (`output/heavy_rock_guitar/heavy_rock_guitar.sf2`)
with 4 presets and renders a multi-track heavy rock riff + solo demo via FluidSynth.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

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
SUSTAIN_DURATION = 3.8
PM1_DURATION = 1.45
PM2_DURATION = 1.60

# Bridge humbucker coil distances from bridge saddle (mm)
D_SLUG_MM = 54.0
D_SCREW_MM = 34.0


@dataclass(frozen=True)
class StringSpec:
    string_num: int
    open_name: str
    open_hz: float
    wound: bool
    scale_mm: float
    b0: float
    h_detune_cents: float


@dataclass(frozen=True)
class NoteSpec:
    midi: int
    name: str
    freq_hz: float
    string_num: int
    fret: int


def _midi_to_hz(midi: int) -> float:
    return round(440.0 * (2.0 ** ((midi - 69) / 12.0)), 4)


STRING_SPECS: Dict[int, StringSpec] = {
    7: StringSpec(7, "Bb1", _midi_to_hz(34), True, 665.0, 5.5e-5, 0.44),
    6: StringSpec(6, "E2", _midi_to_hz(40), True, 647.7, 5.2e-5, 0.46),
    5: StringSpec(5, "A2", _midi_to_hz(45), True, 647.7, 4.5e-5, 0.50),
    4: StringSpec(4, "D3", _midi_to_hz(50), True, 647.7, 4.0e-5, 0.54),
    3: StringSpec(3, "G3", _midi_to_hz(55), False, 647.7, 3.8e-5, 0.60),
    2: StringSpec(2, "B3", _midi_to_hz(59), False, 647.7, 3.4e-5, 0.65),
    1: StringSpec(1, "E4", _midi_to_hz(64), False, 647.7, 3.2e-5, 0.70),
}

NOTE_SPECS: Tuple[NoteSpec, ...] = (
    NoteSpec(34, "Bb1", _midi_to_hz(34), 7, 0),
    NoteSpec(37, "Db2", _midi_to_hz(37), 7, 3),
    NoteSpec(40, "E2", _midi_to_hz(40), 6, 0),
    NoteSpec(43, "G2", _midi_to_hz(43), 6, 3),
    NoteSpec(46, "Bb2", _midi_to_hz(46), 5, 1),
    NoteSpec(49, "Db3", _midi_to_hz(49), 5, 4),
    NoteSpec(52, "E3", _midi_to_hz(52), 4, 2),
    NoteSpec(55, "G3", _midi_to_hz(55), 3, 0),
    NoteSpec(58, "Bb3", _midi_to_hz(58), 3, 3),
    NoteSpec(61, "Db4", _midi_to_hz(61), 2, 2),
    NoteSpec(64, "E4", _midi_to_hz(64), 1, 0),
    NoteSpec(67, "G4", _midi_to_hz(67), 1, 3),
    NoteSpec(70, "Bb4", _midi_to_hz(70), 1, 6),
    NoteSpec(73, "Db5", _midi_to_hz(73), 1, 9),
    NoteSpec(76, "E5", _midi_to_hz(76), 1, 12),
    NoteSpec(79, "G5", _midi_to_hz(79), 1, 15),
    NoteSpec(82, "Bb5", _midi_to_hz(82), 1, 18),
    NoteSpec(85, "Db6", _midi_to_hz(85), 1, 21),
    NoteSpec(88, "E6", _midi_to_hz(88), 1, 24),
)

RIFF_MIDIS = {34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64}

# Heavy mahogany/maple set-neck body resonances (freq_hz, decay_sec, level)
MAHOGANY_BODY_MODES: Tuple[Tuple[float, float, float], ...] = (
    (135.0, 0.065, 0.85),
    (265.0, 0.048, 0.95),
    (440.0, 0.036, 1.00),
    (810.0, 0.024, 0.65),
)

AMP_SETTINGS: Dict[str, Tuple[float, float, float, float, float]] = {
    # mode -> (pre_gain, drive1, drive2, drive3, sag3)
    "v1": (0.72, 2.1, 2.0, 1.45, 0.30),
    "v2": (1.00, 3.2, 3.6, 1.85, 0.45),
    "pm1": (1.02, 3.4, 3.7, 1.75, 0.35),
    "pm2": (1.04, 3.5, 3.9, 1.85, 0.42),
    "pc1": (0.98, 3.0, 3.3, 1.75, 0.38),
    "pc2": (0.96, 2.8, 3.2, 1.80, 0.45),
}


def duration_for_mode(mode: str) -> float:
    if mode in ("pm1", "pc1"):
        return PM1_DURATION
    if mode == "pm2":
        return PM2_DURATION
    return SUSTAIN_DURATION


def target_peak_for_mode(mode: str) -> float:
    return {
        "v1": -3.0,
        "v2": -1.2,
        "pm1": -1.6,
        "pm2": -1.4,
        "pc1": -1.6,
        "pc2": -1.4,
    }[mode]


def pitch_window_for_mode(mode: str) -> Tuple[float, float]:
    if mode in ("pm1", "pc1"):
        return 0.15, 0.52
    if mode == "pm2":
        return 0.16, 0.65
    return 0.45, 1.45


def _measure_note_pitch(audio: np.ndarray, note: NoteSpec, mode: str) -> Tuple[Optional[float], Optional[float]]:
    """Measure pitch, using the 2nd harmonic for sub-80 Hz palm mutes above the 64 Hz cab HPF slope."""
    p_start, p_end = pitch_window_for_mode(mode)
    if mode in ("pm1", "pm2", "pc1") and note.freq_hz < 80.0:
        f2, cents = measure_pitch(audio, SR, note.freq_hz * 2.0, start_sec=p_start, end_sec=p_end)
        return (round(f2 / 2.0, 4) if f2 is not None else None), cents
    return measure_pitch(audio, SR, note.freq_hz, start_sec=p_start, end_sec=p_end)


_RR_BASE_PROFILES: Tuple[dict, ...] = (
    {
        "label": "down_a",
        "d_pick_delta": 0.0,
        "pulse_scale": 1.00,
        "pick_lp_scale": 1.00,
        "noise_delta": 0.00,
        "scrape_scale": 1.00,
        "click_scale": 1.00,
        "pitch_drop_delta": 0.0,
        "t60_scale": 1.00,
        "pc1_offsets": (0.004, 0.009),
        "pc2_offsets": (0.004, 0.011, 0.019),
    },
    {
        "label": "up_a",
        "d_pick_delta": -5.5,
        "pulse_scale": 0.91,
        "pick_lp_scale": 1.06,
        "noise_delta": 0.025,
        "scrape_scale": 1.12,
        "click_scale": 1.08,
        "pitch_drop_delta": -1.2,
        "t60_scale": 0.97,
        "pc1_offsets": (0.006, 0.004),
        "pc2_offsets": (0.008, 0.005, 0.003),
    },
    {
        "label": "down_b",
        "d_pick_delta": 4.5,
        "pulse_scale": 1.07,
        "pick_lp_scale": 0.96,
        "noise_delta": -0.015,
        "scrape_scale": 0.92,
        "click_scale": 0.94,
        "pitch_drop_delta": 1.3,
        "t60_scale": 1.03,
        "pc1_offsets": (0.004, 0.011),
        "pc2_offsets": (0.004, 0.013, 0.022),
    },
    {
        "label": "up_b",
        "d_pick_delta": -3.0,
        "pulse_scale": 0.95,
        "pick_lp_scale": 1.03,
        "noise_delta": 0.015,
        "scrape_scale": 1.05,
        "click_scale": 1.04,
        "pitch_drop_delta": -0.6,
        "t60_scale": 0.99,
        "pc1_offsets": (0.005, 0.004),
        "pc2_offsets": (0.006, 0.004, 0.009),
    },
)


def _rr_profile(rr_idx: int) -> dict:
    """Return physical downstroke/upstroke stroke variation profile for 1-based round-robin index."""
    idx0 = max(0, int(rr_idx) - 1)
    base = dict(_RR_BASE_PROFILES[idx0 % len(_RR_BASE_PROFILES)])
    cycle = idx0 // len(_RR_BASE_PROFILES)
    if cycle > 0:
        drift = ((cycle % 3) - 1) * 1.2
        base["d_pick_delta"] += drift
        base["pulse_scale"] *= 1.0 + 0.015 * ((cycle % 2) * 2 - 1)
    return base


def _string_params(
    midi: int,
    freq_hz: float,
    spec: StringSpec,
    fret: int,
    mode: str,
    attack_time: float,
    rr_idx: int = 1,
) -> dict:
    """Compute physical fretboard, dual-coil bridge humbucker, and plectrum parameters."""
    rr = _rr_profile(rr_idx)
    l_eff = spec.scale_mm * (2.0 ** (-fret / 12.0))
    stiffness = min(spec.b0 * (2.0 ** (fret / 6.0)), 2.6e-4)

    mu_slug = float(np.clip(D_SLUG_MM / l_eff, 0.04, 0.42))
    mu_screw = float(np.clip(D_SCREW_MM / l_eff, 0.03, 0.34))
    pickup_pos = ((mu_slug, 0.52), (mu_screw, 0.48))

    u = float(np.clip((midi - 34) / (88 - 34), 0.0, 1.0))
    u_riff = float(np.clip((midi - 34) / (64 - 34), 0.0, 1.0))
    u_hi = float(np.clip((midi - 52) / (64 - 52), 0.0, 1.0))
    u_top = float(np.clip((midi - 64) / (88 - 64), 0.0, 1.0))

    if mode in ("pm1", "pc1"):
        t60_low = (0.36 - 0.14 * u_riff) * rr["t60_scale"]
        t60_hi = (0.11 - 0.035 * u_riff) * rr["t60_scale"]
    elif mode == "pm2":
        t60_low = (0.62 - 0.26 * u_riff) * rr["t60_scale"]
        t60_hi = (0.22 - 0.08 * u_riff) * rr["t60_scale"]
    else:
        t60_low = (8.5 - 3.9 * u) * (0.93 if fret > 0 else 1.0) * rr["t60_scale"]
        t60_hi = (1.45 + 0.75 * u_hi) * (0.93 if fret > 0 else 1.0) * rr["t60_scale"]

    is_light = mode == "v1"
    d_pick = (96.0 if is_light else 82.0) + rr["d_pick_delta"]
    pulse_ms = ((0.24 - 0.09 * u) if is_light else (0.17 - 0.06 * u)) * rr["pulse_scale"]
    pick_lp_hz = min(((4800.0 + 1400.0 * u) if is_light else (6200.0 + 1600.0 * u)) * rr["pick_lp_scale"], SR * 0.44)
    vel_cutoff_hz = float(np.clip(1.9 * freq_hz, 170.0, 480.0) if is_light else np.clip(2.4 * freq_hz, 210.0, 620.0))
    noise_amount = max(0.15, (0.28 if is_light else (0.36 if mode in ("pm1", "pc1") else 0.34)) + rr["noise_delta"])
    wound_ripple = (0.16 if is_light else 0.24) if spec.wound else 0.0

    if is_light:
        pitch_drop_cents = 8.0 - 4.0 * u
        pitch_drop_tau = 0.055
    elif mode in ("pm1", "pm2", "pc1"):
        pitch_drop_cents = 7.0 - 3.0 * u_riff
        pitch_drop_tau = 0.022
    else:
        pitch_drop_cents = 16.0 - 11.0 * u
        pitch_drop_tau = 0.055
    pitch_drop_cents = max(1.5, pitch_drop_cents + rr["pitch_drop_delta"])

    dual_pol_detune = spec.h_detune_cents * (1.0 - u_top) + 0.18 * u_top if midi >= 64 else spec.h_detune_cents
    cross_pol_mix = 0.22 * (1.0 - u_top) + 0.12 * u_top if midi >= 64 else 0.22

    return {
        "t60_low": t60_low,
        "t60_hi": t60_hi,
        "stiffness": stiffness,
        "pluck_pos": float(np.clip(d_pick / l_eff, 0.08, 0.38)),
        "pickup_pos": pickup_pos,
        "pickup_comb_depth": 0.72,
        "pulse_ms": pulse_ms,
        "vel_cutoff_hz": vel_cutoff_hz,
        "pick_lp_hz": pick_lp_hz,
        "noise_amount": noise_amount,
        "wound_ripple": wound_ripple,
        "dual_pol_detune_cents": dual_pol_detune,
        "bridge_coupling": 0.0012,
        "cross_pol_mix": cross_pol_mix,
        "pitch_drop_cents": pitch_drop_cents,
        "pitch_drop_tau": pitch_drop_tau,
        "flux_asymmetry": 0.036,
        "attack_time": attack_time,
    }


def add_heavy_guitar_mechanics(
    canvas: SpectralCanvas,
    pu_signal: np.ndarray,
    buf_v: np.ndarray,
    midi: int,
    spec: StringSpec,
    fret: int,
    mode: str,
    rng: np.random.Generator,
    *,
    t_attack: float = 0.008,
    rr_idx: int = 1,
) -> np.ndarray:
    """Layer pre-amp pick bevel/scrape, palm-heel bridge thump, mahogany resonance, and fret rattle."""
    rr = _rr_profile(rr_idx)
    sr = canvas.sr
    sig = pu_signal / max(float(np.max(np.abs(pu_signal))), 1e-8)

    mech = canvas.new_layer()
    is_light = mode == "v1"
    is_mute = mode in ("pm1", "pm2", "pc1")

    scrape_gain = ((0.014 if is_light else 0.026) if spec.wound else (0.006 if is_light else 0.011)) * rr["scrape_scale"]
    mech.friction(
        max(0.0004, t_attack - 0.007),
        0.0065 if spec.wound else 0.0042,
        band=(2000, 5400) if spec.wound else (2400, 5600),
        end_band=(1500, 3900),
        roughness=0.76 if spec.wound else 0.42,
        pressure=((0.0, 0.0), (0.35, 1.0), (1.0, 0.0)),
        gain=scrape_gain,
    )
    click_gain = (0.028 if is_light else (0.058 if midi >= 64 else 0.052)) * rr["click_scale"]
    mech.contact(
        max(0.0, t_attack - 0.002),
        0.0030,
        band=(2200, 5600) if midi >= 64 else (1800, 4800),
        gain=click_gain,
        modes=((2650.0, 0.004, 0.25),),
    )
    if fret > 0:
        mech.contact(
            max(0.0, t_attack - 0.0012),
            0.0050,
            band=(140, 760),
            gain=0.014 if is_light else 0.022,
            modes=((230.0, 0.014, 0.40),),
        )
    if is_mute:
        mech.contact(
            max(0.0, t_attack - 0.001),
            0.018,
            band=(90, 280),
            gain=0.065 if mode in ("pm1", "pc1") else 0.055,
            modes=((92.0, 0.022, 0.45),),
        )

    mech.resonate(MAHOGANY_BODY_MODES, dry=0.60, gain=1.0)
    combined = sig + mech.audio.astype(np.float64)
    combined = combined + 0.10 * modal_filter(combined, sr, MAHOGANY_BODY_MODES, dry=0.0)

    if mode in ("v1", "v2"):
        symp_modes: List[Tuple[float, float, float]] = []
        for s_num, s_info in STRING_SPECS.items():
            if s_num == spec.string_num:
                continue
            symp_modes.append((s_info.open_hz, 1.15, 0.85))
            if s_info.open_hz * 2.0 < sr * 0.45:
                symp_modes.append((s_info.open_hz * 2.0, 0.70, 0.30))
        if symp_modes:
            combined = combined + 0.012 * modal_filter(sig, sr, symp_modes, dry=0.0)

    if mode in ("v2", "pc2") and midi <= 49:
        n_buzz = min(len(combined), round(0.14 * sr))
        v_norm = np.abs(buf_v[:n_buzz]) / max(float(np.max(np.abs(buf_v[:n_buzz]))), 1e-8)
        excess = np.maximum(v_norm - 0.54, 0.0)
        if np.any(excess > 0.0):
            buzz = sosfilt(
                butter(2, [1900.0, 5200.0], btype="bandpass", fs=sr, output="sos"),
                rng.standard_normal(n_buzz),
            )
            buzz /= max(float(np.max(np.abs(buzz))), 1e-8)
            buzz_env = excess * np.exp(-np.arange(n_buzz) / (0.042 * sr))
            combined[:n_buzz] += 0.065 * buzz_env * buzz

    return combined


def apply_humbucker_rlc(sig: np.ndarray, midi: int, mode: str, sr: int) -> np.ndarray:
    """Model magnetic bridge humbucker inductance + cable capacitance resonant low-pass."""
    if mode == "v1":
        f_res, q_res = 2350.0, 1.80
    else:
        u_top = float(np.clip((midi - 64) / (88 - 64), 0.0, 1.0))
        f_res, q_res = 2500.0 + 350.0 * u_top, 1.95
    return resonant_lowpass(sig, sr, f_res, q=q_res)


def _apply_peaking(sig: np.ndarray, freq_hz: float, q_val: float, gain_db: float, sr: int) -> np.ndarray:
    b, a = peaking_biquad(freq_hz, q_val, gain_db, sr)
    return lfilter(b, a, sig)


def heavy_rock_amp(
    pickup_sig: np.ndarray,
    midi: int,
    mode: str,
    sr: int,
    *,
    apply_cabinet: bool = True,
    apply_tail_fade: bool = True,
) -> np.ndarray:
    """Process humbucker signal through tightened pre-EQ, 3-stage cascaded tube stack, tone stack, and 4x12 cab."""
    u = float(np.clip((midi - 34) / (88 - 34), 0.0, 1.0))
    u_bass = float(np.clip((midi - 40) / (64 - 40), 0.0, 1.0))

    # 1. Pre-gain tightening (Tube Screamer / JCM800 input stage)
    pre_hp_hz = 125.0 - 55.0 * u_bass  # 125 Hz on Bb1..E2 down to 70 Hz on E4..E6
    x = sosfilt(butter(1, pre_hp_hz, btype="highpass", fs=sr, output="sos"), pickup_sig)
    x = _apply_peaking(x, 880.0, 1.10, 2.2 if mode == "v1" else 4.2, sr)
    x = _apply_peaking(x, 2350.0, 1.40, 1.4 if mode == "v1" else 2.4, sr)

    # Register drive normalization so upper leads (G4..E6) saturate cleanly without thin pick spikes starving sustain
    early_end = min(len(x), round(0.28 * sr))
    early_rms = float(np.sqrt(np.mean(x[:early_end] ** 2))) if early_end > 0 else 1e-8
    ref_level = max(float(np.max(np.abs(x))) * 0.85, 2.6 * early_rms, 1e-8)

    pre_gain, drive1, drive2, drive3, sag3 = AMP_SETTINGS[mode]
    drive_reg = 1.0 + 0.42 * u
    x = (x / ref_level) * pre_gain * drive_reg

    # 2. Stage 1: Preamp V1a + inter-stage Miller LP
    y1 = saturate(x, sr, drive=drive1, bias=0.14, sag=0.0, oversample=4, dc_block_hz=32.0)
    y1 = sosfilt(butter(1, 6800.0, btype="lowpass", fs=sr, output="sos"), y1)

    # 3. Stage 2: Preamp V1b Cold Clipper (rich asymmetric 2nd/3rd harmonics)
    gain_v1a = 2.05 if mode == "v1" else 2.25
    y2 = saturate(y1 * gain_v1a, sr, drive=drive2, bias=0.24, sag=0.0, oversample=4, dc_block_hz=32.0)
    y2 = sosfilt(butter(2, 6000.0, btype="lowpass", fs=sr, output="sos"), y2)

    # 4. Stage 3: EL34 Power Amp + Power-Supply Sag
    gain_pi = 1.85 if mode == "v1" else 1.95
    y = saturate(y2 * gain_pi, sr, drive=drive3, bias=0.05, sag=sag3, sag_release=0.095, oversample=4, dc_block_hz=32.0)

    if not apply_cabinet:
        return y.astype(np.float64)

    # 5. Post-clipping British tone stack (recover 4x12 chest weight & scoop boxiness after clipping)
    bass_boost = 5.2 if mode in ("pm1", "pm2", "pc1") else 4.2
    presence_boost = 1.5 * (1.0 - float(np.clip((midi - 64) / 24.0, 0.0, 1.0)))
    y = _apply_peaking(y, 110.0, 1.15, bass_boost, sr)
    y = _apply_peaking(y, 440.0, 1.25, -3.0, sr)
    y = _apply_peaking(y, 1750.0, 1.20, 2.6, sr)
    if presence_boost > 0.05:
        y = _apply_peaking(y, 3100.0, 1.60, presence_boost, sr)

    # 6. 4x12 V30 Cabinet IR + harshness notch + clean top roll-off + gentle bus tape glue
    y = cabinet_filter(y, sr, preset="guitar_4x12", hp_hz=64.0, lp_hz=5250.0, micro_reflections=True, dc_block_hz=35.0)
    y = _apply_peaking(y, 4150.0, 2.60, -1.3, sr)
    y = sosfilt(butter(2, 5900.0, btype="lowpass", fs=sr, output="sos"), y)

    early_a = min(len(y), round(0.008 * sr))
    early_b = min(len(y), round(0.45 * sr))
    rms_early = float(np.sqrt(np.mean(y[early_a:early_b] ** 2))) if early_b > early_a else 1e-8
    ref_bus = max(float(np.max(np.abs(y))) * 0.72, 2.8 * rms_early, 1e-8)
    bus_drive = 1.15 if mode == "v1" else 1.55
    y = saturate(
        y / ref_bus,
        sr,
        drive=bus_drive,
        bias=0.02,
        sag=0.14 if mode != "v1" else 0.05,
        sag_release=0.06,
        oversample=4,
        dc_block_hz=30.0,
    )
    y = sosfilt(butter(2, 24.0, btype="highpass", fs=sr, output="sos"), y)

    if apply_tail_fade:
        fade_s = 0.28 if mode in ("pm1", "pm2", "pc1") else 0.50
        n_fade = min(round(fade_s * sr), len(y) // 3)
        if n_fade >= 4:
            y[-n_fade:] *= 0.5 + 0.5 * np.cos(np.pi * (np.arange(n_fade) + 1) / n_fade)

    return y.astype(np.float64)


def _find_spec_for_midi(midi: int) -> Tuple[StringSpec, int]:
    """Find best string/fret assignment for an arbitrary chord interval MIDI note."""
    candidates = [s for s in STRING_SPECS.values() if midi >= round(69 + 12 * np.log2(s.open_hz))]
    if not candidates:
        spec = STRING_SPECS[7]
        return spec, 0
    best = min(candidates, key=lambda s: abs(midi - round(69 + 12 * np.log2(s.open_hz))))
    open_midi = round(69 + 12 * np.log2(best.open_hz))
    return best, max(0, midi - open_midi)


def synthesize_power_chord(
    note: NoteSpec,
    mode: str,
    *,
    seed: int,
    rr_idx: int = 1,
    stage: str = "full",
    sr: int = SR,
) -> Tuple[np.ndarray, SpectralCanvas]:
    """Synthesize Root + 5th (+ Octave on pc2) and sum BEFORE humbucker RLC + tube stack for real chord roar."""
    rr = _rr_profile(rr_idx)
    duration = duration_for_mode(mode)
    canvas = SpectralCanvas(duration=duration, sr=sr, seed=seed)

    spec_root = STRING_SPECS[note.string_num]
    if mode == "pc1":
        off0, off1 = rr["pc1_offsets"]
        voices = [
            (note.midi, note.freq_hz, spec_root, note.fret, off0, 1.00, 0.0),
            (note.midi + 7, _midi_to_hz(note.midi + 7), *_find_spec_for_midi(note.midi + 7), off1, 0.78, 0.8),
        ]
    else:
        off0, off1, off2 = rr["pc2_offsets"]
        voices = [
            (note.midi, note.freq_hz, spec_root, note.fret, off0, 1.00, 0.0),
            (note.midi + 7, _midi_to_hz(note.midi + 7), *_find_spec_for_midi(note.midi + 7), off1, 0.76, 1.0),
            (note.midi + 12, _midi_to_hz(note.midi + 12), *_find_spec_for_midi(note.midi + 12), off2, 0.56, -0.8),
        ]

    chord_sum = np.zeros(round(duration * sr), dtype=np.float64)
    for v_idx, (v_midi, v_hz, v_spec, v_fret, v_att, v_weight, detune_c) in enumerate(voices):
        v_hz_det = v_hz * (2.0 ** (detune_c / 1200.0))
        params = _string_params(v_midi, v_hz_det, v_spec, v_fret, mode, v_att, rr_idx=rr_idx)
        sub_rng = np.random.default_rng(seed + 101 * (v_idx + 1) + 19 * rr_idx)
        pu_sig, buf_v, _ = plucked_string(
            duration, sr, v_hz_det, rng=sub_rng, tail_fade=0.0, return_polarizations=True, **params
        )
        mech_v = add_heavy_guitar_mechanics(
            canvas, pu_sig, buf_v, v_midi, v_spec, v_fret, mode, sub_rng, t_attack=v_att, rr_idx=rr_idx
        )
        n_min = min(len(chord_sum), len(mech_v))
        chord_sum[:n_min] += v_weight * mech_v[:n_min]

    if stage == "pass_02":
        n_fade = min(round(0.35 * sr), len(chord_sum) // 3)
        chord_sum[-n_fade:] *= 0.5 + 0.5 * np.cos(np.pi * (np.arange(n_fade) + 1) / n_fade)
        return chord_sum, canvas

    pu_out = apply_humbucker_rlc(chord_sum, note.midi, mode, sr)
    final = heavy_rock_amp(pu_out, note.midi, mode, sr, apply_cabinet=True, apply_tail_fade=True)
    return final, canvas


def synthesize_note(
    note: NoteSpec,
    mode: str,
    *,
    seed: int,
    rr_idx: int = 1,
    stage: str = "full",
    sr: int = SR,
) -> Tuple[np.ndarray, SpectralCanvas]:
    """Synthesize any single-note or power-chord heavy rock guitar sample at the requested stage."""
    if mode in ("pc1", "pc2"):
        return synthesize_power_chord(note, mode, seed=seed, rr_idx=rr_idx, stage=stage, sr=sr)

    duration = duration_for_mode(mode)
    canvas = SpectralCanvas(duration=duration, sr=sr, seed=seed)
    rng = np.random.default_rng(seed)
    spec = STRING_SPECS[note.string_num]
    t_attack = 0.008
    params = _string_params(note.midi, note.freq_hz, spec, note.fret, mode, t_attack, rr_idx=rr_idx)

    if stage == "pass_01":
        p1 = {**params, "pulse_ms": 0.28, "noise_amount": 0.22, "wound_ripple": 0.0, "pitch_drop_cents": 0.0}
        raw = plucked_string(duration, sr, note.freq_hz, rng=rng, tail_fade=0.35, **p1)
        return raw, canvas

    pu_sig, buf_v, _ = plucked_string(
        duration, sr, note.freq_hz, rng=rng, tail_fade=0.0, return_polarizations=True, **params
    )
    pre_pickup = add_heavy_guitar_mechanics(
        canvas, pu_sig, buf_v, note.midi, spec, note.fret, mode, rng, t_attack=t_attack, rr_idx=rr_idx
    )
    if stage == "pass_02":
        pu_pass2 = apply_humbucker_rlc(pre_pickup, note.midi, mode, sr)
        n_fade = min(round(0.35 * sr), len(pu_pass2) // 3)
        pu_pass2[-n_fade:] *= 0.5 + 0.5 * np.cos(np.pi * (np.arange(n_fade) + 1) / n_fade)
        return pu_pass2, canvas

    pu_sig_filtered = apply_humbucker_rlc(pre_pickup, note.midi, mode, sr)
    if stage == "pass_03":
        amp_raw = heavy_rock_amp(pu_sig_filtered, note.midi, mode, sr, apply_cabinet=False, apply_tail_fade=False)
        n_fade = min(round(0.45 * sr), len(amp_raw) // 3)
        amp_raw[-n_fade:] *= 0.5 + 0.5 * np.cos(np.pi * (np.arange(n_fade) + 1) / n_fade)
        return amp_raw, canvas

    final = heavy_rock_amp(pu_sig_filtered, note.midi, mode, sr, apply_cabinet=True, apply_tail_fade=True)
    return final, canvas


def run_refinement_passes(out_dir: Path) -> None:
    """Render iterative Pass 1..4 progression on hero E2 (v2) note and export progression MP4."""
    ref_dir = out_dir / "refinement"
    ref_dir.mkdir(parents=True, exist_ok=True)
    by_name = {n.name: n for n in NOTE_SPECS}
    hero_note = by_name["E2"]
    hero_seed = 6602

    canvas = SpectralCanvas(duration=SUSTAIN_DURATION, sr=SR, seed=hero_seed)
    stages = [
        ("pass_01", "pass_01", "Pass 1: Raw Dual-Coil Bridge Humbucker Waveguide"),
        ("pass_02", "pass_02", "Pass 2: Pick/Fret Foley, Mahogany Body & Humbucker RLC"),
        ("pass_03", "pass_03", "Pass 3: 3-Stage Cascaded Tube Preamp + EL34 Power Sag"),
        ("full", "pass_04", "Pass 4: British Tone Stack + 4x12 V30 Cabinet IR"),
    ]
    for stage_key, fname, title in stages:
        if fname != "pass_01":
            canvas.reset(seed=hero_seed)
        audio, _ = synthesize_note(hero_note, "v2", seed=hero_seed, stage=stage_key)
        diag = canvas.render_mix(
            audio,
            str(ref_dir / f"{fname}.wav"),
            str(ref_dir / f"{fname}.png"),
            title,
            target_peak_db=-1.2,
            subtype="PCM_24",
        )
        print(f"{fname}:", json.dumps(diag))

    if shutil.which("ffmpeg"):
        mp4_path = canvas.export_progression_video(str(ref_dir / "hero_E2_heavy_progression.mp4"))
        print("Exported progression MP4:", mp4_path)


def run_test_register(out_dir: Path) -> None:
    """Render 8 cross-register heavy rock prototypes covering A1..F#6, chugs, and power chords."""
    proto_dir = out_dir / "prototypes"
    proto_dir.mkdir(parents=True, exist_ok=True)
    by_name = {n.name: n for n in NOTE_SPECS}
    proto_specs = [
        ("proto_Bb1_v2", by_name["Bb1"], "v2"),
        ("proto_E2_v1", by_name["E2"], "v1"),
        ("proto_E2_v2", by_name["E2"], "v2"),
        ("proto_E2_pm1", by_name["E2"], "pm1"),
        ("proto_E2_pc1", by_name["E2"], "pc1"),
        ("proto_E2_pc2", by_name["E2"], "pc2"),
        ("proto_E5_v2", by_name["E5"], "v2"),
        ("proto_E6_v2", by_name["E6"], "v2"),
    ]

    summary = []
    for idx, (label, note, mode) in enumerate(proto_specs):
        seed = 7100 + note.midi * 13 + idx
        audio, canvas = synthesize_note(note, mode, seed=seed, stage="full")
        wav_path = proto_dir / f"{label}.wav"
        png_path = proto_dir / f"{label}_spec.png"
        diag = canvas.render_mix(
            audio,
            str(wav_path),
            str(png_path),
            f"{label} ({note.name} / {mode})",
            target_peak_db=target_peak_for_mode(mode),
            subtype="PCM_24",
        )
        f_meas, cents = _measure_note_pitch(audio, note, mode)
        row = {
            "label": label,
            "note": note.name,
            "midi": note.midi,
            "mode": mode,
            "target_hz": note.freq_hz,
            "measured_hz": f_meas,
            "pitch_error_cents": cents,
            "crest_factor_db": diag["crest_factor_db"],
            "weighted_centroid_hz": diag["weighted_centroid_hz"],
            "band_energy_pct": diag["band_energy_pct"],
            "tail_rms_db": diag["tail_rms_db"],
            "clipped": diag["clipped"],
            "wav_path": str(wav_path),
            "spec_path": str(png_path),
        }
        summary.append(row)
        print(json.dumps(row))


def _sample_relpath(note: NoteSpec, mode: str, rr_idx: int = 1, round_robins: int = 1) -> str:
    suffix = f"_rr{rr_idx}" if round_robins > 1 else ""
    if mode in ("v1", "v2"):
        return f"samples/sustained/egtr_hr_sust_{mode}_{note.midi}_{note.name}{suffix}.wav"
    if mode in ("pm1", "pm2"):
        return f"samples/palm_mute/egtr_hr_{mode}_{note.midi}_{note.name}{suffix}.wav"
    return f"samples/power_chord/egtr_hr_{mode}_{note.midi}_{note.name}{suffix}.wav"


def _synth_worker(task: Tuple[NoteSpec, str, int, int, int, str]) -> dict:
    """Headless worker function for parallel round-robin sample rendering."""
    note, mode, rr_idx, round_robins, seed, out_dir_str = task
    out_dir = Path(out_dir_str)
    rel = _sample_relpath(note, mode, rr_idx, round_robins)
    wav_path = out_dir / rel
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    audio, canvas = synthesize_note(note, mode, seed=seed, rr_idx=rr_idx, stage="full")
    diag = canvas.render_mix(
        audio,
        str(wav_path),
        None,
        None,
        target_peak_db=target_peak_for_mode(mode),
        subtype="PCM_24",
    )
    f_meas, cents = _measure_note_pitch(audio, note, mode)
    return {
        "note": note.name,
        "midi": note.midi,
        "string": note.string_num,
        "fret": note.fret,
        "articulation": mode,
        "rr": rr_idx,
        "target_hz": note.freq_hz,
        "measured_hz": f_meas,
        "pitch_error_cents": cents,
        "wav_relpath": rel,
        "diagnostics": diag,
    }


def _root_key_bounds(notes: List[NoteSpec], idx: int, *, upper_max: int) -> Tuple[int, int]:
    note = notes[idx]
    lokey = 33 if idx == 0 else note.midi - 1
    hikey = upper_max if idx == len(notes) - 1 else note.midi + 1
    return lokey, hikey


def _build_rr_regions(
    notes: List[NoteSpec],
    *,
    upper_max: int,
    layer_fn,
    round_robins: int,
) -> List[dict]:
    """Expand note/velocity definitions across `1..round_robins` SFZ `seq_position` slots."""
    regions: List[dict] = []
    for idx, note in enumerate(notes):
        lokey, hikey = _root_key_bounds(notes, idx, upper_max=upper_max)
        for mode, lovel, hivel in layer_fn(note):
            for rr_idx in range(1, round_robins + 1):
                reg = {
                    "sample": _sample_relpath(note, mode, rr_idx, round_robins),
                    "pitch_keycenter": note.midi,
                    "lokey": lokey,
                    "hikey": hikey,
                    "lovel": lovel,
                    "hivel": hivel,
                }
                if round_robins > 1:
                    reg["seq_length"] = round_robins
                    reg["seq_position"] = rr_idx
                regions.append(reg)
    return regions


def _write_sfz_and_sf2_patches(out_dir: Path, round_robins: int = 4) -> List[str]:
    """Export 5 SFZ patches (with round robins + keyswitched master) and 24-bit `heavy_rock_guitar.sf2`."""
    all_notes = list(NOTE_SPECS)
    riff_notes = [n for n in NOTE_SPECS if n.midi in RIFF_MIDIS]

    lead_regions = _build_rr_regions(
        all_notes, upper_max=90, layer_fn=lambda _n: (("v1", 1, 72), ("v2", 73, 127)), round_robins=round_robins
    )
    pm_regions = _build_rr_regions(
        riff_notes, upper_max=65, layer_fn=lambda _n: (("pm1", 1, 84), ("pm2", 85, 127)), round_robins=round_robins
    )
    pc_regions = _build_rr_regions(
        riff_notes, upper_max=65, layer_fn=lambda _n: (("pc1", 1, 78), ("pc2", 79, 127)), round_robins=round_robins
    )
    combo_regions = _build_rr_regions(
        all_notes,
        upper_max=90,
        layer_fn=lambda n: (("pm1", 1, 64), ("pm2", 65, 96), ("v2", 97, 127))
        if n.midi in RIFF_MIDIS
        else (("v1", 1, 72), ("v2", 73, 127)),
        round_robins=round_robins,
    )

    rr_tag = f"{round_robins}x RR" if round_robins > 1 else "1x"
    write_sfz(
        out_dir / "heavy_rock_lead.sfz", lead_regions,
        attack=0.001, release=0.13, amp_veltrack=40.0,
        header_comment=f"AudioShmaudio Heavy Rock Guitar - Lead/Riff Single Notes (A1-F#6, v1/v2, {rr_tag})",
    )
    write_sfz(
        out_dir / "heavy_rock_palm_mute.sfz", pm_regions,
        attack=0.001, release=0.075, amp_veltrack=55.0,
        header_comment=f"AudioShmaudio Heavy Rock Guitar - Palm Mute (A1-F4, pm1/pm2, {rr_tag})",
    )
    write_sfz(
        out_dir / "heavy_rock_power_chord.sfz", pc_regions,
        attack=0.001, release=0.12, amp_veltrack=45.0,
        header_comment=f"AudioShmaudio Heavy Rock Guitar - Amp Power Chords (A1-F4, pc1/pc2, {rr_tag})",
    )
    write_sfz(
        out_dir / "heavy_rock_combo.sfz", combo_regions,
        attack=0.001, release=0.11, amp_veltrack=42.0,
        header_comment=f"AudioShmaudio Heavy Rock Guitar - Velocity-Switched Combo (A1-F#6, {rr_tag})",
    )
    write_sfz(
        out_dir / "heavy_rock_guitar_master.sfz",
        groups=[
            {
                "opcodes": {"sw_last": 24, "sw_label": "Heavy Rock Lead", "ampeg_release": 0.13, "amp_veltrack": 40.0},
                "regions": lead_regions,
            },
            {
                "opcodes": {"sw_last": 25, "sw_label": "Rock Palm Mute", "ampeg_release": 0.075, "amp_veltrack": 55.0},
                "regions": pm_regions,
            },
            {
                "opcodes": {"sw_last": 26, "sw_label": "Rock PowerChord", "ampeg_release": 0.12, "amp_veltrack": 45.0},
                "regions": pc_regions,
            },
            {
                "opcodes": {"sw_last": 27, "sw_label": "Heavy Rock Combo", "ampeg_release": 0.11, "amp_veltrack": 42.0},
                "regions": combo_regions,
            },
        ],
        attack=0.001,
        release=0.12,
        amp_veltrack=42.0,
        global_opcodes={"sw_lokey": 24, "sw_hikey": 27, "sw_default": 24},
        header_comment=(
            f"AudioShmaudio Heavy Rock Guitar Master Keyswitched SFZ (C1-D#1 switches, A1-F#6, {rr_tag})"
        ),
    )

    def _rr1_only(regs: List[dict]) -> List[dict]:
        return [r for r in regs if int(r.get("seq_position", 1)) == 1]

    write_sf2(
        out_dir / "heavy_rock_guitar.sf2",
        [
            {
                "name": "Heavy Rock Lead",
                "bank": 0,
                "program": 0,
                "regions": _rr1_only(lead_regions),
                "attack": 0.001,
                "release": 0.13,
                "amp_veltrack": 40.0,
            },
            {
                "name": "Rock Palm Mute",
                "bank": 0,
                "program": 1,
                "regions": _rr1_only(pm_regions),
                "attack": 0.001,
                "release": 0.075,
                "amp_veltrack": 55.0,
            },
            {
                "name": "Rock PowerChord",
                "bank": 0,
                "program": 2,
                "regions": _rr1_only(pc_regions),
                "attack": 0.001,
                "release": 0.12,
                "amp_veltrack": 45.0,
            },
            {
                "name": "Heavy Rock Combo",
                "bank": 0,
                "program": 3,
                "regions": _rr1_only(combo_regions),
                "attack": 0.001,
                "release": 0.11,
                "amp_veltrack": 42.0,
            },
        ],
        bank_name="AudioShmaudio Heavy Rock Guitar",
        author="AudioShmaudio",
        comment="7-string/24-fret Heavy Rock Guitar (A1-F#6): Lead, Palm Mute, Amp Power Chords, Combo",
        bit_depth=24,
        sample_root=out_dir,
    )

    return [
        "heavy_rock_lead.sfz",
        "heavy_rock_palm_mute.sfz",
        "heavy_rock_power_chord.sfz",
        "heavy_rock_combo.sfz",
        "heavy_rock_guitar_master.sfz",
    ]


def _vlq(val: int) -> bytes:
    buf = [val & 0x7F]
    val >>= 7
    while val:
        buf.append((val & 0x7F) | 0x80)
        val >>= 7
    return bytes(reversed(buf))


def build_sf2_demo(out_dir: Path) -> Optional[dict]:
    """Render a stereo heavy rock riff + 24-fret high lead solo directly through FluidSynth & the .sf2."""
    fs_bin = shutil.which("fluidsynth")
    sf2_path = out_dir / "heavy_rock_guitar.sf2"
    if not fs_bin or not sf2_path.is_file():
        return None

    demo_dir = out_dir / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    bpm = 132.0
    beat = 60.0 / bpm
    tpb = 480
    tempo_us = round(60_000_000 / bpm)

    events: List[Tuple[float, bytes]] = [
        # Ch 0: Prog 2 Rock PowerChord (left rhythm, pan 28)
        (0.0, bytes([0xC0, 2])), (0.0, bytes([0xB0, 10, 28])), (0.0, bytes([0xB0, 7, 108])),
        # Ch 1: Prog 3 Heavy Rock Combo (right double-track rhythm, pan 100)
        (0.0, bytes([0xC1, 3])), (0.0, bytes([0xB1, 10, 100])), (0.0, bytes([0xB1, 7, 108])),
        # Ch 2: Prog 0 Heavy Rock Lead (center lead solo, pan 64)
        (0.0, bytes([0xC2, 0])), (0.0, bytes([0xB2, 10, 64])), (0.0, bytes([0xB2, 7, 104])),
    ]

    def add_note(ch: int, midi_note: int, start_b: float, dur_b: float, vel: int, *, offset_sec: float = 0.0) -> None:
        t_on = 0.08 + start_b * beat + offset_sec
        t_off = t_on + dur_b * beat
        events.append((t_on, bytes([0x90 | ch, midi_note, vel])))
        events.append((t_off, bytes([0x80 | ch, midi_note, 0])))

    def add_pitch_bend(ch: int, t_sec: float, val: int) -> None:
        clamped = max(0, min(16383, int(val)))
        events.append((t_sec, bytes([0xE0 | ch, clamped & 0x7F, (clamped >> 7) & 0x7F])))

    # Bar 1-4 (0..16 beats): Double-tracked A1..D2..E2 heavy palm-mute gallop + power-chord slams
    # Ch 0 (left) = Prog 2 PowerChord (pc1 vel 66..74, pc2 vel 112..120)
    # Ch 1 (right, +0.011 s double-track) = Prog 3 Combo (pm1 vel 54..62, pm2 vel 80..90, v2 vel 112..120)
    riff_hits = [
        # Bar 1: Bb1 (34) / A1 (33) heavy chugs + C2 (36) / D2 (38) slams
        (0.0, 34, 0.42, 70, 58), (0.5, 34, 0.42, 68, 56), (1.0, 34, 0.85, 116, 116),
        (2.0, 34, 0.42, 72, 84), (2.5, 34, 0.42, 70, 58), (3.0, 36, 0.85, 114, 114),
        # Bar 2: A1 (33) lowest drop-note + D2 (38) + E2 (40) ascending groove
        (4.0, 33, 0.42, 72, 60), (4.5, 33, 0.42, 68, 56), (5.0, 33, 0.85, 118, 118),
        (6.0, 38, 0.42, 74, 86), (6.5, 38, 0.42, 70, 58), (7.0, 40, 0.45, 74, 88), (7.5, 43, 0.45, 116, 116),
        # Bar 3-4: E2 / G2 / A2 / C2 chunky riff into A1 resolve
        (8.0, 40, 0.44, 74, 86), (8.5, 40, 0.42, 68, 58), (9.0, 40, 0.42, 72, 84),
        (9.5, 40, 0.44, 114, 114), (10.0, 40, 0.42, 68, 58), (10.5, 43, 0.85, 116, 116),
        (11.5, 40, 0.42, 70, 60),
        (12.0, 45, 0.90, 118, 118), (13.0, 36, 0.42, 72, 84), (13.5, 36, 0.42, 68, 58),
        (14.0, 38, 0.90, 116, 116), (15.0, 33, 1.40, 120, 120),
    ]
    for b_pos, m, dur_b, vel_l, vel_r in riff_hits:
        add_note(0, m, b_pos, dur_b, vel_l)
        add_note(1, m, b_pos, dur_b, vel_r, offset_sec=0.011)

    # Bar 5-8 (16..32 beats): Wide stereo power chord + palm-mute double-track under 24-fret solo
    for b_pos, m, dur_b, vel in (
        (16.0, 40, 3.8, 114), (20.0, 48, 3.8, 116), (24.0, 43, 1.9, 112),
        (26.0, 47, 1.9, 116), (28.0, 40, 7.8, 120),
    ):
        add_note(0, m, b_pos, dur_b, vel)
        add_note(1, m, b_pos, 0.42, 86, offset_sec=0.011)
        add_note(1, m, b_pos + 0.50, 0.42, 58, offset_sec=0.011)
        add_note(1, m, b_pos + 1.00, 0.42, 88, offset_sec=0.011)
        add_note(1, m, b_pos + 1.50, 0.42, 60, offset_sec=0.011)

    solo_notes = [
        (16.0, 64, 0.92, 62, False), (17.0, 67, 0.92, 92, False), (18.0, 71, 0.92, 104, False), (19.0, 76, 0.92, 114, False),
        (20.0, 79, 0.70, 116, False), (20.75, 81, 0.70, 118, False), (21.5, 83, 0.95, 120, False), (22.5, 86, 1.40, 122, True),
        (24.0, 88, 0.92, 124, False), (25.0, 90, 1.40, 126, True), (26.5, 88, 0.70, 116, False), (27.25, 83, 0.70, 112, False),
        (28.0, 88, 7.80, 125, True),
    ]
    for b_pos, m, dur_b, vel, has_vib in solo_notes:
        t_on = 0.08 + b_pos * beat
        t_off = t_on + dur_b * beat
        add_pitch_bend(2, max(0.0, t_on - 0.002), 8192)
        add_note(2, m, b_pos, dur_b, vel)
        if has_vib:
            dt = 0.15
            while t_on + dt < t_off - 0.04:
                ramp = min(1.0, max(0.0, (dt - 0.15) / 0.25))
                pb = round(8192 + 901.0 * ramp * float(np.sin(2.0 * np.pi * 5.6 * dt)))
                add_pitch_bend(2, t_on + dt, pb)
                dt += 0.015
            add_pitch_bend(2, t_off + 0.01, 8192)

    ticks_list = sorted(((max(0, round(t_s * (bpm / 60.0) * tpb)), idx, msg) for idx, (t_s, msg) in enumerate(events)))
    track = bytearray()
    track.extend(_vlq(0) + b"\xFF\x51\x03" + tempo_us.to_bytes(3, "big"))
    last_t = 0
    for tk, _, msg in ticks_list:
        track.extend(_vlq(tk - last_t) + msg)
        last_t = tk
    track.extend(_vlq(round(2.5 * tpb)) + b"\xFF\x2F\x00")

    midi_bytes = b"MThd" + struct.pack(">IHHH", 6, 0, 1, tpb) + b"MTrk" + struct.pack(">I", len(track)) + bytes(track)
    demo_wav = demo_dir / "heavy_rock_sf2_demo.wav"
    demo_png = demo_dir / "heavy_rock_sf2_demo_spec.png"

    with tempfile.TemporaryDirectory() as td:
        mid_path = Path(td) / "heavy_demo.mid"
        mid_path.write_bytes(midi_bytes)
        proc = subprocess.run(
            [fs_bin, "-ni", "-a", "file", "-F", str(demo_wav), "-T", "wav", "-O", "s24", "-r", "44100", "-g", "0.42", "-R", "1", "-C", "0", str(sf2_path), str(mid_path)],
            capture_output=True, text=True, check=True,
        )
        fs_issues = [
            ln for ln in (proc.stderr or "").splitlines()
            if ("fluidsynth: error:" in ln.lower() or "fluidsynth: warning:" in ln.lower())
            and "no preset found on channel 9" not in ln.lower()
        ]
        assert not fs_issues, f"FluidSynth error/warning: {fs_issues}"

    cli_script = ROOT / "skills/audio-shmaudio/scripts/cli.py"
    env = {**os.environ, "MPLCONFIGDIR": "/tmp/matplotlib"}
    res = subprocess.run(
        [sys.executable, str(cli_script), "analyze", str(demo_wav), "--out", str(demo_png)],
        env=env, capture_output=True, text=True, check=True,
    )
    diag = json.loads(res.stdout)
    print("Heavy Rock SF2 Demo:", json.dumps(diag))
    return diag


def _resample_semitones(audio: np.ndarray, semitones: float) -> np.ndarray:
    """High-quality pitch shift of a mono sample by `semitones` via 4x upsampled cubic interpolation."""
    if abs(semitones) < 1e-6:
        return audio.astype(np.float64, copy=True)
    rate = 2.0 ** (semitones / 12.0)
    up = resample_poly(audio.astype(np.float64), 4, 1)
    new_len = max(1, int(round(len(audio) / rate)))
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


def _parse_sfz_regions(sfz_path: Path) -> List[dict]:
    """Parse standalone `<region>` entries from an exported SFZ file."""
    regions: List[dict] = []
    cur: Optional[dict] = None
    int_keys = {"pitch_keycenter", "lokey", "hikey", "lovel", "hivel", "seq_length", "seq_position"}
    for raw in sfz_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line == "<region>":
            if cur is not None:
                regions.append(cur)
            cur = {}
            continue
        if line.startswith("<"):
            if cur is not None:
                regions.append(cur)
                cur = None
            continue
        if cur is not None and "=" in line:
            k, v = line.split("=", 1)
            cur[k.strip()] = int(v.strip()) if k.strip() in int_keys else v.strip()
    if cur is not None:
        regions.append(cur)
    return regions


def build_sfz_rr_demo(out_dir: Path, round_robins: int = 4) -> dict:
    """Render an SFZ round-robin showcase demo cycling `seq_position` 1..N across double-tracked chugs & riffs."""
    pm_regs = _parse_sfz_regions(out_dir / "heavy_rock_palm_mute.sfz")
    pc_regs = _parse_sfz_regions(out_dir / "heavy_rock_power_chord.sfz")
    lead_regs = _parse_sfz_regions(out_dir / "heavy_rock_lead.sfz")

    wav_cache: Dict[str, np.ndarray] = {}
    shifted_cache: Dict[Tuple[str, int], np.ndarray] = {}
    counters: Dict[Tuple[str, int, int], int] = {}

    def _trigger(
        buf: np.ndarray,
        reg_pool: List[dict],
        lane_key: str,
        midi: int,
        vel: int,
        t_sec: float,
        dur_sec: float,
        gain: float,
        *,
        rr_offset: int = 0,
        release_sec: float = 0.045,
    ) -> None:
        matches = [
            r for r in reg_pool
            if r["lokey"] <= midi <= r["hikey"] and r["lovel"] <= vel <= r["hivel"]
        ]
        if not matches:
            return
        vel_bucket = matches[0]["lovel"]
        ckey = (lane_key, midi, vel_bucket)
        hit_num = counters.get(ckey, rr_offset)
        counters[ckey] = hit_num + 1
        seq_pos = (hit_num % max(1, round_robins)) + 1
        reg = next((r for r in matches if r.get("seq_position", 1) == seq_pos), matches[0])

        rel = reg["sample"]
        if rel not in wav_cache:
            data, fs = sf.read(out_dir / rel, dtype="float64")
            assert fs == SR
            wav_cache[rel] = data

        semi = int(midi - reg["pitch_keycenter"])
        skey = (rel, semi)
        if skey not in shifted_cache:
            shifted_cache[skey] = _resample_semitones(wav_cache[rel], float(semi))
        sample = shifted_cache[skey]

        n_hold = round(dur_sec * SR)
        n_rel = max(8, round(release_sec * SR))
        max_len = min(len(sample), n_hold + n_rel)
        clip = sample[:max_len].copy()
        if max_len > n_hold:
            fade_len = max_len - n_hold
            clip[n_hold:] *= 0.5 + 0.5 * np.cos(np.pi * (np.arange(fade_len) + 1) / fade_len)
        else:
            tail = min(round(0.025 * SR), len(clip) // 3)
            if tail >= 4:
                clip[-tail:] *= 0.5 + 0.5 * np.cos(np.pi * (np.arange(tail) + 1) / tail)

        start = max(0, round(t_sec * SR))
        end = min(len(buf), start + len(clip))
        if end > start:
            buf[start:end] += gain * clip[: end - start]

    duration = 11.8
    canvas = SpectralCanvas(duration=duration, sr=SR, seed=909)
    rng = np.random.default_rng(909)
    left_gtr = canvas.new_layer()
    right_gtr = canvas.new_layer()
    lead_gtr = canvas.new_layer()

    bpm = 136.0
    beat = 60.0 / bpm
    sixteenth = beat * 0.25
    t0 = 0.08

    # Section 1 (0.08..3.61 s): Rapid 16th-note palm-mute gallop exposing 4x Down/Up RR alternation cleanly
    gallop_blocks = [
        (40, [110, 62, 68, 56, 108, 60, 70, 58, 112, 64, 66, 55, 114, 62, 72, 60]),  # 1 bar E2 16ths
        (34, [112, 60, 68, 58, 110, 62, 70, 56, 112, 64, 68, 58, 98, 62, 116, 0]),    # 1 bar Bb1 + E2 transition
    ]
    step = 0
    for midi_root, vels in gallop_blocks:
        for idx_16, vel in enumerate(vels):
            t_hit = t0 + step * sixteenth
            step += 1
            if vel <= 0:
                continue
            j_l = float(rng.uniform(-0.002, 0.002))
            j_r = float(rng.uniform(0.007, 0.011))
            dur = 0.28 * beat if idx_16 % 4 == 0 else 0.19 * beat
            if vel >= 108:
                _trigger(left_gtr.audio, pc_regs, "pc_L", midi_root, 72, t_hit + j_l, dur, 0.78, rr_offset=0)
                _trigger(right_gtr.audio, pm_regs, "pm_R", midi_root, 106, t_hit + j_r, dur, 0.80, rr_offset=2)
            else:
                _trigger(left_gtr.audio, pm_regs, "pm_L", midi_root, vel, t_hit + j_l, dur, 0.78, rr_offset=0)
                _trigger(right_gtr.audio, pm_regs, "pm_R", midi_root, vel, t_hit + j_r, dur, 0.78, rr_offset=2)

    # Section 2 (beats 8.0..24.0): Double-tracked A1..D2..E2 heavy riff + repeated-note 4x RR high E5..E6 lead
    t_s2 = t0 + 8.0 * beat
    sec2_riffs = [
        (0.0, 34, 0.42, 72, 60), (0.5, 34, 0.42, 68, 58), (1.0, 34, 0.85, 116, 116),
        (2.0, 33, 0.42, 72, 60), (2.5, 33, 0.42, 68, 58), (3.0, 36, 0.85, 116, 114),
        (4.0, 38, 0.42, 74, 88), (4.5, 38, 0.42, 70, 58), (5.0, 40, 0.45, 74, 90), (5.5, 43, 0.45, 116, 116),
        (6.0, 40, 0.44, 74, 88), (6.5, 40, 0.42, 68, 58), (7.0, 40, 0.42, 72, 86), (7.5, 33, 0.85, 118, 118),
        (8.5, 40, 3.60, 118, 118), (12.5, 34, 3.80, 118, 118),
    ]
    for b_pos, midi_n, dur_b, vel_l, vel_r in sec2_riffs:
        t_h = t_s2 + b_pos * beat
        _trigger(left_gtr.audio, pc_regs, "pc_L", midi_n, vel_l, t_h, dur_b * beat, 0.80, rr_offset=0)
        if vel_r >= 100:
            _trigger(right_gtr.audio, pc_regs, "pc_R", midi_n, vel_r, t_h + 0.010, dur_b * beat, 0.78, rr_offset=2)
        else:
            _trigger(right_gtr.audio, pm_regs, "pm_R", midi_n, vel_r, t_h + 0.010, dur_b * beat, 0.80, rr_offset=2)

    lead_hits = [
        (8.50, 76, 0.24, 110), (8.75, 76, 0.24, 112), (9.00, 76, 0.24, 114), (9.25, 76, 0.45, 116),
        (9.75, 79, 0.45, 116), (10.25, 83, 0.68, 118), (11.00, 86, 0.68, 120),
        (11.75, 88, 0.24, 118), (12.00, 88, 0.24, 120), (12.25, 88, 0.24, 122), (12.50, 88, 3.60, 124),
    ]
    for b_pos, midi_n, dur_b, vel_l in lead_hits:
        _trigger(lead_gtr.audio, lead_regs, "lead", midi_n, vel_l, t_s2 + b_pos * beat, dur_b * beat, 0.74, release_sec=0.11)

    scene = canvas.new_scene()
    scene.add(left_gtr, pan=-0.72, distance=0.01, gain=1.0)
    scene.add(right_gtr, pan=0.72, distance=0.01, gain=1.0)
    scene.add(lead_gtr, pan=0.08, distance=0.02, gain=1.0)
    scene.apply_room(size=0.22, damping=0.58, wet=0.11, predelay=0.008, early_reflections=6)
    scene.soft_limit(ceiling_db=-1.0, drive=1.05)
    scene.fade_edges(fade_in=0.004, fade_out=0.65)

    demo_wav = out_dir / "demo/heavy_rock_rr_demo.wav"
    demo_png = out_dir / "demo/heavy_rock_rr_demo_spec.png"
    diag = canvas.render_mix(
        scene.audio,
        str(demo_wav),
        str(demo_png),
        f"Heavy Rock Guitar {round_robins}x Round-Robin SFZ Demo (Stereo)",
        subtype="PCM_24",
    )
    print("Heavy Rock RR SFZ Demo:", json.dumps(diag))
    return diag


def run_full_pack(out_dir: Path, round_robins: int = 4) -> None:
    """Render all round-robin WAVs, representative PNGs, 5 SFZ patches, SF2 bank, and FluidSynth + SFZ RR demos."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("samples/sustained", "samples/palm_mute", "samples/power_chord"):
        subdir = out_dir / sub
        subdir.mkdir(parents=True, exist_ok=True)
        for old_wav in subdir.glob("*.wav"):
            old_wav.unlink()

    tasks: List[Tuple[NoteSpec, str, int, int, int, str]] = []
    for note in NOTE_SPECS:
        modes = ["v1", "v2"] + (["pm1", "pm2", "pc1", "pc2"] if note.midi in RIFF_MIDIS else [])
        for m_idx, mode in enumerate(modes):
            for rr_idx in range(1, round_robins + 1):
                seed = 8000 + note.midi * 131 + m_idx * 17 + rr_idx * 7
                tasks.append((note, mode, rr_idx, round_robins, seed, str(out_dir)))

    max_workers = min(len(tasks), max(1, (os.cpu_count() or 4) - 2))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        entries = list(pool.map(_synth_worker, tasks))

    demo_dir = out_dir / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    rep_set = {
        ("Bb1", "v2"), ("E2", "v1"), ("E2", "v2"), ("E2", "pm1"),
        ("E2", "pm2"), ("E2", "pc1"), ("E2", "pc2"), ("E5", "v2"), ("E6", "v2"),
    }
    for e in entries:
        print(
            f"Rendered {e['wav_relpath']:54s} | err={str(e['pitch_error_cents']):>6s} c | "
            f"centroid={e['diagnostics']['weighted_centroid_hz']:>6.1f} Hz | "
            f"crest={e['diagnostics']['crest_factor_db']:>5.2f} dB"
        )
        if e["rr"] == 1 and (e["note"], e["articulation"]) in rep_set:
            wav_p = out_dir / e["wav_relpath"]
            png_p = demo_dir / f"spec_{e['note']}_{e['articulation']}.png"
            data, _ = sf.read(wav_p, dtype="float64")
            canvas_plot = SpectralCanvas(duration=duration_for_mode(e["articulation"]), sr=SR, seed=0)
            canvas_plot.render_mix(
                data,
                str(wav_p),
                str(png_p),
                f"{e['note']} ({e['articulation']})",
                target_peak_db=target_peak_for_mode(e["articulation"]),
                subtype="PCM_24",
            )

    sfz_files = _write_sfz_and_sf2_patches(out_dir, round_robins=round_robins)
    demo_diag = build_sf2_demo(out_dir)
    rr_demo_diag = build_sfz_rr_demo(out_dir, round_robins=round_robins)

    manifest = {
        "sample_rate": SR,
        "subtype": "PCM_24",
        "channels": 1,
        "round_robins": round_robins,
        "total_samples": len(entries),
        "sustained_count": sum(1 for e in entries if e["articulation"] in ("v1", "v2")),
        "palm_mute_count": sum(1 for e in entries if e["articulation"] in ("pm1", "pm2")),
        "power_chord_count": sum(1 for e in entries if e["articulation"] in ("pc1", "pc2")),
        "sfz_files": sfz_files,
        "sf2": "heavy_rock_guitar.sf2",
        "samples": entries,
        "demo": demo_diag,
        "rr_demo": rr_demo_diag,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest.json, {len(sfz_files)} SFZ patches ({round_robins}x RR), and heavy_rock_guitar.sf2.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize heavy rock 7-string/24-fret electric guitar SFZ/SF2 pack.")
    parser.add_argument("--refinement-only", action="store_true", help="Render hero E2 Pass 1-4 + MP4.")
    parser.add_argument("--test-register", action="store_true", help="Render 8 cross-register listening prototypes.")
    parser.add_argument("--full", action="store_true", help="Render full round-robin pack, SFZ/SF2, and demos.")
    parser.add_argument("--round-robins", type=int, default=4, help="Round-robin variations per note/layer (>= 1).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output/heavy_rock_guitar",
        help="Output root directory.",
    )
    args = parser.parse_args()
    if args.round_robins < 1:
        parser.error("--round-robins must be >= 1")

    if args.refinement_only:
        run_refinement_passes(args.out_dir)
    if args.test_register:
        run_test_register(args.out_dir)
    if args.full or not (args.refinement_only or args.test_register):
        run_refinement_passes(args.out_dir)
        run_full_pack(args.out_dir, round_robins=args.round_robins)


if __name__ == "__main__":
    main()
