"""Physical-modeling plucked-string synthesis, tube saturation, speaker cabinets, and DAW sampling helpers."""

from __future__ import annotations

import math
import os
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.signal import butter, fftconvolve, lfilter, resample_poly, sosfilt
import soundfile as sf

PickupSpec = Optional[Union[float, Sequence[Tuple[float, float]]]]

CABINET_PRESETS: Dict[str, dict] = {
    "guitar_1x12": {
        "hp_hz": 82.0,
        "lp_hz": 4850.0,
        "ir_sec": 0.050,
        "eq": (
            (112.0, 1.20, 2.8),
            (245.0, 1.10, 2.0),
            (540.0, 1.35, -3.0),
            (2100.0, 1.75, 3.2),
            (3350.0, 2.20, 2.3),
            (4300.0, 2.70, -2.2),
        ),
    },
    "guitar_4x12": {
        "hp_hz": 72.0,
        "lp_hz": 4650.0,
        "ir_sec": 0.050,
        "eq": (
            (102.0, 1.35, 3.6),
            (220.0, 1.10, 1.8),
            (480.0, 1.25, -3.5),
            (1950.0, 1.65, 3.5),
            (3100.0, 2.10, 2.6),
            (4150.0, 2.60, -2.5),
        ),
    },
    "bass_1x15": {
        "hp_hz": 42.0,
        "lp_hz": 3600.0,
        "ir_sec": 0.060,
        "eq": (
            (68.0, 1.25, 3.2),
            (160.0, 1.00, 1.8),
            (420.0, 1.20, -2.2),
            (1450.0, 1.50, 2.4),
            (2700.0, 2.00, -2.0),
        ),
    },
    "radio": {
        "hp_hz": 380.0,
        "lp_hz": 3100.0,
        "ir_sec": 0.040,
        "eq": (
            (680.0, 1.20, -2.0),
            (1450.0, 1.90, 4.2),
            (2400.0, 2.20, 2.5),
        ),
    },
}


def _as_finite_mono(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio)
    if arr.ndim != 1 or arr.dtype.kind not in "fiu" or not np.all(np.isfinite(arr)):
        raise ValueError("Audio must be a finite mono signal")
    if arr.dtype.kind == "i":
        return arr.astype(np.float64) / float(np.iinfo(arr.dtype).max)
    return arr.astype(np.float64, copy=False)


def peaking_biquad(
    freq_hz: float,
    q: float,
    gain_db: float,
    sr: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return Audio EQ Cookbook peaking EQ (b, a) filter coefficients."""
    if sr <= 0:
        raise ValueError("Sample rate must be positive")
    if not np.isfinite(freq_hz) or not 0.0 < freq_hz < sr * 0.5:
        raise ValueError("Filter frequency must be above zero and below Nyquist")
    if not np.isfinite(q) or q <= 0.0:
        raise ValueError("Filter Q must be finite and positive")
    if not np.isfinite(gain_db):
        raise ValueError("Filter gain_db must be finite")

    a_lin = 10.0 ** (float(gain_db) / 40.0)
    w0 = 2.0 * math.pi * float(freq_hz) / sr
    alpha = math.sin(w0) / (2.0 * float(q))
    cos_w0 = math.cos(w0)

    b0 = 1.0 + alpha * a_lin
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a_lin
    a0 = 1.0 + alpha / a_lin
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a_lin
    return (
        np.array([b0, b1, b2], dtype=np.float64) / a0,
        np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64),
    )


def resonant_lowpass(
    audio: np.ndarray,
    sr: int,
    freq_hz: float,
    q: float = 1.8,
) -> np.ndarray:
    """Filter a mono signal through a 2-pole resonant low-pass (models magnetic pickup RLC peak)."""
    x = _as_finite_mono(audio)
    if sr <= 0:
        raise ValueError("Sample rate must be positive")
    if not np.isfinite(freq_hz) or not 0.0 < freq_hz < sr * 0.5:
        raise ValueError("Filter frequency must be above zero and below Nyquist")
    if not np.isfinite(q) or q <= 0.0:
        raise ValueError("Filter Q must be finite and positive")
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)

    w0 = 2.0 * math.pi * float(freq_hz) / sr
    alpha = math.sin(w0) / (2.0 * float(q))
    cos_w0 = math.cos(w0)
    b0 = (1.0 - cos_w0) * 0.5
    b1 = 1.0 - cos_w0
    b2 = (1.0 - cos_w0) * 0.5
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return lfilter(b, a, x)


def saturate(
    audio: np.ndarray,
    sr: int,
    *,
    drive: float = 2.0,
    bias: float = 0.14,
    sag: float = 0.0,
    sag_release: float = 0.12,
    oversample: int = 4,
    dc_block_hz: float = 45.0,
) -> np.ndarray:
    """Apply oversampled asymmetric tube-style saturation, coupling-cap DC block, and optional power-amp sag."""
    x = _as_finite_mono(audio)
    if sr <= 0:
        raise ValueError("Sample rate must be positive")
    if not np.isfinite(drive) or drive <= 0.0:
        raise ValueError("Drive must be finite and positive")
    if not np.isfinite(bias):
        raise ValueError("Bias must be finite")
    if not np.isfinite(sag) or sag < 0.0:
        raise ValueError("Sag must be finite and nonnegative")
    if not np.isfinite(sag_release) or sag_release <= 0.0:
        raise ValueError("Sag release must be finite and positive")
    if not isinstance(oversample, (int, np.integer)) or oversample < 1:
        raise ValueError("Oversample factor must be an integer >= 1")
    if not np.isfinite(dc_block_hz) or not 0.0 <= dc_block_hz < sr * 0.45:
        raise ValueError("dc_block_hz must be nonnegative and below Nyquist")

    if x.size == 0:
        return np.zeros(0, dtype=np.float64)

    os_factor = int(oversample)
    x_ov = resample_poly(x, os_factor, 1) if os_factor > 1 else x
    bias_tanh = math.tanh(float(drive) * float(bias))
    slope0 = float(drive) * max(1.0 - bias_tanh * bias_tanh, 1e-4)
    y_ov = (np.tanh(float(drive) * (x_ov + float(bias))) - bias_tanh) / slope0
    y = resample_poly(y_ov, 1, os_factor)[: x.size] if os_factor > 1 else y_ov

    if dc_block_hz > 0.0:
        hp_sos = butter(1, float(dc_block_hz), btype="highpass", fs=sr, output="sos")
        y = sosfilt(hp_sos, y)

    if sag > 0.0:
        r_rel = math.exp(-1.0 / max(float(sag_release) * sr, 1.0))
        env = lfilter([1.0 - r_rel], [1.0, -r_rel], np.abs(y))
        sag_gain = 1.0 / (1.0 + float(sag) * env)
        y = np.tanh(1.15 * y * sag_gain) / 1.15

    return y.astype(np.float64, copy=False)


def build_cabinet_ir(
    sr: int,
    *,
    preset: str = "guitar_1x12",
    lp_hz: Optional[float] = None,
    hp_hz: Optional[float] = None,
    micro_reflections: bool = True,
) -> np.ndarray:
    """Synthesize a zero-latency speaker cabinet impulse response for the chosen preset."""
    if sr <= 0:
        raise ValueError("Sample rate must be positive")
    if preset not in CABINET_PRESETS:
        raise ValueError(f"Unknown cabinet preset {preset!r}; choose from {tuple(CABINET_PRESETS)}")

    cfg = CABINET_PRESETS[preset]
    hp = float(cfg["hp_hz"] if hp_hz is None else hp_hz)
    lp = float(cfg["lp_hz"] if lp_hz is None else lp_hz)
    if not np.isfinite(hp) or not np.isfinite(lp) or not 0.0 < hp < lp:
        raise ValueError("Cabinet high-pass and low-pass cutoffs must satisfy 0 < hp_hz < lp_hz")

    ir_len = max(64, round(float(cfg["ir_sec"]) * sr))
    imp = np.zeros(ir_len, dtype=np.float64)
    imp[0] = 1.0

    y = sosfilt(butter(2, min(hp, sr * 0.42), btype="highpass", fs=sr, output="sos"), imp)
    for freq_hz, q_val, gain_db in cfg["eq"]:
        if freq_hz < sr * 0.46:
            b, a = peaking_biquad(freq_hz, q_val, gain_db, sr)
            y = lfilter(b, a, y)

    y = sosfilt(butter(4, min(lp, sr * 0.45), btype="lowpass", fs=sr, output="sos"), y)

    if micro_reflections:
        refl_len = max(32, round(0.020 * sr))
        refl = np.zeros(refl_len, dtype=np.float64)
        refl[0] = 1.0
        taps = (
            (0.00045, -0.14),
            (0.00115, 0.11),
            (0.00235, -0.09),
            (0.00390, 0.07),
            (0.00620, -0.05),
            (0.00940, 0.035),
            (0.01380, -0.025),
        )
        for delay_s, amp in taps:
            idx = round(delay_s * sr)
            if idx < refl_len:
                refl[idx] += amp
        refl = sosfilt(
            butter(2, min(3600.0, sr * 0.44), btype="lowpass", fs=sr, output="sos"),
            refl,
        )
        refl[0] = max(refl[0], 0.85)
        ir = fftconvolve(y, refl, mode="full")[:ir_len]
    else:
        ir = y

    energy = float(np.sqrt(np.sum(ir * ir)))
    if energy > 1e-12:
        ir /= energy
    return ir


def cabinet_filter(
    audio: np.ndarray,
    sr: int,
    *,
    preset: str = "guitar_1x12",
    lp_hz: Optional[float] = None,
    hp_hz: Optional[float] = None,
    micro_reflections: bool = True,
    dc_block_hz: float = 35.0,
) -> np.ndarray:
    """Filter a mono signal through a physically modeled speaker cabinet impulse response."""
    x = _as_finite_mono(audio)
    if not np.isfinite(dc_block_hz) or not 0.0 <= dc_block_hz < sr * 0.45:
        raise ValueError("dc_block_hz must be nonnegative and below Nyquist")
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)

    ir = build_cabinet_ir(
        sr,
        preset=preset,
        lp_hz=lp_hz,
        hp_hz=hp_hz,
        micro_reflections=micro_reflections,
    )
    out = fftconvolve(x, ir, mode="full")[: x.size]
    if dc_block_hz > 0.0:
        out = sosfilt(butter(1, float(dc_block_hz), btype="highpass", fs=sr, output="sos"), out)
    return out.astype(np.float64, copy=False)


def _solve_loss_filter(
    f0: float,
    t60_low: float,
    t60_hi: float,
    sr: int,
) -> Tuple[float, float, float]:
    """Calibrate 1-pole loop loss filter H(z) = g*(1-a)/(1-a*z^-1) and return (g, a, tau_loss)."""
    f_hi = min(4000.0, max(f0 * 4.0, 2000.0), sr * 0.42)
    g0 = 10.0 ** (-3.0 / max(f0 * t60_low, 1e-4))
    ghi = 10.0 ** (-3.0 / max(f0 * t60_hi, 1e-4))
    r = float(np.clip(ghi / max(g0, 1e-9), 0.10, 0.9995))

    w0 = 2.0 * math.pi * f0 / sr
    wh = 2.0 * math.pi * f_hi / sr
    denom = max(1.0 - r * r, 1e-9)
    c_val = (math.cos(w0) - r * r * math.cos(wh)) / denom
    if c_val >= 1.0:
        a = c_val - math.sqrt(max(c_val * c_val - 1.0, 0.0))
    else:
        a = 0.25
    a = float(np.clip(a, 0.02, 0.96))

    mag_den = math.sqrt(max(1.0 - 2.0 * a * math.cos(w0) + a * a, 1e-12))
    g = float(min(g0 * mag_den / max(1.0 - a, 1e-6), 0.99985))
    tau_loss = float(math.atan2(a * math.sin(w0), 1.0 - a * math.cos(w0)) / w0)
    return g, a, tau_loss


def _ap_phase_delay(c: float, freq_hz: float, sr: int, m_stages: int = 4) -> float:
    w = 2.0 * math.pi * freq_hz / sr
    z_inv = np.exp(-1j * w)
    h1 = (c + z_inv) / (1.0 + c * z_inv)
    phi1 = float(np.angle(h1))
    if phi1 > 0.0:
        phi1 -= 2.0 * math.pi
    return float(-m_stages * phi1 / max(w, 1e-12))


def _solve_dispersion_allpass(
    f0: float,
    stiffness_b: float,
    sr: int,
    m_stages: int = 4,
) -> Tuple[np.ndarray, float]:
    """Calibrate M cascaded 1st-order allpass sections to match Euler-Bernoulli 5th-partial sharpness."""
    target_c5 = 1200.0 * math.log2(
        math.sqrt(1.0 + 25.0 * stiffness_b) / math.sqrt(1.0 + stiffness_b)
    )
    max_tau = sr / f0 - 15.0
    f5 = min(5.0 * f0, sr * 0.44)

    best_err = float("inf")
    best_c = -0.15
    best_tau1 = _ap_phase_delay(best_c, f0, sr, m_stages)
    for c in np.linspace(-0.90, -0.05, 851):
        c_f = float(c)
        td1 = _ap_phase_delay(c_f, f0, sr, m_stages)
        if td1 > max_tau:
            continue
        td5 = _ap_phase_delay(c_f, f5, sr, m_stages)
        d0 = sr / f0 - td1
        if d0 <= 10.0:
            continue
        actual_c5 = 1200.0 * math.log2((sr / f0) / (d0 + td5))
        err = abs(actual_c5 - target_c5)
        if err < best_err:
            best_err = err
            best_c = c_f
            best_tau1 = td1

    c2 = best_c * best_c
    two_c = 2.0 * best_c
    sos = np.array(
        [
            [c2, two_c, 1.0, 1.0, two_c, c2],
            [c2, two_c, 1.0, 1.0, two_c, c2],
        ],
        dtype=np.float64,
    )
    return sos, best_tau1


def _fractional_delay_read(buf: np.ndarray, delay_samples: float) -> np.ndarray:
    """Apply constant fractional delay using passive 4-point cubic Lagrange interpolation."""
    n = buf.shape[0]
    if delay_samples <= 1.0:
        return buf.copy()
    padded = np.pad(buf, (2, 2), mode="constant")
    idx = np.arange(n, dtype=np.float64) + 2.0 - delay_samples
    valid = (idx >= 1.0) & (idx < n + 2.0)
    out = np.zeros(n, dtype=np.float64)
    if np.any(valid):
        pv = idx[valid]
        m = np.floor(pv).astype(np.int64)
        u = pv - m
        c_m1 = -u * (u - 1.0) * (u - 2.0) / 6.0
        c_0 = (u + 1.0) * (u - 1.0) * (u - 2.0) / 2.0
        c_1 = -(u + 1.0) * u * (u - 2.0) / 2.0
        c_2 = (u + 1.0) * u * (u - 1.0) / 6.0
        out[valid] = (
            c_m1 * padded[m - 1]
            + c_0 * padded[m]
            + c_1 * padded[m + 1]
            + c_2 * padded[m + 2]
        )
    return out


def _normalize_pickup_spec(pickup_pos: PickupSpec) -> Tuple[Tuple[float, float], ...]:
    if pickup_pos is None:
        return ()
    if np.isscalar(pickup_pos) and not isinstance(pickup_pos, (bool, str, bytes)):
        mu = float(pickup_pos)
        if not np.isfinite(mu) or not 0.0 < mu < 1.0:
            raise ValueError("Pickup position must be in (0, 1)")
        return ((mu, 1.0),)

    pairs = []
    total_w = 0.0
    for item in pickup_pos:
        if len(item) != 2:
            raise ValueError("Pickup sequence entries must be (fraction, weight) pairs")
        mu, weight = float(item[0]), float(item[1])
        if not np.isfinite(mu) or not 0.0 < mu < 1.0:
            raise ValueError("Pickup position fractions must be in (0, 1)")
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("Pickup weights must be finite and nonnegative")
        if weight > 0.0:
            pairs.append((mu, weight))
            total_w += weight
    if not pairs or total_w <= 0.0:
        raise ValueError("Pickup sequence must contain at least one positive weight")
    return tuple((mu, w / total_w) for mu, w in pairs)


def plucked_string(
    duration: float,
    sr: int,
    pitch_hz: float,
    *,
    t60_low: float = 6.0,
    t60_hi: float = 1.35,
    stiffness: float = 5.0e-5,
    pluck_pos: float = 0.18,
    pickup_pos: PickupSpec = ((0.24, 0.52), (0.155, 0.48)),
    pickup_comb_depth: float = 0.76,
    pulse_ms: float = 0.22,
    vel_cutoff_hz: Optional[float] = None,
    pick_lp_hz: float = 5600.0,
    noise_amount: float = 0.24,
    wound_ripple: float = 0.0,
    dual_pol_detune_cents: float = 0.55,
    bridge_coupling: float = 0.0012,
    cross_pol_mix: float = 0.22,
    pitch_drop_cents: float = 10.0,
    pitch_drop_tau: float = 0.06,
    flux_asymmetry: float = 0.032,
    attack_time: float = 0.004,
    tail_fade: float = 0.35,
    seed: Optional[int] = 0,
    rng: Optional[np.random.Generator] = None,
    return_polarizations: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Synthesize a realistic plucked string via coupled dual-polarization stiff waveguides."""
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("Duration must be finite and positive")
    if sr <= 0:
        raise ValueError("Sample rate must be positive")
    if not np.isfinite(pitch_hz) or not 20.0 <= pitch_hz < sr * 0.22:
        raise ValueError("pitch_hz must be in [20, 0.22 * sr)")
    if not np.isfinite(t60_low) or t60_low <= 0.0 or not np.isfinite(t60_hi) or t60_hi <= 0.0:
        raise ValueError("t60_low and t60_hi must be finite and positive")
    if not np.isfinite(stiffness) or not 0.0 <= stiffness <= 2e-3:
        raise ValueError("Stiffness must be in [0, 2e-3]")
    if not np.isfinite(pluck_pos) or not 0.03 <= pluck_pos <= 0.50:
        raise ValueError("pluck_pos must be in [0.03, 0.50]")
    if not np.isfinite(pickup_comb_depth) or not 0.0 <= pickup_comb_depth <= 1.0:
        raise ValueError("pickup_comb_depth must be in [0, 1]")
    if not np.isfinite(pulse_ms) or pulse_ms <= 0.0:
        raise ValueError("pulse_ms must be finite and positive")
    if not np.isfinite(pick_lp_hz) or pick_lp_hz <= 0.0:
        raise ValueError("pick_lp_hz must be finite and positive")
    if not np.isfinite(noise_amount) or noise_amount < 0.0:
        raise ValueError("noise_amount must be finite and nonnegative")
    if not np.isfinite(wound_ripple) or wound_ripple < 0.0:
        raise ValueError("wound_ripple must be finite and nonnegative")
    if not np.isfinite(bridge_coupling) or not 0.0 <= bridge_coupling <= 0.05:
        raise ValueError("bridge_coupling must be in [0, 0.05]")
    if not np.isfinite(cross_pol_mix) or not 0.0 <= cross_pol_mix <= 1.0:
        raise ValueError("cross_pol_mix must be in [0, 1]")
    if not np.isfinite(pitch_drop_cents) or pitch_drop_cents < 0.0:
        raise ValueError("pitch_drop_cents must be finite and nonnegative")
    if not np.isfinite(pitch_drop_tau) or pitch_drop_tau <= 0.0:
        raise ValueError("pitch_drop_tau must be finite and positive")
    if not np.isfinite(attack_time) or attack_time < 0.0:
        raise ValueError("attack_time must be finite and nonnegative")
    if not np.isfinite(tail_fade) or tail_fade < 0.0:
        raise ValueError("tail_fade must be finite and nonnegative")

    pickup_pairs = _normalize_pickup_spec(pickup_pos)
    gen = np.random.default_rng(seed) if rng is None else rng

    n_samples = max(3, round(duration * sr))
    n_attack = min(round(attack_time * sr), max(0, n_samples - 3))
    f0 = float(pitch_hz)

    # 1. Plectrum velocity excitation burst (pulse width independent of 1/f0).
    exc_len = min(max(32, round(0.055 * sr)), max(32, n_samples - n_attack))
    t_exc = np.arange(exc_len, dtype=np.float64) / sr
    sigma = (float(pulse_ms) * 0.001) * 0.45
    t_peak = min(0.0016, 0.35 * exc_len / sr)
    dt = t_exc - t_peak
    asym = np.where(dt < 0.0, dt / max(0.50 * sigma, 1e-5), dt / max(1.30 * sigma, 1e-5))
    pulse = np.exp(-0.5 * asym * asym)

    raw = pulse
    if noise_amount > 0.0:
        bp_lo = min(420.0, sr * 0.20)
        bp_hi = min(6200.0, sr * 0.44)
        if bp_hi > bp_lo:
            bp_sos = butter(2, [bp_lo, bp_hi], btype="bandpass", fs=sr, output="sos")
            pick_noise = sosfilt(bp_sos, gen.standard_normal(exc_len))
            pick_noise /= max(float(np.max(np.abs(pick_noise))), 1e-8)
            pick_noise *= np.exp(-np.maximum(t_exc - 0.0003, 0.0) / 0.0034)
            raw = raw + float(noise_amount) * pick_noise

    if wound_ripple > 0.0:
        coil_lo = min(2100.0, sr * 0.25)
        coil_hi = min(5600.0, sr * 0.44)
        if coil_hi > coil_lo:
            coil_sos = butter(2, [coil_lo, coil_hi], btype="bandpass", fs=sr, output="sos")
            coil = sosfilt(coil_sos, gen.standard_normal(exc_len))
            coil /= max(float(np.max(np.abs(coil))), 1e-8)
            raw = raw + float(wound_ripple) * coil * np.exp(-t_exc / 0.0045)

    fc_int = float(np.clip(2.0 * f0, 145.0, 540.0) if vel_cutoff_hz is None else vel_cutoff_hz)
    fc_int = min(fc_int, sr * 0.44)
    p_int = math.exp(-2.0 * math.pi * fc_int / sr)
    vel_wave = lfilter([1.0 - p_int], [1.0, -p_int], raw)

    lp_sos = butter(2, min(float(pick_lp_hz), sr * 0.45), btype="lowpass", fs=sr, output="sos")
    hp_sos = butter(1, min(25.0, sr * 0.10), btype="highpass", fs=sr, output="sos")
    exc = sosfilt(hp_sos, sosfilt(lp_sos, vel_wave))

    fade_in = min(max(2, round(0.0003 * sr)), exc_len // 8)
    exc[:fade_in] *= 0.5 - 0.5 * np.cos(math.pi * np.arange(fade_in) / fade_in)
    fade_out = min(max(2, round(0.008 * sr)), exc_len // 3)
    exc[-fade_out:] *= 0.5 + 0.5 * np.cos(math.pi * (np.arange(fade_out) + 1) / fade_out)

    pluck_delay = float(pluck_pos) * sr / f0
    exc_comb = exc - 0.85 * _fractional_delay_read(exc, pluck_delay)
    exc_comb /= max(float(np.max(np.abs(exc_comb))), 1e-9)

    angle_rad = math.radians(28.0)
    h_jitter = np.zeros(exc_len, dtype=np.float64)
    if sr > 1200:
        h_sos = butter(2, [400.0, min(4200.0, sr * 0.44)], btype="bandpass", fs=sr, output="sos")
        h_jitter = sosfilt(h_sos, gen.standard_normal(exc_len))
        h_jitter = (h_jitter / max(float(np.max(np.abs(h_jitter))), 1e-8)) * np.exp(-t_exc / 0.004) * 0.06
    exc_v_burst = math.cos(angle_rad) * exc_comb
    exc_h_burst = math.sin(angle_rad) * (_fractional_delay_read(exc_comb, 0.35) + h_jitter)

    # 2. Coupled dual-polarization stiff digital waveguide.
    f_v = f0 * (2.0 ** (-0.25 / 1200.0))
    f_h = f0 * (2.0 ** ((float(dual_pol_detune_cents) - 0.25) / 1200.0))

    g_v, a_v, tau_lv = _solve_loss_filter(f_v, float(t60_low), float(t60_hi), sr)
    g_h, a_h, tau_lh = _solve_loss_filter(f_h, float(t60_low) * 1.07, float(t60_hi) * 1.06, sr)
    sos_v, tau_dv = _solve_dispersion_allpass(f_v, float(stiffness), sr, m_stages=4)
    sos_h, tau_dh = _solve_dispersion_allpass(f_h, float(stiffness), sr, m_stages=4)

    d0_v = sr / f_v - tau_lv - tau_dv
    d0_h = sr / f_h - tau_lh - tau_dh

    n_grid = np.arange(n_samples, dtype=np.float64)
    t_since = np.maximum((n_grid - n_attack) / sr, 0.0)
    delta_c = float(pitch_drop_cents) * np.exp(-t_since / float(pitch_drop_tau))
    scale_d = 2.0 ** (-delta_c / 1200.0)
    d_vec_v = d0_v * scale_d
    d_vec_h = d0_h * scale_d

    exc_v = np.zeros(n_samples, dtype=np.float64)
    exc_h = np.zeros(n_samples, dtype=np.float64)
    end_exc = min(n_samples, n_attack + exc_len)
    exc_v[n_attack:end_exc] = exc_v_burst[: end_exc - n_attack]
    exc_h[n_attack:end_exc] = exc_h_burst[: end_exc - n_attack]

    buf_v = np.zeros(n_samples, dtype=np.float64)
    buf_h = np.zeros(n_samples, dtype=np.float64)
    b_lv = np.array([g_v * (1.0 - a_v)], dtype=np.float64)
    a_lv = np.array([1.0, -a_v], dtype=np.float64)
    b_lh = np.array([g_h * (1.0 - a_h)], dtype=np.float64)
    a_lh = np.array([1.0, -a_h], dtype=np.float64)

    zi_lv = np.zeros(1, dtype=np.float64)
    zi_lh = np.zeros(1, dtype=np.float64)
    zi_dv = np.zeros((2, 2), dtype=np.float64)
    zi_dh = np.zeros((2, 2), dtype=np.float64)

    eps = float(bridge_coupling)
    min_d = min(float(np.min(d_vec_v)), float(np.min(d_vec_h)))
    block_size = min(64, max(4, int(math.floor(min_d)) - 4))
    start = n_attack
    while start < n_samples:
        end = min(n_samples, start + block_size)
        n_idx = np.arange(start, end, dtype=np.float64)

        def _read_lagrange(buf: np.ndarray, d_vec: np.ndarray) -> np.ndarray:
            pos = n_idx - d_vec[start:end]
            valid = pos >= 1.0
            delayed = np.zeros(end - start, dtype=np.float64)
            if np.any(valid):
                pv = pos[valid]
                m = np.floor(pv).astype(np.int64)
                u = pv - m
                c_m1 = -u * (u - 1.0) * (u - 2.0) / 6.0
                c_0 = (u + 1.0) * (u - 1.0) * (u - 2.0) / 2.0
                c_1 = -(u + 1.0) * u * (u - 2.0) / 2.0
                c_2 = (u + 1.0) * u * (u - 1.0) / 6.0
                delayed[valid] = (
                    c_m1 * buf[m - 1]
                    + c_0 * buf[m]
                    + c_1 * buf[m + 1]
                    + c_2 * buf[m + 2]
                )
            return delayed

        del_v = _read_lagrange(buf_v, d_vec_v)
        del_h = _read_lagrange(buf_h, d_vec_h)

        yv, zi_lv = lfilter(b_lv, a_lv, del_v, zi=zi_lv)
        yv, zi_dv = sosfilt(sos_v, yv, zi=zi_dv)

        yh, zi_lh = lfilter(b_lh, a_lh, del_h, zi=zi_lh)
        yh, zi_dh = sosfilt(sos_h, yh, zi=zi_dh)

        buf_v[start:end] = (1.0 - eps) * yv + eps * yh + exc_v[start:end]
        buf_h[start:end] = eps * yv + (1.0 - eps) * yh + exc_h[start:end]
        start = end

    # 3. Bridge or magnetic pickup comb readout.
    sig = buf_v + float(cross_pol_mix) * buf_h
    if pickup_pairs:
        period_samples = sr / f0
        alpha_u = float(pickup_comb_depth)
        y_pu = np.zeros(n_samples, dtype=np.float64)
        for mu, weight in pickup_pairs:
            y_pu += weight * (sig - alpha_u * _fractional_delay_read(sig, mu * period_samples))
        ap_sos = butter(1, min(9200.0, sr * 0.44), btype="lowpass", fs=sr, output="sos")
        y_pu = sosfilt(ap_sos, y_pu)
        y_pu /= max(float(np.max(np.abs(y_pu))), 1e-8)
        if flux_asymmetry != 0.0:
            y_4x = resample_poly(y_pu, 4, 1)
            y_4x = y_4x + float(flux_asymmetry) * (y_4x * y_4x)
            y_pu = resample_poly(y_4x, 1, 4)[:n_samples]
        out = sosfilt(butter(1, min(35.0, sr * 0.10), btype="highpass", fs=sr, output="sos"), y_pu)
    else:
        out = sosfilt(butter(1, min(25.0, sr * 0.10), btype="highpass", fs=sr, output="sos"), sig)

    if tail_fade > 0.0:
        n_fade = min(round(float(tail_fade) * sr), n_samples // 3)
        if n_fade >= 4:
            out[-n_fade:] *= 0.5 + 0.5 * np.cos(math.pi * (np.arange(n_fade) + 1) / n_fade)

    out = out.astype(np.float64, copy=False)
    if return_polarizations:
        return out, buf_v, buf_h
    return out


def measure_pitch(
    audio: np.ndarray,
    sr: int,
    target_hz: float,
    *,
    start_sec: float = 0.45,
    end_sec: float = 1.45,
    tolerance_ratio: float = 0.06,
) -> Tuple[Optional[float], Optional[float]]:
    """Measure steady-state pitch in Hz and cents error via zero-padded parabolic FFT peak."""
    x = _as_finite_mono(audio)
    if sr <= 0:
        raise ValueError("Sample rate must be positive")
    if not np.isfinite(target_hz) or not 0.0 < target_hz < sr * 0.5:
        raise ValueError("target_hz must be above zero and below Nyquist")
    if not np.isfinite(tolerance_ratio) or not 0.0 < tolerance_ratio < 0.5:
        raise ValueError("tolerance_ratio must be in (0, 0.5)")
    if x.size == 0:
        return None, None

    dur_sec = x.size / sr
    if dur_sec < end_sec:
        start_idx = round(0.15 * x.size)
        end_idx = round(0.85 * x.size)
    else:
        start_idx = max(0, round(start_sec * sr))
        end_idx = min(x.size, round(end_sec * sr))

    if end_idx - start_idx < max(256, round(0.10 * sr)):
        return None, None

    seg = x[start_idx:end_idx]
    if float(np.max(np.abs(seg))) < 1e-6:
        return None, None

    seg = seg * np.hanning(seg.size)
    nfft = max(1 << 18, 1 << (seg.size - 1).bit_length())
    spec = np.abs(np.fft.rfft(seg, n=nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    mask = (freqs > target_hz * (1.0 - tolerance_ratio)) & (
        freqs < target_hz * (1.0 + tolerance_ratio)
    )
    indices = np.flatnonzero(mask)
    if indices.size < 3:
        return None, None

    idx = int(indices[int(np.argmax(spec[indices]))])
    if 0 < idx < len(spec) - 1:
        y0 = math.log(float(spec[idx - 1]) + 1e-12)
        y1 = math.log(float(spec[idx]) + 1e-12)
        y2 = math.log(float(spec[idx + 1]) + 1e-12)
        denom = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    else:
        delta = 0.0

    f_meas = float((idx + delta) * sr / nfft)
    cents = float(1200.0 * math.log2(f_meas / target_hz))
    return round(f_meas, 4), round(cents, 3)


def write_sfz(
    path: Union[str, os.PathLike],
    regions: Iterable[Mapping[str, Union[str, int, float]]],
    *,
    attack: float = 0.001,
    release: float = 0.14,
    amp_veltrack: float = 45.0,
    header_comment: Optional[str] = None,
) -> Path:
    """Write a DAW/sampler-ready SFZ instrument patch from region mappings."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    if header_comment:
        for line in str(header_comment).splitlines():
            lines.append(f"// {line}")
    lines.extend([
        "<global>",
        f"ampeg_attack={attack}",
        f"ampeg_release={release}",
        f"amp_veltrack={amp_veltrack}",
        "",
    ])
    region_list = list(regions)
    if not region_list:
        raise ValueError("SFZ patch must contain at least one region")
    for reg in region_list:
        if "sample" not in reg:
            raise ValueError("Each SFZ region must specify a 'sample' path")
        lines.append("<region>")
        for key, val in reg.items():
            lines.append(f"{key}={val}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _riff_chunk(fourcc: bytes, payload: bytes) -> bytes:
    """Pack a RIFF subchunk, adding a single zero pad byte if payload length is odd."""
    pad = b"\x00" if (len(payload) % 2) else b""
    return fourcc + struct.pack("<I", len(payload)) + payload + pad


def _riff_list(list_type: bytes, chunks: Sequence[bytes]) -> bytes:
    """Pack a RIFF LIST chunk containing subchunks under a 4-byte list type tag."""
    return _riff_chunk(b"LIST", list_type + b"".join(chunks))


def _sf2_zstr(text: str, max_len: int = 256) -> bytes:
    """Encode an ASCII zstr terminated by 1 or 2 NUL bytes so length is even per SF2 5.1."""
    raw = str(text).encode("ascii", errors="replace")[: max_len - 1]
    out = raw + b"\x00"
    if len(out) % 2:
        out += b"\x00"
    return out


def _sf2_fixed_name(text: str) -> bytes:
    """Encode a 20-byte fixed-width SF2 name field (max 19 ASCII bytes + NUL padding)."""
    raw = str(text).encode("ascii", errors="replace")[:19]
    return raw.ljust(20, b"\x00")


def _unique_sf2_name(base: str, used: set, suffix: str = "") -> str:
    """Produce a unique ASCII name fitting within SF2's 19-character display limit."""
    clean_base = str(base).encode("ascii", errors="replace").decode("ascii")
    clean_suffix = str(suffix).encode("ascii", errors="replace").decode("ascii")
    for idx in range(1000):
        tag = f"_{idx + 1}" if idx > 0 else ""
        tail = f"{clean_suffix}{tag}"
        keep = max(1, 19 - len(tail))
        candidate = f"{clean_base[:keep]}{tail}"[:19]
        if candidate not in used:
            used.add(candidate)
            return candidate
    raise ValueError(f"Could not allocate unique SF2 name near {base!r}")


def _sec_to_timecents(seconds: float) -> int:
    """Convert envelope stage duration in seconds to SF2 int16 timecents."""
    val = int(round(1200.0 * math.log2(max(float(seconds), 1e-4))))
    return max(-12000, min(8000, val))


def _pack_gen_u16(oper: int, val: int) -> bytes:
    return struct.pack("<HH", int(oper), int(val))


def _pack_gen_i16(oper: int, val: int) -> bytes:
    return struct.pack("<Hh", int(oper), int(val))


def _pack_gen_range(oper: int, lo: int, hi: int) -> bytes:
    return struct.pack("<HBB", int(oper), int(lo), int(hi))


def _pack_mod(src: int, dest: int, amount: int, amt_src: int = 0, trans: int = 0) -> bytes:
    return struct.pack("<HHhHH", int(src), int(dest), int(amount), int(amt_src), int(trans))


def _resolve_sf2_sample_path(
    sample_ref: Union[str, os.PathLike],
    out_dir: Path,
    sample_root: Optional[Union[str, os.PathLike]],
) -> Path:
    """Resolve an SF2 region's sample path against sample_root, out_dir, or cwd."""
    ref = Path(sample_ref)
    if ref.is_absolute():
        if ref.is_file():
            return ref.resolve()
        raise FileNotFoundError(f"Sample file not found: {ref}")
    if sample_root is not None:
        candidate = Path(sample_root) / ref
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(f"Sample file not found under sample_root: {candidate}")
    near_out = out_dir / ref
    if near_out.is_file():
        return near_out.resolve()
    if ref.is_file():
        return ref.resolve()
    raise FileNotFoundError(f"Sample file not found: {ref} (checked {near_out} and cwd)")


def _encode_sf2_channel(ch: np.ndarray, bit_depth: int) -> Tuple[bytes, Optional[bytes], int]:
    """Encode one channel + >=46 trailing zeros into SF2 hi-16 and optional lo-8 byte arrays."""
    n_frames = int(ch.size)
    pad_len = 46 if (n_frames % 2 == 0) else 47
    padded_zeros_16 = np.zeros(pad_len, dtype="<i2")

    if bit_depth == 24:
        s24 = np.clip(np.rint(ch * 8388608.0), -8388608, 8388607).astype(np.int32)
        hi16 = np.concatenate([(s24 >> 8).astype("<i2"), padded_zeros_16])
        lo8 = np.concatenate([(s24 & 0xFF).astype(np.uint8), np.zeros(pad_len, dtype=np.uint8)])
        return hi16.tobytes(), lo8.tobytes(), n_frames + pad_len

    hi16 = np.concatenate([np.clip(np.rint(ch * 32768.0), -32768, 32767).astype("<i2"), padded_zeros_16])
    return hi16.tobytes(), None, n_frames + pad_len


def write_sf2(
    path: Union[str, os.PathLike],
    presets: Sequence[Mapping[str, object]],
    *,
    bank_name: str = "AudioShmaudio",
    author: str = "AudioShmaudio",
    comment: Optional[str] = None,
    bit_depth: int = 24,
    sample_root: Optional[Union[str, os.PathLike]] = None,
) -> Path:
    """Write a SoundFont 2.04 (.sf2) bank from preset definitions referencing mono or stereo WAVs."""
    if bit_depth not in (16, 24):
        raise ValueError("bit_depth must be 16 or 24")
    preset_list = list(presets)
    if not preset_list:
        raise ValueError("SoundFont must contain at least one preset")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_presets: List[dict] = []
    seen_slots = set()
    used_preset_names = {"EOP"}
    for default_prog, p in enumerate(preset_list):
        raw_name = str(p.get("name", "")).strip()
        if not raw_name:
            raise ValueError("Each preset must specify a non-empty 'name'")
        bank = int(p.get("bank", 0))
        program = int(p.get("program", default_prog))
        if not 0 <= bank <= 128:
            raise ValueError(f"Preset bank {bank} out of range 0..128")
        if not 0 <= program <= 127:
            raise ValueError(f"Preset program {program} out of range 0..127")
        if (bank, program) in seen_slots:
            raise ValueError(f"Duplicate (bank, program) slot ({bank}, {program})")
        seen_slots.add((bank, program))

        regions = list(p.get("regions", ()))  # type: ignore[arg-type]
        if not regions:
            raise ValueError(f"Preset {raw_name!r} must contain at least one region")

        attack = float(p.get("attack", 0.001))
        release = float(p.get("release", 0.14))
        amp_veltrack = float(p.get("amp_veltrack", 100.0))
        if not 0.0 <= amp_veltrack <= 100.0:
            raise ValueError(f"amp_veltrack must be in 0..100, got {amp_veltrack}")

        normalized_presets.append({
            "name": _unique_sf2_name(raw_name, used_preset_names),
            "bank": bank,
            "program": program,
            "regions": regions,
            "attack": attack,
            "release": release,
            "amp_veltrack": amp_veltrack,
        })

    normalized_presets.sort(key=lambda item: (item["bank"], item["program"]))

    # Sample pool deduplicated by resolved file path in `smpl`/`sm24`, plus header deduplication
    smpl_chunks: List[bytes] = []
    sm24_chunks: List[bytes] = []
    total_sample_frames = 0
    stored_files: Dict[Path, Tuple[int, List[Tuple[int, int]]]] = {}
    sample_headers: List[dict] = []
    header_cache: Dict[Tuple[Path, int, int, int], Tuple[int, ...]] = {}
    used_sample_names = {"EOS"}

    def _ensure_file_samples(wav_path: Path) -> Tuple[int, List[Tuple[int, int]]]:
        nonlocal total_sample_frames
        if wav_path in stored_files:
            return stored_files[wav_path]
        audio, file_sr = sf.read(wav_path, dtype="float64", always_2d=True)
        n_frames, n_ch = audio.shape
        if n_frames == 0:
            raise ValueError(f"Sample file is empty: {wav_path}")
        if n_ch not in (1, 2):
            raise ValueError(f"SF2 only supports mono or stereo WAVs, got {n_ch} channels in {wav_path}")
        if not np.all(np.isfinite(audio)):
            raise ValueError(f"Non-finite audio in sample file: {wav_path}")

        ch_ranges: List[Tuple[int, int]] = []
        for ch_idx in range(n_ch):
            hi_bytes, lo_bytes, padded_frames = _encode_sf2_channel(audio[:, ch_idx], bit_depth)
            start = total_sample_frames
            end = start + n_frames
            smpl_chunks.append(hi_bytes)
            if lo_bytes is not None:
                sm24_chunks.append(lo_bytes)
            total_sample_frames += padded_frames
            ch_ranges.append((start, end))
        stored_files[wav_path] = (int(file_sr), ch_ranges)
        return int(file_sr), ch_ranges

    def _get_or_add_sample_indices(
        wav_path: Path,
        root_key: int,
        loop_mode: int,
        loop_start_req: Optional[int],
        loop_end_req: Optional[int],
    ) -> Tuple[int, ...]:
        file_sr, ch_ranges = _ensure_file_samples(wav_path)
        n_frames = ch_ranges[0][1] - ch_ranges[0][0]
        if loop_mode in (1, 3):
            ls_rel = max(0, min(n_frames - 2, int(loop_start_req if loop_start_req is not None else 0)))
            le_rel = max(ls_rel + 1, min(n_frames, int(loop_end_req if loop_end_req is not None else n_frames)))
        else:
            ls_rel = 0
            le_rel = n_frames

        cache_key = (wav_path, int(root_key), ls_rel, le_rel)
        if cache_key in header_cache:
            return header_cache[cache_key]

        n_ch = len(ch_ranges)
        if n_ch == 1:
            start, end = ch_ranges[0]
            idx = len(sample_headers)
            sample_headers.append({
                "name": _unique_sf2_name(wav_path.stem, used_sample_names),
                "start": start,
                "end": end,
                "loop_start": start + ls_rel,
                "loop_end": start + le_rel,
                "sr": file_sr,
                "root_key": root_key,
                "link": 0,
                "type": 1,
            })
            indices: Tuple[int, ...] = (idx,)
        else:
            left_idx = len(sample_headers)
            right_idx = left_idx + 1
            for ch_pos, (suffix, stype, link) in enumerate((("_L", 4, right_idx), ("_R", 2, left_idx))):
                start, end = ch_ranges[ch_pos]
                sample_headers.append({
                    "name": _unique_sf2_name(wav_path.stem, used_sample_names, suffix=suffix),
                    "start": start,
                    "end": end,
                    "loop_start": start + ls_rel,
                    "loop_end": start + le_rel,
                    "sr": file_sr,
                    "root_key": root_key,
                    "link": link,
                    "type": stype,
                })
            indices = (left_idx, right_idx)
        header_cache[cache_key] = indices
        return indices

    phdr_records: List[bytes] = []
    pbag_records: List[bytes] = []
    pgen_records: List[bytes] = []
    inst_records: List[bytes] = []
    ibag_records: List[bytes] = []
    imod_records: List[bytes] = []
    igen_records: List[bytes] = []

    used_inst_names = {"EOI"}

    for inst_idx, p in enumerate(normalized_presets):
        pbag_start = len(pbag_records)
        phdr_records.append(
            struct.pack(
                "<20sHHHIII",
                _sf2_fixed_name(p["name"]),
                p["program"],
                p["bank"],
                pbag_start,
                0,
                0,
                0,
            )
        )
        pbag_records.append(struct.pack("<HH", len(pgen_records), 0))
        pgen_records.append(_pack_gen_u16(41, inst_idx))  # instrument generator

        inst_name = _unique_sf2_name(p["name"], used_inst_names)
        inst_records.append(struct.pack("<20sH", _sf2_fixed_name(inst_name), len(ibag_records)))

        # Instrument global zone (first ibag, no sampleID)
        ibag_records.append(struct.pack("<HH", len(igen_records), len(imod_records)))
        igen_records.append(_pack_gen_i16(34, _sec_to_timecents(p["attack"])))   # attackVolEnv
        igen_records.append(_pack_gen_i16(37, 0))                                  # sustainVolEnv = full level
        igen_records.append(_pack_gen_i16(38, _sec_to_timecents(p["release"])))  # releaseVolEnv
        igen_records.append(_pack_gen_u16(54, 0))                                  # sampleModes = no loop
        if abs(p["amp_veltrack"] - 100.0) > 1e-6:
            vel_cb = int(round(960.0 * p["amp_veltrack"] / 100.0))
            imod_records.append(_pack_mod(0x0502, 48, vel_cb, 0, 0))

        # Per-region instrument zones
        for reg in p["regions"]:
            if "sample" not in reg:
                raise ValueError(f"Preset {p['name']!r} has region missing 'sample'")
            wav_path = _resolve_sf2_sample_path(reg["sample"], out_path.parent, sample_root)  # type: ignore[arg-type]
            root_key = int(reg.get("pitch_keycenter", 60))
            lokey = int(reg.get("lokey", 0))
            hikey = int(reg.get("hikey", 127))
            lovel = int(reg.get("lovel", 1))
            hivel = int(reg.get("hivel", 127))
            if not (0 <= lokey <= hikey <= 127):
                raise ValueError(f"Invalid key range {lokey}..{hikey} in {wav_path.name}")
            if not (0 <= lovel <= hivel <= 127):
                raise ValueError(f"Invalid vel range {lovel}..{hivel} in {wav_path.name}")
            if not 0 <= root_key <= 127:
                raise ValueError(f"Invalid pitch_keycenter {root_key} in {wav_path.name}")

            loop_mode = int(reg.get("loop_mode", 0))
            if loop_mode not in (0, 1, 3):
                raise ValueError(f"Unsupported loop_mode {loop_mode}; expected 0, 1, or 3")
            loop_start_req = int(reg["loop_start"]) if "loop_start" in reg else None
            loop_end_req = int(reg["loop_end"]) if "loop_end" in reg else None
            sample_indices = _get_or_add_sample_indices(
                wav_path, root_key, loop_mode, loop_start_req, loop_end_req
            )

            tune_cents = int(reg.get("tune", 0))
            has_pan = "pan" in reg
            pan_permille = int(round(max(-1.0, min(1.0, float(reg.get("pan", 0.0)))) * 500.0))

            for ch_pos, sample_idx in enumerate(sample_indices):
                ibag_records.append(struct.pack("<HH", len(igen_records), len(imod_records)))
                igen_records.append(_pack_gen_range(43, lokey, hikey))  # keyRange MUST be first
                igen_records.append(_pack_gen_range(44, lovel, hivel))  # velRange MUST be second
                if len(sample_indices) == 2:
                    igen_records.append(_pack_gen_i16(17, -500 if ch_pos == 0 else 500))
                elif has_pan:
                    igen_records.append(_pack_gen_i16(17, pan_permille))
                if tune_cents != 0:
                    igen_records.append(_pack_gen_i16(52, max(-99, min(99, tune_cents))))
                if loop_mode != 0:
                    igen_records.append(_pack_gen_u16(54, loop_mode))
                igen_records.append(_pack_gen_i16(58, root_key))  # overridingRootKey
                igen_records.append(_pack_gen_u16(53, sample_idx))  # sampleID MUST be last

    # Terminal records required by SF2 2.04 spec
    phdr_records.append(struct.pack("<20sHHHIII", _sf2_fixed_name("EOP"), 0, 0, len(pbag_records), 0, 0, 0))
    pbag_records.append(struct.pack("<HH", len(pgen_records), 0))
    pmod_records = [b"\x00" * 10]
    pgen_records.append(b"\x00" * 4)

    inst_records.append(struct.pack("<20sH", _sf2_fixed_name("EOI"), len(ibag_records)))
    ibag_records.append(struct.pack("<HH", len(igen_records), len(imod_records)))
    imod_records.append(b"\x00" * 10)
    igen_records.append(b"\x00" * 4)

    if len(pgen_records) >= 65536 or len(igen_records) >= 65536:
        raise ValueError("SF2 generator table exceeded uint16 index limit (65535)")

    shdr_records: List[bytes] = []
    for sh in sample_headers:
        shdr_records.append(
            struct.pack(
                "<20sIIIIIBbHH",
                _sf2_fixed_name(sh["name"]),
                sh["start"],
                sh["end"],
                sh["loop_start"],
                sh["loop_end"],
                sh["sr"],
                sh["root_key"],
                0,
                sh["link"],
                sh["type"],
            )
        )
    shdr_records.append(struct.pack("<20sIIIIIBbHH", _sf2_fixed_name("EOS"), 0, 0, 0, 0, 0, 0, 0, 0, 0))

    info_chunks: List[bytes] = [
        _riff_chunk(b"ifil", struct.pack("<HH", 2, 4 if bit_depth == 24 else 1)),
        _riff_chunk(b"isng", _sf2_zstr("EMU8000", 256)),
        _riff_chunk(b"INAM", _sf2_zstr(bank_name, 256)),
        _riff_chunk(b"ISFT", _sf2_zstr("AudioShmaudio", 256)),
    ]
    if author:
        info_chunks.append(_riff_chunk(b"IENG", _sf2_zstr(author, 256)))
    if comment:
        info_chunks.append(_riff_chunk(b"ICMT", _sf2_zstr(comment, 65536)))

    sdta_chunks: List[bytes] = [_riff_chunk(b"smpl", b"".join(smpl_chunks))]
    if bit_depth == 24:
        sdta_chunks.append(_riff_chunk(b"sm24", b"".join(sm24_chunks)))

    pdta_chunks: List[bytes] = [
        _riff_chunk(b"phdr", b"".join(phdr_records)),
        _riff_chunk(b"pbag", b"".join(pbag_records)),
        _riff_chunk(b"pmod", b"".join(pmod_records)),
        _riff_chunk(b"pgen", b"".join(pgen_records)),
        _riff_chunk(b"inst", b"".join(inst_records)),
        _riff_chunk(b"ibag", b"".join(ibag_records)),
        _riff_chunk(b"imod", b"".join(imod_records)),
        _riff_chunk(b"igen", b"".join(igen_records)),
        _riff_chunk(b"shdr", b"".join(shdr_records)),
    ]

    sfbk_body = (
        b"sfbk"
        + _riff_list(b"INFO", info_chunks)
        + _riff_list(b"sdta", sdta_chunks)
        + _riff_list(b"pdta", pdta_chunks)
    )
    out_path.write_bytes(_riff_chunk(b"RIFF", sfbk_body))
    return out_path

