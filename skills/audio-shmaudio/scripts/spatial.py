"""Spatial placement, distance absorption, room reflections, and scene mixing."""

from typing import Optional, Union

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from procedural import Contour, FoleyLayer, evaluate_contour


def _as_finite_array(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio)
    if arr.ndim not in (1, 2) or arr.dtype.kind not in "fiu" or not np.all(np.isfinite(arr)):
        raise ValueError("Audio must be a finite 1D mono or 2D stereo array")
    if arr.ndim == 2 and arr.shape[1] not in (1, 2):
        raise ValueError("2D audio must have 1 or 2 channels")
    if arr.dtype.kind == "i":
        return arr.astype(np.float64) / float(np.iinfo(arr.dtype).max)
    return arr.astype(np.float64, copy=True)


def pan_mono(audio: np.ndarray, pan: Contour = 0.0) -> np.ndarray:
    """Place a mono signal in a stereo field using equal-power panning.

    `pan` may be a scalar or (fraction, value) contour in [-1.0, 1.0],
    where -1.0 is hard left, 0.0 is center, and +1.0 is hard right.
    """
    mono = _as_finite_array(audio)
    if mono.ndim == 2:
        if mono.shape[1] != 1:
            raise ValueError("pan_mono requires a mono 1D array")
        mono = mono[:, 0]
    if mono.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    pan_curve = evaluate_contour(pan, mono.size, "Pan", -1.0, 1.0)
    angle = (pan_curve + 1.0) * (np.pi * 0.25)
    return np.column_stack((mono * np.cos(angle), mono * np.sin(angle)))


def distance_filter(audio: np.ndarray, sr: int, distance: float = 0.0) -> np.ndarray:
    """Attenuate high frequencies to model air absorption at a distance in [0, 1]."""
    if sr <= 0:
        raise ValueError("Sample rate must be positive")
    if not np.isfinite(distance) or not 0.0 <= distance <= 1.0:
        raise ValueError("Distance must be between zero and one")
    arr = _as_finite_array(audio)
    if distance == 0.0 or arr.shape[0] == 0:
        return arr
    hi_hz = min(18000.0, sr * 0.45)
    low_hz = min(900.0, hi_hz * 0.9)
    cutoff = hi_hz * (low_hz / hi_hz) ** (distance ** 0.85)
    sos = butter(2, cutoff, btype="lowpass", fs=sr, output="sos")
    return sosfilt(sos, arr, axis=0)


def add_room(
    audio: np.ndarray,
    sr: int,
    *,
    size: float = 0.45,
    damping: float = 0.55,
    wet: float = 0.22,
    predelay: float = 0.012,
    early_reflections: int = 6,
    seed: Optional[int] = 0,
) -> np.ndarray:
    """Apply deterministic early reflections and a decorrelated stereo room tail."""
    if sr <= 0:
        raise ValueError("Sample rate must be positive")
    if not np.isfinite(size) or not 0.0 <= size <= 1.0:
        raise ValueError("Room size must be in [0, 1]")
    if not np.isfinite(damping) or not 0.0 <= damping <= 1.0:
        raise ValueError("Room damping must be in [0, 1]")
    if not np.isfinite(wet) or not 0.0 <= wet <= 1.0:
        raise ValueError("Wet mix must be in [0, 1]")
    if not np.isfinite(predelay) or predelay < 0.0:
        raise ValueError("Predelay must be finite and nonnegative")
    if early_reflections < 0:
        raise ValueError("Early reflection count must be nonnegative")

    arr = _as_finite_array(audio)
    if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
        dry = pan_mono(arr, 0.0)
    else:
        dry = arr

    n_samples = dry.shape[0]
    if wet == 0.0 or n_samples == 0:
        return dry

    rt60 = min(0.12 + 2.6 * (size ** 1.6), 6.0)
    ir_len = max(64, round((predelay + rt60 * 1.15) * sr))
    rng = np.random.default_rng(seed)
    t = np.arange(ir_len, dtype=np.float64) / sr
    tail_start = predelay + 0.015
    rise = np.clip((t - tail_start) / 0.015, 0.0, 1.0)
    rise = 0.5 - 0.5 * np.cos(np.pi * rise)
    decay_env = np.exp(-6.9 * np.maximum(t - tail_start, 0.0) / rt60)

    damp_hi = min(14000.0, sr * 0.42)
    damp_lo = min(1400.0, damp_hi * 0.8)
    lp_cutoff = damp_hi * (damp_lo / damp_hi) ** damping
    lp_sos = butter(2, lp_cutoff, btype="lowpass", fs=sr, output="sos")
    hp_sos = butter(1, 80.0, btype="highpass", fs=sr, output="sos") if sr > 250 else None

    ir = np.zeros((ir_len, 2), dtype=np.float64)
    tap_span = 0.025 + 0.08 * size
    for ch in range(2):
        ir_ch = np.zeros(ir_len, dtype=np.float64)
        if early_reflections > 0:
            base_offsets = np.linspace(0.0, tap_span, early_reflections)
            for idx, base in enumerate(base_offsets):
                jitter = rng.uniform(-0.0025, 0.0025) if idx > 0 else rng.uniform(0.0, 0.002)
                tap_t = max(predelay, predelay + base + jitter)
                tap_idx = round(tap_t * sr)
                if 0 <= tap_idx < ir_len:
                    amp = np.exp(-2.8 * (tap_t - predelay) / max(0.05 + 0.15 * size, 1e-3))
                    polarity = float(rng.choice((-1.0, 1.0)))
                    ir_ch[tap_idx] += polarity * amp

        ir_ch += rng.standard_normal(ir_len) * rise * decay_env * 0.35
        ir_ch = sosfilt(lp_sos, ir_ch)
        if hp_sos is not None:
            ir_ch = sosfilt(hp_sos, ir_ch)
        ir[:, ch] = ir_ch

    energy = float(np.sqrt(0.5 * np.sum(ir**2)))
    if energy > 1e-12:
        ir /= energy

    reverb = np.column_stack(
        (
            fftconvolve(dry[:, 0], ir[:, 0], mode="full")[:n_samples],
            fftconvolve(dry[:, 1], ir[:, 1], mode="full")[:n_samples],
        )
    )
    return (1.0 - 0.3 * wet) * dry + wet * reverb


def soft_limit(
    audio: np.ndarray,
    ceiling_db: float = -1.0,
    drive: float = 1.0,
) -> np.ndarray:
    """Transparent soft-knee tanh limiter that preserves quiet dynamics."""
    if not np.isfinite(ceiling_db):
        raise ValueError("Ceiling must be finite")
    if not np.isfinite(drive) or drive <= 0.0:
        raise ValueError("Drive must be finite and positive")

    x = _as_finite_array(audio) * drive
    ceiling = 10.0 ** (ceiling_db / 20.0)
    knee = 0.65 * ceiling
    head = max(ceiling - knee, 1e-12)
    abs_x = np.abs(x)
    over = abs_x > knee
    if not np.any(over):
        return x
    out = x.copy()
    excess = (abs_x[over] - knee) / head
    out[over] = np.sign(x[over]) * (knee + head * np.tanh(excess))
    return out


class StereoScene:
    """Compose layered mono or stereo sources into a spatialized stereo mix."""

    def __init__(self, n_samples: int, sr: int, seed: Optional[int] = 0):
        if n_samples <= 0 or sr <= 0:
            raise ValueError("Scene sample count and sample rate must be positive")
        self.n_samples = int(n_samples)
        self.sr = int(sr)
        self.audio = np.zeros((self.n_samples, 2), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def add(
        self,
        source: Union[FoleyLayer, np.ndarray],
        *,
        pan: Contour = 0.0,
        gain: float = 1.0,
        distance: float = 0.0,
        delay: float = 0.0,
    ) -> None:
        """Mix a FoleyLayer or array into the stereo scene with pan, distance, and delay."""
        if not np.isfinite(gain) or gain < 0.0:
            raise ValueError("Gain must be finite and nonnegative")
        if not np.isfinite(delay):
            raise ValueError("Delay must be finite")

        raw = source.audio if isinstance(source, FoleyLayer) else np.asarray(source)
        arr = _as_finite_array(raw)
        if arr.shape[0] == 0 or gain == 0.0:
            return

        filtered = distance_filter(arr, self.sr, distance=distance) if distance > 0.0 else arr
        if filtered.ndim == 1 or filtered.shape[1] == 1:
            stereo = pan_mono(filtered, pan=pan)
        else:
            pan_curve = evaluate_contour(pan, filtered.shape[0], "Pan", -1.0, 1.0)
            angle = (pan_curve + 1.0) * (np.pi * 0.25)
            stereo = np.column_stack(
                (
                    filtered[:, 0] * (np.sqrt(2.0) * np.cos(angle)),
                    filtered[:, 1] * (np.sqrt(2.0) * np.sin(angle)),
                )
            )

        start = round(delay * self.sr)
        first = max(0, start)
        end = min(self.n_samples, start + stereo.shape[0])
        if end > first:
            self.audio[first:end] += (gain * stereo[first - start : end - start]).astype(
                np.float32
            )

    def apply_room(
        self,
        *,
        size: float = 0.45,
        damping: float = 0.55,
        wet: float = 0.22,
        predelay: float = 0.012,
        early_reflections: int = 6,
        seed: Optional[int] = None,
    ) -> None:
        """Apply stereo room reflections to the accumulated scene in place."""
        room_seed = int(self.rng.integers(0, 2**31 - 1)) if seed is None else seed
        self.audio = add_room(
            self.audio,
            self.sr,
            size=size,
            damping=damping,
            wet=wet,
            predelay=predelay,
            early_reflections=early_reflections,
            seed=room_seed,
        ).astype(np.float32)

    def fade_edges(self, fade_in: float = 0.0, fade_out: float = 0.0) -> None:
        """Apply smooth raised-cosine edge fades in seconds."""
        if not np.isfinite(fade_in) or fade_in < 0.0:
            raise ValueError("fade_in must be finite and nonnegative")
        if not np.isfinite(fade_out) or fade_out < 0.0:
            raise ValueError("fade_out must be finite and nonnegative")

        n_in = min(round(fade_in * self.sr), self.n_samples)
        if n_in >= 2:
            ramp_in = (0.5 - 0.5 * np.cos(np.pi * np.arange(n_in) / n_in)).astype(np.float32)
            self.audio[:n_in] *= ramp_in[:, None]

        n_out = min(round(fade_out * self.sr), self.n_samples)
        if n_out >= 2:
            ramp_out = (
                0.5 + 0.5 * np.cos(np.pi * (np.arange(n_out) + 1) / n_out)
            ).astype(np.float32)
            self.audio[-n_out:] *= ramp_out[:, None]

    def soft_limit(self, ceiling_db: float = -1.0, drive: float = 1.0) -> None:
        """Apply transparent soft-knee limiting to the scene in place."""
        self.audio = soft_limit(self.audio, ceiling_db=ceiling_db, drive=drive).astype(
            np.float32
        )

    def clear(self) -> None:
        """Reset the scene buffer to silence."""
        self.audio.fill(0.0)
