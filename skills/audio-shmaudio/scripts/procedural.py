"""Sample-accurate building blocks for source-free foley."""

from typing import Optional, Sequence, Tuple, Union

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, iirpeak, lfilter, sosfilt

from instruments import PickupSpec, cabinet_filter, plucked_string, saturate


Band = Tuple[float, float]
Pressure = Sequence[Tuple[float, float]]
Contour = Union[float, Pressure]
Mode = Union[Tuple[float, float, float], Tuple[float, float, float, float]]
Partial = Tuple[float, float]
Formant = Tuple[float, float, float, float]
DEFAULT_PRESSURE = ((0.0, 0.0), (0.18, 1.0), (0.55, 0.65), (1.0, 0.0))
DEFAULT_FORMANTS = (
    (620, 980, 2.2, 0.64),
    (1300, 2100, 3.1, 0.83),
    (2680, 3400, 3.3, 0.38),
)
DEFAULT_STICK_SLIP_MODES: Tuple[Tuple[float, float, float], ...] = (
    (168.0, 0.052, 0.72),
    (285.0, 0.044, 0.85),
    (472.0, 0.036, 1.00),
    (760.0, 0.028, 0.88),
    (1140.0, 0.022, 0.72),
    (1680.0, 0.017, 0.52),
    (2390.0, 0.014, 0.36),
    (3320.0, 0.011, 0.20),
)


def modal_filter(
    audio: np.ndarray,
    sr: int,
    modes: Sequence[Tuple[float, float, float]],
    *,
    dry: float = 0.0,
) -> np.ndarray:
    """Filter a mono signal through a parallel bank of damped 2-pole resonators.

    Each mode is a (freq_hz, decay_sec, level) tuple normalized to unity gain at
    its resonant peak. Returns `dry * audio + wet_sum`.
    """
    if sr <= 0:
        raise ValueError("Sample rate must be positive")
    if not np.isfinite(dry) or dry < 0.0:
        raise ValueError("Dry level must be finite and nonnegative")
    arr = np.asarray(audio)
    if arr.ndim != 1 or arr.dtype.kind not in "fiu" or not np.all(np.isfinite(arr)):
        raise ValueError("Audio must be a finite mono signal")
    if arr.dtype.kind == "i":
        x = arr.astype(np.float64) / float(np.iinfo(arr.dtype).max)
    else:
        x = arr.astype(np.float64, copy=False)
    if not modes:
        raise ValueError("Modes must be non-empty")

    validated = []
    total_level = 0.0
    for mode in modes:
        if len(mode) != 3:
            raise ValueError("Each modal filter entry must be (freq_hz, decay_sec, level)")
        freq_hz, decay_sec, level = mode
        if not np.isfinite(freq_hz) or not 0.0 < freq_hz < sr * 0.5:
            raise ValueError("Mode frequency must be above zero and below Nyquist")
        if not np.isfinite(decay_sec) or decay_sec <= 0.0:
            raise ValueError("Mode decay must be finite and positive")
        if not np.isfinite(level) or level < 0.0:
            raise ValueError("Mode level must be finite and nonnegative")
        validated.append((float(freq_hz), float(decay_sec), float(level)))
        total_level += float(level)

    if x.size == 0:
        return np.zeros(0, dtype=np.float64)

    wet = np.zeros(x.size, dtype=np.float64)
    for freq_hz, decay_sec, level in validated:
        if level == 0.0:
            continue
        r = min(float(np.exp(-6.907755 / max(decay_sec * sr, 1.0))), 0.99995)
        w0 = 2.0 * np.pi * freq_hz / sr
        a = np.array([1.0, -2.0 * r * np.cos(w0), r * r], dtype=np.float64)
        b_raw = np.array([1.0 - r, 0.0, -(1.0 - r)], dtype=np.float64)
        z_inv = np.exp(-1j * w0)
        peak_mag = abs(
            (b_raw[0] + b_raw[2] * z_inv * z_inv)
            / (a[0] + a[1] * z_inv + a[2] * z_inv * z_inv)
        )
        b = b_raw / max(peak_mag, 1e-12)
        wet += level * lfilter(b, a, x)

    if total_level > 0.0:
        wet /= total_level
    return dry * x + wet


def evaluate_contour(
    contour: Contour,
    count: int,
    name: str,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    """Evaluate a scalar or (fraction, value) knot curve over `count` samples."""
    if count <= 0:
        raise ValueError("Sample count must be positive")
    if np.isscalar(contour) and not isinstance(contour, (bool, str, bytes)):
        points = np.array([[0.0, float(contour)], [1.0, float(contour)]], dtype=float)
    else:
        points = np.asarray(contour, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        raise ValueError(f"{name} needs a constant or (fraction, value) pairs")
    fractions, values = points.T
    if (
        not np.all(np.isfinite(points))
        or np.any(values < minimum)
        or np.any(values > maximum)
        or fractions[0] != 0
        or fractions[-1] != 1
        or not np.all(np.diff(fractions) > 0)
    ):
        raise ValueError(
            f"{name} needs increasing fractions from 0 to 1 and values in [{minimum}, {maximum}]"
        )
    evaluated = PchipInterpolator(fractions, values)(np.linspace(0.0, 1.0, count))
    return np.clip(evaluated, minimum, maximum)


class FoleyLayer:
    """Mix events on a mono timeline without phase reconstruction."""

    def __init__(self, n_samples: int, sr: int, seed: Optional[int] = 0):
        self.sr = sr
        self.audio = np.zeros(n_samples, dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def contact(
        self,
        time: float,
        duration: float = 0.018,
        *,
        band: Band = (250, 4500),
        gain: float = 0.3,
        modes: Sequence[Mode] = (),
    ) -> None:
        """Add a damped noise strike and optional resonant or pitch-gliding modes.

        Modes are (hz, decay, level) or (start_hz, decay, level, end_hz) tuples.
        Duration sets the length of the noise strike. Each mode reaches about
        -60 dB by its own decay time with a smooth attack and tail taper.
        """
        strike = self._burst(self._sample_count(duration), band)
        for mode in modes:
            if len(mode) == 3:
                hz, decay, level = mode
                end_hz = hz
            elif len(mode) == 4:
                hz, decay, level, end_hz = mode
            else:
                raise ValueError(
                    "Mode must be (hz, decay, level) or (start_hz, decay, level, end_hz)"
                )
            if not (0 < hz < self.sr / 2 and 0 < end_hz < self.sr / 2):
                raise ValueError("Mode frequency must be below Nyquist and above zero")
            self._check_gain(level)
            n = self._sample_count(decay)
            t = np.arange(n, dtype=np.float64) / self.sr
            if abs(end_hz - hz) < 1e-6:
                phase = 2.0 * np.pi * hz * t
            else:
                tau = max(decay / 3.5, 1e-4)
                freq = end_hz + (hz - end_hz) * np.exp(-t / tau)
                phase = (
                    2.0
                    * np.pi
                    * np.cumsum(np.concatenate(([0.0], freq[:-1])))
                    / self.sr
                )
            env = np.exp(-6.9 * t / decay)
            att = min(max(2, round(0.0004 * self.sr)), n // 4)
            if att >= 2:
                env[:att] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(att) / att)
            tail = min(max(2, round(0.08 * n)), round(0.025 * self.sr), n // 2)
            if tail >= 2:
                env[-tail:] *= 0.5 + 0.5 * np.cos(
                    np.pi * (np.arange(tail) + 1) / tail
                )
            tone = np.sin(phase) * env
            if tone.size > strike.size:
                strike = np.pad(strike, (0, tone.size - strike.size))
            strike[: tone.size] += level * tone

        self._mix(time, self._unit_peak(strike), gain)

    def granular(
        self,
        time: float,
        duration: float,
        *,
        density: float = 120.0,
        band: Band = (650, 7500),
        grain_duration: Band = (0.0025, 0.022),
        pressure: Contour = DEFAULT_PRESSURE,
        gain: float = 0.15,
    ) -> None:
        """Scatter short colored bursts whose rate and strength follow pressure.

        Density is the expected grains per second at full pressure. Band gives
        the possible center frequencies. Lower grains generally ring longer.
        Gain scales individual grains. Grains decay inside the event window.
        """
        envelope = self._pressure(duration, pressure)
        low, high = self._band(band)
        min_duration, max_duration = grain_duration
        if not 0 < min_duration <= max_duration or not np.isfinite(max_duration):
            raise ValueError("Grain durations must be positive and ordered")
        self._check_gain(density)
        self._check_gain(gain)

        total = envelope.sum()
        if total == 0 or gain == 0:
            return

        count = self.rng.poisson(density * total / self.sr)
        starts = self.rng.choice(envelope.size, size=count, p=envelope / total)
        event = np.zeros_like(envelope)
        for start in starts:
            size = self.rng.random()
            center = high * (low / high) ** size
            # A range of bandwidths and lifetimes avoids a repeating click.
            width = self.rng.uniform(1.25, 2.1)
            length = self._sample_count(min_duration * (max_duration / min_duration) ** size)
            length = min(length, event.size - start)
            if length < 3:
                continue
            strength = envelope[start] ** 0.55 * self.rng.uniform(0.16, 1.0) ** 1.6
            grain = self._burst(length, (center / width, center * width))
            event[start : start + length] += strength * grain
        self._mix(time, event, gain)

    def friction(
        self,
        time: float,
        duration: float,
        *,
        band: Band = (450, 5400),
        end_band: Optional[Band] = None,
        pressure: Contour = DEFAULT_PRESSURE,
        roughness: float = 0.45,
        gain: float = 0.15,
    ) -> None:
        """Add a pressure-shaped rub, swish, scrape or air movement.

        End band continuously sweeps the color during the motion via overlap-add
        bandpass filtering. Roughness controls irregular fluctuations on a 5-25 ms scale.
        """
        if not 0 <= roughness <= 1:
            raise ValueError("Roughness must be between zero and one")
        envelope = self._pressure(duration, pressure)
        white = self.rng.standard_normal(envelope.size)
        if end_band is None:
            noise = self._unit_peak(self._color(white, band))
        else:
            start_lo, start_hi = self._band(band)
            end_lo, end_hi = self._band(end_band)
            n = envelope.size
            hop = max(64, round(0.012 * self.sr))
            win_len = 2 * hop
            warmup = max(256, round(0.025 * self.sr))
            window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(win_len) / win_len)
            origin = warmup + hop
            padded = np.pad(white, (origin, win_len))
            out = np.zeros(n, dtype=np.float64)
            for start in range(-hop, n, hop):
                center = int(np.clip(start + hop, 0, max(n - 1, 0)))
                p = center / max(n - 1, 1)
                lo = start_lo * (end_lo / start_lo) ** p
                hi = start_hi * (end_hi / start_hi) ** p
                seg = padded[origin + start - warmup : origin + start + win_len]
                filtered = self._color(seg, (lo, hi))[warmup:]
                bw_scale = np.sqrt(max((hi - lo) / (self.sr * 0.5), 1e-4))
                filtered = (filtered / bw_scale) * window
                dst_start = max(0, start)
                dst_end = min(n, start + win_len)
                src_start = dst_start - start
                src_end = src_start + (dst_end - dst_start)
                out[dst_start:dst_end] += filtered[src_start:src_end]
            noise = self._unit_peak(out)

        points = np.cumsum(self.rng.uniform(0.005, 0.025, size=int(duration / 0.005) + 2))
        points = np.insert(points, 0, 0.0)
        texture = np.interp(
            np.arange(noise.size) / self.sr,
            points,
            self.rng.uniform(1.0 - roughness, 1.0, size=points.size),
        )
        self._mix(time, self._fade(noise * envelope * texture), gain)

    def vocalize(
        self,
        time: float,
        duration: float,
        *,
        pitch: Contour,
        pressure: Contour = DEFAULT_PRESSURE,
        opening: Contour = 0.55,
        strain: Contour = 0.25,
        band: Band = (150, 7300),
        formants: Sequence[Formant] = DEFAULT_FORMANTS,
        gain: float = 0.25,
    ) -> None:
        """Add a pitched call with independent airflow, resonance and strain.

        Controls are constants or (fraction, value) knots through the event.
        Pitch is in Hz. Pressure, opening and strain are between zero and one.
        Formants are (closed Hz, open Hz, Q, level) resonances over the dry voice.
        """
        count = self._sample_count(duration)
        f0 = self._control(pitch, count, "Pitch", 1.0, self.sr * 0.2)
        force = self._control(pressure, count, "Pressure", 0.0, 1.0)
        mouth = self._control(opening, count, "Opening", 0.0, 1.0)
        tension = self._control(strain, count, "Strain", 0.0, 1.0)
        for closed_hz, open_hz, quality, level in formants:
            if not (0 < closed_hz < self.sr / 2 and 0 < open_hz < self.sr / 2):
                raise ValueError("Formants must be above zero and below Nyquist")
            if not np.isfinite(quality) or quality <= 0:
                raise ValueError("Formant Q must be finite and positive")
            self._check_gain(level)

        slow = self._variation(count, 0.009)
        flutter = self._variation(count, 0.0018)
        drift = tension * (0.013 * slow + 0.025 * flutter)
        f0 *= np.clip(1 + drift, 0.85, 1.15)
        position = np.cumsum(f0) / self.sr
        cycles = position.astype(np.int64)
        phase = position - cycles
        color = self.rng.uniform(-1, 1, cycles[-1] + 1)[cycles]
        alternation = 1 - 2 * (cycles % 2)

        attack = np.clip(0.48 - 0.075 * mouth + 0.065 * tension * color, 0.31, 0.62)
        release = np.clip(0.26 - 0.10 * mouth + 0.036 * tension * flutter, 0.105, 0.33)
        rising = 0.5 - 0.5 * np.cos(np.pi * np.minimum(phase / attack, 1))
        falling = 0.5 + 0.5 * np.cos(np.pi * np.clip((phase - attack) / release, 0, 1))
        pulse_gain = 1 + tension * (0.12 * color + 0.07 * slow + 0.26 * alternation)
        flow = rising * falling * force * np.clip(pulse_gain, 0.4, 1.6)
        voiced = np.diff(flow, prepend=0) / (f0 / self.sr)

        air = self.rng.standard_normal(count)
        flutter_gain = 0.8 + 0.2 * np.sin(2 * np.pi * position)
        breath = air * force * (0.06 + 0.73 * tension**1.6) * flutter_gain
        excitation = self._color(voiced + breath, band)
        voice = 0.47 * excitation

        if formants:
            hop = max(64, round(0.010 * self.sr))
            win_len = 2 * hop
            warmup = max(256, round(0.020 * self.sr))
            window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(win_len) / win_len)
            origin = warmup + hop
            padded_exc = np.pad(excitation, (origin, win_len))
            static_mouth = float(np.ptp(mouth)) < 1e-4

            for closed_hz, open_hz, quality, level in formants:
                if abs(open_hz - closed_hz) < 1e-3 or static_mouth:
                    hz = closed_hz * (open_hz / closed_hz) ** float(mouth[0])
                    b, a = iirpeak(hz, quality, fs=self.sr)
                    voice += level * lfilter(b, a, excitation)
                    continue

                formant_out = np.zeros(count, dtype=np.float64)
                for start in range(-hop, count, hop):
                    lo_idx = max(start, 0)
                    hi_idx = min(start + win_len, count)
                    m_val = (
                        float(np.mean(mouth[lo_idx:hi_idx]))
                        if hi_idx > lo_idx
                        else float(mouth[-1])
                    )
                    hz = closed_hz * (open_hz / closed_hz) ** m_val
                    b, a = iirpeak(hz, quality, fs=self.sr)
                    seg = padded_exc[origin + start - warmup : origin + start + win_len]
                    filtered = lfilter(b, a, seg)[warmup:] * window
                    dst_start = max(0, start)
                    dst_end = min(count, start + win_len)
                    src_start = dst_start - start
                    src_end = src_start + (dst_end - dst_start)
                    formant_out[dst_start:dst_end] += filtered[src_start:src_end]
                voice += level * formant_out

        self._mix(time, self._fade(voice * 0.18), gain)

    def tonal(
        self,
        time: float,
        duration: float,
        *,
        pitch: Contour,
        pressure: Contour = DEFAULT_PRESSURE,
        partials: Sequence[Partial] = ((1.0, 1.0),),
        jitter: Contour = 0.0,
        vibrato_hz: float = 0.0,
        vibrato_depth: float = 0.0,
        gain: float = 0.2,
    ) -> None:
        """Add a sustained or swept harmonic/inharmonic tonal source.

        Useful for door creaks, whistles, servos, sirens, singing glass, and insects.
        Partials are (ratio, level) tuples relative to pitch. Jitter adds organic
        pitch flutter in [0, 1]. Vibrato applies sinusoidal FM depth as a pitch fraction.
        """
        count = self._sample_count(duration)
        f0 = evaluate_contour(pitch, count, "Pitch", 1.0, self.sr * 0.45)
        env = evaluate_contour(pressure, count, "Pressure", 0.0, 1.0)
        jit = evaluate_contour(jitter, count, "Jitter", 0.0, 1.0)
        if not np.isfinite(vibrato_hz) or vibrato_hz < 0:
            raise ValueError("Vibrato frequency must be finite and nonnegative")
        if not np.isfinite(vibrato_depth) or not 0 <= vibrato_depth <= 0.5:
            raise ValueError("Vibrato depth must be in [0, 0.5]")
        if not partials:
            raise ValueError("Partials must be non-empty")
        total_level = 0.0
        for ratio, level in partials:
            if not np.isfinite(ratio) or ratio <= 0:
                raise ValueError("Partial ratios must be finite and positive")
            self._check_gain(level)
            total_level += float(level)
        self._check_gain(gain)

        if np.any(jit > 0):
            slow = self._variation(count, 0.015)
            fast = self._variation(count, 0.0025)
            f0 = f0 * np.clip(1.0 + jit * (0.018 * slow + 0.028 * fast), 0.75, 1.25)
        if vibrato_hz > 0 and vibrato_depth > 0:
            t = np.arange(count, dtype=np.float64) / self.sr
            f0 = f0 * (1.0 + vibrato_depth * np.sin(2.0 * np.pi * vibrato_hz * t))

        base_phase = 2.0 * np.pi * np.cumsum(f0) / self.sr
        signal = np.zeros(count, dtype=np.float64)
        for ratio, level in partials:
            if level == 0:
                continue
            inst_f = ratio * f0
            nyq_weight = np.clip((0.48 * self.sr - inst_f) / (0.06 * self.sr), 0.0, 1.0)
            phi0 = self.rng.uniform(0.0, 2.0 * np.pi)
            signal += level * nyq_weight * np.sin(ratio * base_phase + phi0)

        if total_level > 0:
            signal /= total_level
        self._mix(time, self._fade(signal * env), gain)

    def stick_slip(
        self,
        time: float,
        duration: float,
        *,
        rate: Contour,
        pressure: Contour = DEFAULT_PRESSURE,
        modes: Sequence[Tuple[float, float, float]] = DEFAULT_STICK_SLIP_MODES,
        roughness: float = 0.35,
        grit: float = 0.15,
        dry: float = 0.0,
        gain: float = 0.25,
    ) -> None:
        """Synthesize a stick-slip impulse train exciting a stationary resonant body.

        `rate` controls catch-and-slip frequency in Hz ([2, min(2500, 0.15 * sr)]).
        Sub-sample slip timing, smooth torque drift, and coherent body filtering via
        `modal_filter` are well suited for geared tracks, ratchets, and slow dry catches.
        """
        count = self._sample_count(duration)
        max_rate = min(2500.0, self.sr * 0.15)
        rate_curve = evaluate_contour(rate, count, "Rate", 2.0, max_rate)
        env = evaluate_contour(pressure, count, "Pressure", 0.0, 1.0)
        if not np.isfinite(roughness) or not 0.0 <= roughness <= 1.0:
            raise ValueError("Roughness must be between zero and one")
        if not np.isfinite(grit) or not 0.0 <= grit <= 1.0:
            raise ValueError("Grit must be between zero and one")
        self._check_gain(gain)
        if not modes:
            raise ValueError("Modes must be non-empty")

        max_decay = max(float(dec) for _, dec, _ in modes)
        tail_count = round(min(max_decay, 0.25) * self.sr)
        total_count = count + tail_count

        if roughness > 0.0:
            slow = self._variation(count, 0.028)
            fast = self._variation(count, 0.006)
            rate_mod = rate_curve * np.clip(
                1.0 + roughness * (0.12 * slow + 0.055 * fast), 0.55, 1.65
            )
        else:
            rate_mod = rate_curve

        exc = np.zeros(total_count, dtype=np.float64)
        phase = 0.0
        threshold = float(self.rng.uniform(0.35, 0.85))
        for idx in range(count):
            r_hz = float(rate_mod[idx])
            step = r_hz / self.sr
            phase += step
            if phase >= threshold:
                alpha = float(np.clip((phase - threshold) / max(step, 1e-12), 0.0, 1.0))
                amp = (float(env[idx]) ** 0.75) * float(
                    self.rng.uniform(max(0.25, 1.0 - 0.50 * roughness), 1.0)
                )
                if amp > 0.0:
                    exc[idx] += amp * (1.0 - alpha)
                    if idx > 0:
                        exc[idx - 1] += amp * alpha
                phase -= threshold
                cycle_jit = roughness * 0.28 * float(
                    np.clip(1.0 - (r_hz - 40.0) / 220.0, 0.05, 1.0)
                )
                threshold = float(
                    self.rng.uniform(max(0.30, 1.0 - cycle_jit), 1.0 + cycle_jit)
                )

        p_pole = float(np.exp(-1.0 / max(0.00035 * self.sr, 1.0)))
        exc = lfilter([1.0], [1.0, -p_pole], exc)

        if grit > 0.0:
            rub = self._color(
                self.rng.standard_normal(count),
                (240.0, min(4200.0, self.sr * 0.45)),
            )
            exc[:count] += (grit * 0.03) * env * self._unit_peak(rub)

        out = modal_filter(exc, self.sr, modes, dry=dry)
        if tail_count >= 4:
            taper = min(max(2, round(0.025 * self.sr)), tail_count // 2)
            out[-taper:] *= 0.5 + 0.5 * np.cos(
                np.pi * (np.arange(taper) + 1) / taper
            )
        max_env = max(float(np.max(env)), 0.0)
        if max_env == 0.0 or gain == 0.0:
            return
        self._mix(time, self._fade(self._unit_peak(out) * max_env), gain)

    def resonate(
        self,
        modes: Sequence[Tuple[float, float, float]],
        *,
        dry: float = 0.15,
        gain: float = 1.0,
    ) -> None:
        """Filter this entire layer through a stationary modal object body in place."""
        self._check_gain(gain)
        self.audio = (
            modal_filter(self.audio, self.sr, modes, dry=dry) * gain
        ).astype(np.float32)

    def pluck(
        self,
        time: float,
        duration: float,
        *,
        pitch: float,
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
        tail_fade: float = 0.35,
        gain: float = 0.35,
    ) -> None:
        """Synthesize a plucked string using dual-polarization stiff waveguides and mix at `time`."""
        self._check_gain(gain)
        if gain == 0.0:
            return
        sig = plucked_string(
            duration,
            self.sr,
            pitch,
            t60_low=t60_low,
            t60_hi=t60_hi,
            stiffness=stiffness,
            pluck_pos=pluck_pos,
            pickup_pos=pickup_pos,
            pickup_comb_depth=pickup_comb_depth,
            pulse_ms=pulse_ms,
            vel_cutoff_hz=vel_cutoff_hz,
            pick_lp_hz=pick_lp_hz,
            noise_amount=noise_amount,
            wound_ripple=wound_ripple,
            dual_pol_detune_cents=dual_pol_detune_cents,
            bridge_coupling=bridge_coupling,
            cross_pol_mix=cross_pol_mix,
            pitch_drop_cents=pitch_drop_cents,
            pitch_drop_tau=pitch_drop_tau,
            flux_asymmetry=flux_asymmetry,
            attack_time=0.0,
            tail_fade=tail_fade,
            rng=self.rng,
        )
        self._mix(time, self._fade(self._unit_peak(sig)), gain)

    def saturate(
        self,
        *,
        drive: float = 2.0,
        bias: float = 0.14,
        sag: float = 0.0,
        sag_release: float = 0.12,
        oversample: int = 4,
        dc_block_hz: float = 45.0,
        gain: float = 1.0,
    ) -> None:
        """Apply oversampled asymmetric tube-style saturation to this layer in place."""
        self._check_gain(gain)
        self.audio = (
            saturate(
                self.audio,
                self.sr,
                drive=drive,
                bias=bias,
                sag=sag,
                sag_release=sag_release,
                oversample=oversample,
                dc_block_hz=dc_block_hz,
            )
            * gain
        ).astype(np.float32)

    def cabinet(
        self,
        *,
        preset: str = "guitar_1x12",
        lp_hz: Optional[float] = None,
        hp_hz: Optional[float] = None,
        micro_reflections: bool = True,
        dc_block_hz: float = 35.0,
        gain: float = 1.0,
    ) -> None:
        """Filter this entire layer through a physically modeled speaker cabinet IR in place."""
        self._check_gain(gain)
        self.audio = (
            cabinet_filter(
                self.audio,
                self.sr,
                preset=preset,
                lp_hz=lp_hz,
                hp_hz=hp_hz,
                micro_reflections=micro_reflections,
                dc_block_hz=dc_block_hz,
            )
            * gain
        ).astype(np.float32)

    def add_waveform(self, time: float, audio: np.ndarray, *, gain: float = 1.0) -> None:
        """Place a custom mono synthesis layer at the canvas sample rate."""
        event = np.asarray(audio)
        if event.ndim != 1 or event.dtype.kind not in "fiu" or not np.all(np.isfinite(event)):
            raise ValueError("Waveform must be a finite mono signal")
        self._mix(time, event, gain)

    def _variation(self, count: int, time: float) -> np.ndarray:
        noise = gaussian_filter1d(self.rng.standard_normal(count), self.sr * time)
        noise -= noise.mean()
        return noise / max(noise.std(), 1e-8)

    @staticmethod
    def _control(
        contour: Contour,
        count: int,
        name: str,
        minimum: float,
        maximum: float,
    ) -> np.ndarray:
        return evaluate_contour(contour, count, name, minimum, maximum)

    def _mix(self, time: float, event: np.ndarray, gain: float) -> None:
        if not np.isfinite(time):
            raise ValueError("Event time must be finite")
        self._check_gain(gain)
        start = round(time * self.sr)
        first = max(0, start)
        end = min(self.audio.size, start + event.size)
        if end > first:
            self.audio[first:end] += gain * event[first - start : end - start]

    def _sample_count(self, duration: float) -> int:
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Duration must be finite and positive")
        return max(3, round(duration * self.sr))

    def _band(self, band: Band) -> Band:
        low, high = band
        high = min(high, self.sr * 0.49)
        if not 0 < low < high:
            raise ValueError("Frequency band must have positive width below Nyquist")
        return low, high

    def _color(self, white: np.ndarray, band: Band) -> np.ndarray:
        return sosfilt(
            butter(2, self._band(band), btype="bandpass", fs=self.sr, output="sos"),
            white,
        )

    def _burst(self, length: int, band: Band) -> np.ndarray:
        noise = self._color(self.rng.standard_normal(length), band)
        decay = np.exp(-4.5 * np.linspace(0.0, 1.0, length))
        return self._unit_peak(self._fade(noise * decay))

    def _fade(self, audio: np.ndarray) -> np.ndarray:
        width = min(max(2, round(0.0006 * self.sr)), audio.size // 2)
        fade = np.sin(np.linspace(0.0, np.pi / 2, width)) ** 2
        audio[:width] *= fade
        audio[-width:] *= fade[::-1]
        return audio

    def _pressure(self, duration: float, pressure: Contour) -> np.ndarray:
        return evaluate_contour(
            pressure, self._sample_count(duration), "Pressure", 0.0, 1.0
        )

    @staticmethod
    def _unit_peak(audio: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(audio))
        return audio / max(peak, 1e-12)

    @staticmethod
    def _check_gain(gain: float) -> None:
        if not np.isfinite(gain) or gain < 0:
            raise ValueError("Gain and density must be finite and nonnegative")
