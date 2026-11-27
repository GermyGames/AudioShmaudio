import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from inspect import signature
from typing import List, Optional, Tuple

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from procedural import FoleyLayer, modal_filter
from spatial import StereoScene, add_room, distance_filter, pan_mono, soft_limit


@dataclass
class ImpulseEvent:
    time: float
    sharpness: float  # 0.0 (soft thump) to 1.0 (crisp 2ms click)
    gain: float
    seed: Optional[int] = None


class SpectralCanvas:
    """Hybrid 2D spectrogram painting + foley primitive synthesizer."""

    def __init__(
        self,
        duration: float = 0.5,
        sr: int = 44100,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        *,
        seed: Optional[int] = 0,
    ):
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Duration must be finite and positive")
        if sr <= 0:
            raise ValueError("Sample rate must be positive")
        if n_fft is None:
            n_fft = 2048 if sr >= 32000 else 1024
        if hop_length is None:
            hop_length = max(1, n_fft // 8)

        self.duration = duration
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.seed = seed
        self.paint_rng = np.random.default_rng(seed)

        self.n_samples = int(duration * sr)
        self.n_frames = 1 + self.n_samples // hop_length
        self.n_bins = 1 + n_fft // 2

        # Magnitude spectrogram canvas: shape (freq_bins, time_frames)
        self.S = np.zeros((self.n_bins, self.n_frames), dtype=np.float32)

        # Coordinate grids for fast vectorized brush math
        self.freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        self.times = librosa.frames_to_time(
            np.arange(self.n_frames), sr=sr, hop_length=hop_length
        )
        self.F, self.T = np.meshgrid(self.freqs, self.times, indexing="ij")

        # Procedural contacts and textures retain their phase during rendering.
        self.foley = FoleyLayer(self.n_samples, sr, seed)
        self.impulses: List[ImpulseEvent] = []

        # Track rendered (wav_path, spec_path) pairs for progression video export
        self.history: List[Tuple[str, str]] = []

    def reset(self, seed: Optional[int] = None) -> None:
        """Clear synthesized audio, spectral paint, and impulses while keeping pass history."""
        if seed is not None:
            self.seed = seed
        self.paint_rng = np.random.default_rng(self.seed)
        self.S.fill(0.0)
        self.foley.audio.fill(0.0)
        self.foley.rng = np.random.default_rng(self.seed)
        self.impulses.clear()

    def new_layer(self, seed: Optional[int] = None) -> FoleyLayer:
        """Create an independent FoleyLayer sized to this canvas."""
        layer_seed = (
            int(self.paint_rng.integers(0, 2**31 - 1)) if seed is None else seed
        )
        return FoleyLayer(self.n_samples, self.sr, layer_seed)

    def new_scene(self, seed: Optional[int] = None) -> StereoScene:
        """Create a StereoScene sized to this canvas for spatial mixing and room sound."""
        scene_seed = (
            int(self.paint_rng.integers(0, 2**31 - 1)) if seed is None else seed
        )
        return StereoScene(self.n_samples, self.sr, scene_seed)

    # ------------------------------------------------------------------
    # 1. PURE SPECTRAL PAINTING BRUSHES
    # ------------------------------------------------------------------

    def paint_oval(
        self,
        t_center: float,
        f_center: float,
        t_radius: float,
        f_radius_oct: float = 0.5,
        gain: float = 1.0,
    ) -> None:
        """Paints a smooth Gaussian oval (vertical radius in octaves)."""
        t_dist = (self.T - t_center) / max(t_radius, 1e-4)
        safe_f = np.maximum(self.F, 1.0)
        f_dist = np.log2(safe_f / max(f_center, 1.0)) / max(f_radius_oct, 1e-4)

        mask = np.exp(-0.5 * (t_dist**2 + f_dist**2))
        self.S += (mask * gain).astype(np.float32)

    def paint_cloud(
        self,
        t_start: float,
        t_end: float,
        f_low: float,
        f_high: float,
        gain: float = 0.5,
        roughness: float = 0.6,
        seed: Optional[int] = None,
    ) -> None:
        """Paints a soft-edged, textured broadband noise cloud."""
        rng = self.paint_rng if seed is None else np.random.default_rng(seed)

        t_env = np.zeros_like(self.T)
        in_t = (self.T >= t_start) & (self.T <= t_end)
        if np.any(in_t):
            norm_t = (self.T[in_t] - t_start) / max(t_end - t_start, 1e-4)
            t_env[in_t] = np.sin(np.pi * norm_t) ** 0.7

        f_env = np.zeros_like(self.F)
        in_f = (self.F >= f_low) & (self.F <= f_high)
        if np.any(in_f):
            norm_f = (
                np.log2(np.maximum(self.F[in_f], 1.0)) - np.log2(max(f_low, 1.0))
            ) / max(np.log2(f_high / max(f_low, 1.0)), 1e-4)
            f_env[in_f] = np.sin(np.pi * norm_f) ** 0.5

        noise = (1.0 - roughness) + roughness * rng.uniform(0.0, 1.0, size=self.S.shape)
        self.S += (t_env * f_env * noise * gain).astype(np.float32)

    def erase_region(
        self,
        t_start: float,
        t_end: float,
        f_low: float = 0.0,
        f_high: float = 20000.0,
        strength: float = 0.7,
    ) -> None:
        """Soft-edged eraser brush to carve rhythm dips or tame harsh highs."""
        t_center = 0.5 * (t_start + t_end)
        t_rad = max(0.5 * (t_end - t_start), 1e-4)
        t_mask = np.exp(-0.5 * ((self.T - t_center) / t_rad) ** 2)

        bin_hz = self.sr / max(self.n_fft, 1)
        edge_hz = max(2.0 * bin_hz, 20.0)
        if f_low <= 0.0:
            low_mask = np.ones_like(self.F, dtype=np.float32)
        else:
            low_mask = 0.5 * (1.0 + np.tanh((self.F - f_low) / (0.5 * edge_hz)))
        if f_high >= self.sr * 0.5:
            high_mask = np.ones_like(self.F, dtype=np.float32)
        else:
            high_mask = 0.5 * (1.0 + np.tanh((f_high - self.F) / (0.5 * edge_hz)))

        f_mask = (low_mask * high_mask).astype(np.float32)
        attenuation = 1.0 - np.clip(strength, 0.0, 1.0) * t_mask * f_mask
        self.S *= attenuation.astype(np.float32)

    # ------------------------------------------------------------------
    # 2. FOLEY PRIMITIVES
    # ------------------------------------------------------------------

    def add_modal_strike(
        self,
        t_start: float,
        base_hz: float,
        material: str = "wood",
        decay: float = 0.12,
        gain: float = 1.0,
    ) -> None:
        """Stamps inharmonic resonant modes with frequency-dependent decay."""
        presets = {
            "wood": [(1.0, 1.0), (2.58, 0.45), (4.62, 0.2), (6.9, 0.08)],
            "metal": [(1.0, 1.0), (2.76, 0.75), (5.40, 0.55), (8.93, 0.35), (13.3, 0.2)],
            "glass": [(1.0, 1.0), (2.32, 0.65), (4.25, 0.45), (6.63, 0.25)],
        }
        modes = presets.get(material, presets["wood"])

        dt = self.T - t_start
        active_t = dt >= 0

        for ratio, amp in modes:
            mode_hz = base_hz * ratio
            if mode_hz >= self.sr / 2:
                continue
            mode_decay = decay / (ratio**0.65)
            t_env = np.where(active_t, np.exp(-dt / max(mode_decay, 1e-4)), 0.0)
            bw = max(mode_hz * 0.03, 18.0)
            f_env = np.exp(-0.5 * ((self.F - mode_hz) / bw) ** 2)
            self.S += (t_env * f_env * amp * gain).astype(np.float32)

    def add_formant_noise(
        self,
        t_start: float,
        t_end: float,
        formants: List[Tuple[float, float, float]],
        gain: float = 0.6,
        seed: Optional[int] = None,
    ) -> None:
        """
        Paints broadband noise carved through resonant formant peaks.
        formants: list of (start_hz, end_hz, bandwidth_hz) tuples.
        """
        rng = self.paint_rng if seed is None else np.random.default_rng(seed)
        in_t = (self.T >= t_start) & (self.T <= t_end)
        norm_t = np.clip((self.T - t_start) / max(t_end - t_start, 1e-4), 0.0, 1.0)
        t_env = np.where(in_t, np.sin(np.pi * norm_t) ** 0.6, 0.0)

        formant_mask = np.zeros_like(self.S)
        for f_start, f_end, bw in formants:
            center_f = f_start + (f_end - f_start) * norm_t
            formant_mask += np.exp(-0.5 * ((self.F - center_f) / max(bw, 10.0)) ** 2)

        noise = rng.uniform(0.3, 1.0, size=self.S.shape)
        self.S += (t_env * formant_mask * noise * gain).astype(np.float32)

    def scatter_micro_impacts(
        self,
        t_start: float,
        t_end: float,
        count: int = 25,
        freq_range: Tuple[float, float] = (1500.0, 7500.0),
        gain: float = 0.8,
        seed: Optional[int] = None,
    ) -> None:
        """Scatters 1-2 frame vertical spikes for gravel, rain, or debris grit."""
        rng = self.paint_rng if seed is None else np.random.default_rng(seed)
        f_low, f_high = freq_range

        for _ in range(count):
            t_hit = rng.uniform(t_start, t_end)
            frame_idx = int(np.argmin(np.abs(self.times - t_hit)))
            width_frames = rng.integers(1, 3)

            f_c = 2 ** rng.uniform(np.log2(max(f_low, 20.0)), np.log2(f_high))
            bw = f_c * rng.uniform(0.15, 0.45)
            f_profile = np.exp(-0.5 * ((self.freqs - f_c) / bw) ** 2)

            hit_gain = gain * rng.uniform(0.4, 1.2)
            end_frame = min(frame_idx + width_frames, self.n_frames)
            self.S[:, frame_idx:end_frame] += (
                f_profile[:, None] * hit_gain
            ).astype(np.float32)

    # ------------------------------------------------------------------
    # 3. TIME-DOMAIN TRANSIENT LAYER (POST GRIFFIN-LIM)
    # ------------------------------------------------------------------

    def add_impulse(
        self,
        time: float,
        sharpness: float = 0.85,
        gain: float = 0.3,
        seed: Optional[int] = None,
    ) -> None:
        """Queues a crisp time-domain click/burst mixed after Griffin-Lim."""
        self.impulses.append(ImpulseEvent(time, sharpness, gain, seed))

    def _render_impulses(self, length: int) -> np.ndarray:
        out = np.zeros(length, dtype=np.float32)
        for idx, imp in enumerate(self.impulses):
            rng = np.random.default_rng(imp.seed if imp.seed is not None else idx + 100)
            start_sample = int(imp.time * self.sr)
            if start_sample < 0 or start_sample >= length:
                continue

            burst_ms = 2.0 + (1.0 - np.clip(imp.sharpness, 0.0, 1.0)) * 13.0
            burst_len = min(int(self.sr * burst_ms / 1000.0), length - start_sample)
            if burst_len <= 1:
                continue

            t = np.linspace(0.0, 1.0, burst_len, dtype=np.float32)
            env = np.exp(-6.0 * t)
            burst = rng.standard_normal(burst_len).astype(np.float32) * env
            burst /= np.max(np.abs(burst)) + 1e-8
            out[start_sample : start_sample + burst_len] += burst * imp.gain
        return out

    # ------------------------------------------------------------------
    # 4. RECONSTRUCTION, DIAGNOSTICS & VISUAL FEEDBACK
    # ------------------------------------------------------------------

    def render(
        self,
        wav_path: str,
        spec_path: Optional[str] = None,
        pass_label: Optional[str] = None,
        n_iter: int = 64,
        target_peak_db: Optional[float] = -1.0,
        *,
        spectral_gain: float = 0.85,
        subtype: str = "PCM_16",
    ) -> dict:
        """Mix, normalize if requested, and save the WAV and optional feedback."""
        audio = self.foley.audio.copy()
        if spectral_gain and np.any(self.S):
            # librosa 1.x uses rng. The supported 0.10/0.11 branch uses random_state.
            griffinlim = librosa.griffinlim
            seed_parameter = "rng" if "rng" in signature(griffinlim).parameters else "random_state"
            painted = griffinlim(
                self.S,
                n_iter=n_iter,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                length=self.n_samples,
                **{seed_parameter: self.seed},
            )
            spec_peak = np.max(np.abs(painted))
            if spec_peak > 1e-8:
                audio += painted * (spectral_gain / spec_peak)

        if self.impulses:
            audio = audio + self._render_impulses(len(audio))

        return self.render_mix(
            audio, wav_path, spec_path, pass_label,
            target_peak_db=target_peak_db, subtype=subtype,
        )

    def render_mix(
        self,
        audio: np.ndarray,
        wav_path: str,
        spec_path: Optional[str] = None,
        pass_label: Optional[str] = None,
        *,
        target_peak_db: Optional[float] = -1.0,
        subtype: str = "PCM_16",
    ) -> dict:
        """Export a mono or stereo mix at the canvas rate, bypassing reconstruction."""
        audio = np.asarray(audio)
        if (
            audio.ndim not in (1, 2) or len(audio) != self.n_samples or not self.n_samples
            or (audio.ndim == 2 and audio.shape[1] not in (1, 2))
            or audio.dtype.kind not in "fiu" or not np.all(np.isfinite(audio))
        ):
            raise ValueError("Mix must be finite mono or frame-by-channel stereo audio of canvas length")
        if audio.dtype.kind == "i":
            audio = audio.astype(np.float64) / float(np.iinfo(audio.dtype).max)
        elif audio.dtype not in (np.float32, np.float64):
            audio = audio.astype(np.float64)

        raw_peak = float(np.max(np.abs(audio)))
        if target_peak_db is not None:
            if not np.isfinite(target_peak_db):
                raise ValueError("Target peak must be finite")
            if raw_peak > 1e-8:
                target_lin = 10.0 ** (target_peak_db / 20.0)
                audio = audio * (target_lin / raw_peak)

        os.makedirs(os.path.dirname(os.path.abspath(wav_path)), exist_ok=True)
        sf.write(wav_path, audio, self.sr, subtype=subtype)
        diagnostics = compute_diagnostics(audio, self.sr, wav_path, spec_path)

        if spec_path:
            os.makedirs(os.path.dirname(os.path.abspath(spec_path)), exist_ok=True)
            save_feedback_spectrogram(
                audio=audio,
                sr=self.sr,
                spec_path=spec_path,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                diagnostics=diagnostics,
                pass_label=pass_label,
            )
            entry = (os.path.abspath(wav_path), os.path.abspath(spec_path))
            self.history = [item for item in self.history if item != entry]
            self.history.append(entry)
        return diagnostics

    def export_progression_video(
        self,
        output_mp4: str,
        pad_silence: float = 0.35,
    ) -> str:
        """Stitches all recorded (WAV, PNG) passes into a single progression MP4."""
        return stitch_progression_video(self.history, output_mp4, pad_silence=pad_silence)


def _spectrogram_magnitude(audio: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    magnitude = np.abs(librosa.stft(audio.T, n_fft=n_fft, hop_length=hop_length))
    if audio.ndim == 2:
        # Average power across channels so opposite-phase details remain visible.
        return np.sqrt(np.mean(magnitude**2, axis=0))
    return magnitude


def _binned_envelope(signal_1d: np.ndarray, sr: int, max_bins: int = 1800) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = signal_1d.shape[0]
    if n <= max_bins:
        t = np.arange(n, dtype=np.float64) / sr
        return t, signal_1d, signal_1d
    step = max(1, n // max_bins)
    n_bins = int(np.ceil(n / step))
    pad = n_bins * step - n
    if pad:
        padded = np.pad(signal_1d, (0, pad), mode="edge")
    else:
        padded = signal_1d
    blocks = padded.reshape(n_bins, step)
    t = np.minimum((np.arange(n_bins, dtype=np.float64) + 0.5) * step / sr, n / sr)
    return t, blocks.min(axis=1), blocks.max(axis=1)


def compute_diagnostics(
    audio: np.ndarray,
    sr: int,
    wav_path: str,
    spec_path: Optional[str] = None,
) -> dict:
    channels = audio.shape[1] if audio.ndim == 2 else 1
    magnitude = np.abs(audio)
    peak_sample = int(np.argmax(magnitude)) // channels
    peak = float(magnitude.max())
    peak_db = float(20.0 * np.log10(peak + 1e-12))
    sample_power = np.mean(audio**2, axis=1) if audio.ndim == 2 else audio**2
    rms_db = float(20.0 * np.log10(np.sqrt(np.mean(sample_power)) + 1e-12))
    n_fft = 2048 if len(audio) >= 2048 else max(64, 1 << (max(len(audio), 64).bit_length() - 1))
    spectrum = _spectrogram_magnitude(audio, n_fft, max(1, n_fft // 4))
    frame_centroids = librosa.feature.spectral_centroid(S=spectrum, sr=sr, n_fft=n_fft)[0]
    power = spectrum**2
    frame_power = power.sum(axis=0)
    total_power = float(frame_power.sum())
    weighted_centroid = float(np.dot(frame_centroids, frame_power)) / max(total_power, 1e-24)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    denom_power = max(total_power, 1e-24)
    band_defs = (
        ("below_180_hz", freqs < 180.0),
        ("180_800_hz", (freqs >= 180.0) & (freqs < 800.0)),
        ("800_2500_hz", (freqs >= 800.0) & (freqs < 2500.0)),
        ("2500_7000_hz", (freqs >= 2500.0) & (freqs < 7000.0)),
        ("above_7000_hz", freqs >= 7000.0),
    )
    band_energy_pct = {
        name: round(100.0 * float(power[mask].sum()) / denom_power, 2)
        for name, mask in band_defs
    }
    low_energy_pct = band_energy_pct["below_180_hz"]
    block_size = max(1, round(0.01 * sr))
    blocks = np.pad(sample_power, (0, (-len(audio)) % block_size)).reshape(-1, block_size)
    block_power = blocks.mean(axis=1)
    active = block_power > block_power.max() * 1e-4
    active_power = float(np.mean(block_power[active])) if active.any() else 0.0
    active_rms_db = round(float(10 * np.log10(max(active_power, 1e-24))), 2)
    crest_factor_db = round(peak_db - active_rms_db, 2) if active.any() else 0.0
    active_indices = np.flatnonzero(active)
    if active_indices.size:
        active_start_sec = round(float(active_indices[0] * block_size / sr), 4)
        active_end_sec = round(
            float(min(len(audio), (active_indices[-1] + 1) * block_size) / sr), 4
        )
    else:
        active_start_sec = None
        active_end_sec = None
    tail_samples = min(len(audio), max(1, round(0.02 * sr)))
    tail_power = float(np.mean(sample_power[-tail_samples:]))
    tail_rms_db = round(float(10 * np.log10(max(tail_power, 1e-24))), 2)

    diagnostics = {
        "wav_path": os.fspath(wav_path),
        "spec_path": os.fspath(spec_path) if spec_path is not None else None,
        "duration_sec": round(len(audio) / sr, 3),
        "channels": channels,
        "peak_db": round(peak_db, 2),
        "peak_time_sec": round(peak_sample / sr, 4),
        "rms_db": round(rms_db, 2),
        "active_rms_db": active_rms_db,
        "crest_factor_db": crest_factor_db,
        "dc_offset": round(float(np.mean(audio)), 6),
        "active_start_sec": active_start_sec,
        "active_end_sec": active_end_sec,
        "tail_rms_db": tail_rms_db,
        "spectral_centroid_hz": round(float(np.mean(frame_centroids)), 1),
        "weighted_centroid_hz": round(weighted_centroid, 1),
        "energy_below_180_hz_pct": low_energy_pct,
        "band_energy_pct": band_energy_pct,
        "clipped": peak >= 0.999,
    }
    if channels == 2:
        mono = audio.mean(axis=1)
        left, right = (audio - audio.mean(axis=0)).T
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        correlation = float(np.dot(left, right)) / denom if denom > 0 else None
        diagnostics["mono_rms_db"] = round(float(20 * np.log10(np.sqrt(np.mean(mono**2)) + 1e-12)), 2)
        diagnostics["stereo_correlation"] = round(correlation, 3) if correlation is not None else None
    return diagnostics


def save_feedback_spectrogram(
    audio: np.ndarray,
    sr: int,
    spec_path: str,
    n_fft: Optional[int] = None,
    hop_length: Optional[int] = None,
    diagnostics: Optional[dict] = None,
    pass_label: Optional[str] = None,
) -> None:
    if n_fft is None:
        n_fft = 2048 if sr >= 32000 else 1024
    if hop_length is None:
        hop_length = max(1, n_fft // 8)

    S_final = _spectrogram_magnitude(audio, n_fft, hop_length)
    S_db = librosa.amplitude_to_db(
        S_final, ref=max(float(np.max(S_final)), 1e-12), top_db=80.0
    )

    fig = plt.figure(figsize=(9, 5.2), dpi=120)
    fig.subplots_adjust(left=0.09, right=0.92, top=0.91, bottom=0.10)
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.0, 2.7],
        width_ratios=[1.0, 0.032],
        hspace=0.12,
        wspace=0.04,
    )
    ax_wave = fig.add_subplot(gs[0, 0])
    ax_spec = fig.add_subplot(gs[1, 0], sharex=ax_wave)
    cax = fig.add_subplot(gs[1, 1])
    ax_empty = fig.add_subplot(gs[0, 1])
    ax_empty.axis("off")

    duration_sec = len(audio) / sr
    ax_wave.axhline(0.0, color="0.6", linewidth=0.6)
    ax_wave.axhline(1.0, color="0.78", linestyle="--", linewidth=0.6)
    ax_wave.axhline(-1.0, color="0.78", linestyle="--", linewidth=0.6)

    if audio.ndim == 2 and audio.shape[1] == 2:
        t_l, min_l, max_l = _binned_envelope(audio[:, 0], sr)
        t_r, min_r, max_r = _binned_envelope(audio[:, 1], sr)
        ax_wave.fill_between(t_l, min_l, max_l, color="#2b8cbe", alpha=0.55, label="L")
        ax_wave.fill_between(t_r, min_r, max_r, color="#e34a33", alpha=0.45, label="R")
        ax_wave.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    else:
        mono = audio[:, 0] if audio.ndim == 2 else audio
        t_m, min_m, max_m = _binned_envelope(mono, sr)
        if len(mono) <= 1800:
            ax_wave.plot(t_m, mono, color="#2b8cbe", linewidth=0.8)
        else:
            ax_wave.fill_between(t_m, min_m, max_m, color="#2b8cbe", alpha=0.82)

    if diagnostics and float(np.max(np.abs(audio))) > 1e-8:
        ax_wave.axvline(
            diagnostics["peak_time_sec"],
            color="#d95f02",
            linestyle=":",
            linewidth=0.9,
            alpha=0.85,
        )

    ax_wave.set_ylim(-1.05, 1.05)
    ax_wave.set_xlim(0.0, duration_sec)
    ax_wave.set_ylabel("Amp")
    ax_wave.tick_params(labelbottom=False)

    prefix = f"{pass_label} | " if pass_label else ""
    if diagnostics:
        centroid = diagnostics.get("weighted_centroid_hz", diagnostics.get("spectral_centroid_hz", 0))
        rms = diagnostics.get("active_rms_db", diagnostics.get("rms_db", 0))
        rms_label = "Active RMS" if "active_rms_db" in diagnostics else "RMS"
        title = f"{prefix}Centroid: {centroid:.0f} Hz | {rms_label}: {rms:.1f} dB"
        crest = diagnostics.get("crest_factor_db")
        if crest is not None:
            title += f" | Crest: {crest:.1f} dB"
        corr = diagnostics.get("stereo_correlation")
        if corr is not None:
            title += f" | Corr: {corr:+.2f}"
    else:
        title = f"{prefix}Feedback Spectrogram"
    ax_wave.set_title(title, fontsize=10)

    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="log",
        ax=ax_spec,
        cmap="magma",
        vmin=-80.0,
        vmax=0.0,
    )
    ax_spec.set_ylim(40, sr // 2)
    ax_spec.set_xlim(0.0, duration_sec)
    fig.colorbar(img, cax=cax, format="%+2.0f dB")
    plt.savefig(spec_path)
    plt.close(fig)


def stitch_progression_video(
    pairs: List[Tuple[str, str]],
    output_mp4: str,
    pad_silence: float = 0.35,
) -> str:
    """Stitches a list of (wav_path, spec_png_path) pairs into a single MP4."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg to export video.")
    if not pairs:
        raise ValueError("No (WAV, PNG) pairs provided.")

    with tempfile.TemporaryDirectory() as tmpdir:
        sources = [sf.info(wav_path) for wav_path, _ in pairs]
        audio_format = ["-ac", str(max(info.channels for info in sources))]
        if len({info.samplerate for info in sources}) > 1:
            audio_format.extend(["-ar", "44100"])
        segment_paths = []
        for idx, ((wav_path, spec_path), info) in enumerate(zip(pairs, sources)):
            seg_path = os.path.join(tmpdir, f"seg_{idx:03d}.mp4")
            duration = info.duration + pad_silence
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-loop", "1", "-i", spec_path,
                "-i", wav_path,
                "-af", f"apad=pad_dur={pad_silence}",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", *audio_format,
                "-t", f"{duration:.6f}",
                seg_path,
            ]
            subprocess.run(cmd, check=True)
            segment_paths.append(seg_path)

        concat_file = os.path.join(tmpdir, "concat.txt")
        with open(concat_file, "w", encoding="utf-8") as f:
            for seg in segment_paths:
                safe_path = seg.replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")

        os.makedirs(os.path.dirname(os.path.abspath(output_mp4)), exist_ok=True)
        concat_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", concat_file,
            "-c", "copy",
            output_mp4,
        ]
        subprocess.run(concat_cmd, check=True)

    return output_mp4
