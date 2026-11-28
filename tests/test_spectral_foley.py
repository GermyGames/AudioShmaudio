import os
import librosa
import numpy as np
import soundfile as sf
from spectral_foley import SpectralCanvas


def test_canvas_brushes_and_primitives_modify_spectrum():
    c = SpectralCanvas(duration=0.4, sr=22050)
    assert np.all(c.S == 0.0)

    c.paint_oval(0.1, 150.0, 0.03, 0.5, gain=1.0)
    assert np.max(c.S) > 0.5

    before = float(np.sum(c.S))
    c.paint_cloud(0.05, 0.3, 500, 4000, gain=0.3, seed=1)
    c.add_modal_strike(0.1, 300.0, material="wood", decay=0.08, gain=0.8)
    c.add_formant_noise(0.05, 0.3, [(400, 900, 150)], gain=0.4, seed=2)
    c.scatter_micro_impacts(0.05, 0.3, count=10, seed=3)
    assert float(np.sum(c.S)) > before

    after_add = float(np.sum(c.S))
    c.erase_region(0.0, 0.4, 0, 11025, strength=0.5)
    assert float(np.sum(c.S)) < after_add


def test_render_creates_wav_png_and_diagnostics(tmp_path):
    c = SpectralCanvas(duration=0.3, sr=22050)
    c.paint_oval(0.05, 120.0, 0.02, 0.5, gain=1.0)
    c.add_impulse(0.05, sharpness=0.9, gain=0.3, seed=42)
    c.foley.contact(0.12, band=(800, 3600), gain=0.15)

    wav_out = tmp_path / "test.wav"
    png_out = tmp_path / "test.png"
    diag = c.render(str(wav_out), str(png_out), pass_label="Test Pass", n_iter=8)

    assert os.path.exists(wav_out)
    assert os.path.exists(png_out)
    assert diag["clipped"] is False
    assert diag["duration_sec"] == 0.3
    assert len(c.history) == 1


def test_waveform_render_preserves_authored_timing_and_seed(tmp_path, monkeypatch):
    def unexpected_reconstruction(*args, **kwargs):
        raise AssertionError("Waveform-only renders should not run phase reconstruction")

    monkeypatch.setattr(librosa, "griffinlim", unexpected_reconstruction)

    def render(seed, path):
        canvas = SpectralCanvas(duration=0.35, sr=44100, seed=seed)
        canvas.foley.contact(.07, band=(450, 3800), modes=((390, .09, .3),))
        canvas.foley.granular(.16, .1, density=350)
        canvas.foley.friction(.175, .12, end_band=(1200, 6500))
        diagnostics = canvas.render(path, target_peak_db=None, subtype="FLOAT")
        samples, rate = sf.read(path, dtype="float32")
        np.testing.assert_array_equal(samples, canvas.foley.audio)
        assert rate == 44100
        assert np.count_nonzero(samples[:round(.07 * rate)]) == 0
        assert np.count_nonzero(samples[round(.295 * rate):]) == 0
        assert diagnostics["active_rms_db"] > diagnostics["rms_db"]
        assert diagnostics["clipped"] is False
        return samples

    first = render(73, tmp_path / "first.wav")
    np.testing.assert_array_equal(first, render(73, tmp_path / "repeated.wav"))
    assert not np.array_equal(first, render(74, tmp_path / "different.wav"))


def test_pressure_produces_a_quiet_release_for_grains_and_friction():
    canvas = SpectralCanvas(duration=.45, sr=44100, seed=4)
    pressure = ((0, 0), (.2, 1), (.4, 0), (1, 0))
    canvas.foley.granular(.03, .3, pressure=pressure, density=400, grain_duration=(.003, .01))
    canvas.foley.friction(.03, .3, pressure=pressure)

    audio = canvas.foley.audio
    assert np.max(np.abs(audio)) > .03
    # Grains already in flight can continue for their own decay after pressure drops.
    release = round((.03 + .3 * .4 + .01) * canvas.sr)
    assert np.count_nonzero(audio[release:]) == 0


def test_vocalize_tracks_pitch_and_repeats_with_seed():
    def call(seed):
        canvas = SpectralCanvas(duration=.52, sr=24000, seed=seed)
        canvas.foley.vocalize(
            .05, .40,
            pitch=((0, 310), (.33, 310), (.67, 725), (1, 725)),
            strain=((0, .12), (.45, .8), (1, .12)),
            formants=(),
            band=(150, 5600),
        )
        return canvas.foley.audio

    audio = call(83)
    np.testing.assert_array_equal(audio, call(83))
    assert not np.array_equal(audio, call(84))
    assert np.count_nonzero(audio[:1200]) == 0
    assert np.count_nonzero(audio[10800:]) == 0

    def fundamental(segment):
        spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
        frequencies = np.fft.rfftfreq(len(segment), 1 / 24000)
        low = (frequencies > 230) & (frequencies < 800)
        # The low fundamental can be weaker than its first overtone.
        resolved = (spectrum[1:-1] > spectrum[:-2]) & (spectrum[1:-1] > spectrum[2:])
        peaks = np.flatnonzero(low[1:-1] & resolved) + 1
        return frequencies[peaks[spectrum[peaks] > .30 * spectrum[peaks].max()][0]]

    assert abs(fundamental(audio[2160:3840]) - 310) < 25
    assert abs(fundamental(audio[8160:9840]) - 725) < 25


def test_render_mix_preserves_stereo_and_detects_mono_cancellation(tmp_path):
    canvas = SpectralCanvas(duration=.30, sr=24000)
    mono = .2 * np.sin(2 * np.pi * 500 * np.arange(canvas.n_samples) / canvas.sr)
    mono *= np.hanning(canvas.n_samples)
    canvas.foley.add_waveform(0, mono, gain=.8)
    np.testing.assert_allclose(canvas.foley.audio, .8 * mono, atol=1e-8)
    stereo = np.column_stack((canvas.foley.audio, -canvas.foley.audio))
    wav, spec = tmp_path / "room.wav", tmp_path / "room.png"

    diagnostics = canvas.render_mix(stereo, wav, spec, target_peak_db=None, subtype="FLOAT")
    audio, rate = sf.read(wav, dtype="float32")
    np.testing.assert_array_equal(audio, stereo)
    assert rate == canvas.sr
    assert diagnostics["channels"] == 2
    assert diagnostics["stereo_correlation"] == -1
    assert diagnostics["mono_rms_db"] < -100
    assert abs(diagnostics["weighted_centroid_hz"] - 500) < 20
    assert spec.is_file() and len(canvas.history) == 1


def _spectral_centroid_1d(segment: np.ndarray, sr: int) -> float:
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
    freqs = np.fft.rfftfreq(len(segment), 1.0 / sr)
    power = spectrum**2
    return float(np.sum(freqs * power) / max(np.sum(power), 1e-24))


def test_contact_pitch_gliding_modes_and_smooth_tail():
    canvas = SpectralCanvas(duration=0.45, sr=44100, seed=11)
    canvas.foley.contact(
        0.02,
        0.004,
        band=(800, 3000),
        gain=0.4,
        modes=((700.0, 0.30, 1.0, 2100.0),),
    )
    audio = canvas.foley.audio
    sr = canvas.sr
    start_idx = round(0.02 * sr)
    early_seg = audio[start_idx + round(0.01 * sr) : start_idx + round(0.06 * sr)]
    late_seg = audio[start_idx + round(0.18 * sr) : start_idx + round(0.25 * sr)]
    early_freq = _spectral_centroid_1d(early_seg, sr)
    late_freq = _spectral_centroid_1d(late_seg, sr)
    assert early_freq < 1400.0
    assert late_freq > 1750.0
    assert late_freq > early_freq + 500.0
    # Smooth end-of-mode taper reaches zero rather than stepping at -60 dB.
    end_idx = start_idx + round(0.30 * sr)
    assert abs(float(audio[end_idx - 1])) < 1e-4


def test_friction_true_sweep_and_scalar_pressure():
    canvas = SpectralCanvas(duration=0.7, sr=44100, seed=19)
    canvas.foley.friction(
        0.05,
        0.60,
        band=(300, 900),
        end_band=(2600, 6200),
        pressure=1.0,
        roughness=0.1,
        gain=0.3,
    )
    audio = canvas.foley.audio
    sr = canvas.sr
    s0 = round(0.08 * sr)
    seg_early = audio[s0 : s0 + round(0.15 * sr)]
    seg_mid = audio[s0 + round(0.20 * sr) : s0 + round(0.35 * sr)]
    seg_late = audio[s0 + round(0.38 * sr) : s0 + round(0.53 * sr)]
    c_early = _spectral_centroid_1d(seg_early, sr)
    c_mid = _spectral_centroid_1d(seg_mid, sr)
    c_late = _spectral_centroid_1d(seg_late, sr)
    assert 400.0 < c_early < c_mid < c_late
    assert c_mid > c_early + 300.0
    assert c_late > c_mid + 600.0

    # Scalar pressure also works cleanly on granular without raising.
    g_canvas = SpectralCanvas(duration=0.25, sr=44100, seed=5)
    g_canvas.foley.granular(0.02, 0.18, density=250, pressure=0.75, gain=0.2)
    assert np.max(np.abs(g_canvas.foley.audio)) > 0.01


def test_vocalize_sweeps_formant_through_midpoint():
    def render_voice(formants):
        canvas = SpectralCanvas(duration=0.65, sr=24000, seed=31)
        canvas.foley.vocalize(
            0.02,
            0.60,
            pitch=140.0,
            pressure=((0.0, 0.0), (0.08, 1.0), (0.92, 1.0), (1.0, 0.0)),
            opening=((0.0, 0.0), (1.0, 1.0)),
            strain=0.4,
            band=(150, 6000),
            formants=formants,
            gain=0.4,
        )
        return canvas.foley.audio

    dry = render_voice(())
    wet = render_voice(((500.0, 2000.0, 6.0, 4.0),))
    sr = 24000

    def peak_boost_hz(s0, s1):
        i0, i1 = round(s0 * sr), round(s1 * sr)
        win = np.hanning(i1 - i0)
        sd = np.abs(np.fft.rfft(dry[i0:i1] * win))
        sw = np.abs(np.fft.rfft(wet[i0:i1] * win))
        freqs = np.fft.rfftfreq(i1 - i0, 1.0 / sr)
        mask = (freqs >= 350.0) & (freqs <= 2600.0)
        ratio = sw[mask] / (sd[mask] + 1e-6)
        return float(freqs[mask][int(np.argmax(ratio))])

    p_early = peak_boost_hz(0.05, 0.15)
    p_mid = peak_boost_hz(0.27, 0.37)
    p_late = peak_boost_hz(0.49, 0.59)
    assert 480.0 <= p_early <= 750.0
    assert 800.0 <= p_mid <= 1250.0
    assert 1450.0 <= p_late <= 2150.0
    assert p_early < p_mid < p_late


def test_tonal_tracks_pitch_partials_jitter_vibrato_and_nyquist():
    def render_tonal(seed, jitter=0.4):
        canvas = SpectralCanvas(duration=0.5, sr=24000, seed=seed)
        canvas.foley.tonal(
            0.03,
            0.42,
            pitch=((0.0, 300.0), (0.5, 450.0), (1.0, 600.0)),
            pressure=((0.0, 0.0), (0.1, 1.0), (0.9, 1.0), (1.0, 0.0)),
            partials=((1.0, 1.0), (2.0, 0.5), (100.0, 0.4)),  # 100x aliases above Nyquist if unguarded
            jitter=jitter,
            vibrato_hz=6.0,
            vibrato_depth=0.02,
            gain=0.25,
        )
        return canvas.foley.audio

    a = render_tonal(41)
    np.testing.assert_array_equal(a, render_tonal(41))
    assert not np.array_equal(a, render_tonal(42))
    assert np.count_nonzero(a[: round(0.03 * 24000)]) == 0
    assert np.count_nonzero(a[round(0.45 * 24000) :]) == 0
    assert np.all(np.isfinite(a))
    # Fundamental should rise across the sweep.
    c1 = _spectral_centroid_1d(a[round(0.08 * 24000) : round(0.18 * 24000)], 24000)
    c2 = _spectral_centroid_1d(a[round(0.32 * 24000) : round(0.42 * 24000)], 24000)
    assert c2 > c1 + 150.0


def test_canvas_reset_preserves_history_and_restores_seed(tmp_path):
    canvas = SpectralCanvas(duration=0.25, sr=44100, seed=99)
    canvas.foley.contact(0.04, gain=0.3)
    w1, p1 = tmp_path / "pass_01.wav", tmp_path / "pass_01.png"
    canvas.render(str(w1), str(p1), "Pass 1", target_peak_db=None)
    first_audio = canvas.foley.audio.copy()

    canvas.reset()
    assert len(canvas.history) == 1
    assert np.count_nonzero(canvas.foley.audio) == 0
    canvas.foley.contact(0.04, gain=0.3)
    np.testing.assert_array_equal(canvas.foley.audio, first_audio)

    w2, p2 = tmp_path / "pass_02.wav", tmp_path / "pass_02.png"
    canvas.render(str(w2), str(p2), "Pass 2", target_peak_db=None)
    assert len(canvas.history) == 2

    # Re-rendering pass_02 should deduplicate history rather than appending a third item.
    canvas.render(str(w2), str(p2), "Pass 2 updated", target_peak_db=None)
    assert len(canvas.history) == 2


def test_stereo_scene_spatialization_room_and_limiter():
    from spectral_foley import add_room, distance_filter, soft_limit

    canvas = SpectralCanvas(duration=0.6, sr=44100, seed=17)
    layer = canvas.new_layer()
    layer.friction(
        0.05,
        0.45,
        band=(400, 12000),
        pressure=((0.0, 0.0), (0.1, 1.0), (0.9, 1.0), (1.0, 0.0)),
        gain=0.4,
    )

    near = distance_filter(layer.audio, canvas.sr, distance=0.0)
    far = distance_filter(layer.audio, canvas.sr, distance=0.85)
    assert _spectral_centroid_1d(far, canvas.sr) < _spectral_centroid_1d(near, canvas.sr) - 1000.0

    scene = canvas.new_scene()
    scene.add(layer, pan=((0.0, -1.0), (1.0, 1.0)), gain=1.0)
    early_l = float(np.sqrt(np.mean(scene.audio[round(0.06 * canvas.sr) : round(0.14 * canvas.sr), 0] ** 2)))
    early_r = float(np.sqrt(np.mean(scene.audio[round(0.06 * canvas.sr) : round(0.14 * canvas.sr), 1] ** 2)))
    late_l = float(np.sqrt(np.mean(scene.audio[round(0.42 * canvas.sr) : round(0.49 * canvas.sr), 0] ** 2)))
    late_r = float(np.sqrt(np.mean(scene.audio[round(0.42 * canvas.sr) : round(0.49 * canvas.sr), 1] ** 2)))
    assert early_l > early_r * 2.0
    assert late_r > late_l * 2.0

    # wet=0 returns exact dry centered stereo.
    dry_mono = layer.audio.copy()
    dry_room = add_room(dry_mono, canvas.sr, wet=0.0)
    assert dry_room.shape == (canvas.n_samples, 2)
    np.testing.assert_allclose(dry_room[:, 0], dry_room[:, 1], atol=1e-7)

    scene.apply_room(size=0.6, damping=0.5, wet=0.25)
    scene.fade_edges(fade_in=0.02, fade_out=0.05)
    assert np.max(np.abs(scene.audio[-1])) < 1e-6

    spiky = np.array([0.1, -0.2, 3.0, -2.5], dtype=np.float64)
    limited = soft_limit(spiky, ceiling_db=-1.0, drive=1.0)
    ceiling = 10.0 ** (-1.0 / 20.0)
    assert np.max(np.abs(limited)) <= ceiling
    np.testing.assert_allclose(limited[:2], spiky[:2], atol=1e-12)


def test_diagnostics_fields_and_silent_dual_panel_plot(tmp_path):
    canvas = SpectralCanvas(duration=0.35, sr=44100, seed=55)
    wav_silent = tmp_path / "silent.wav"
    png_silent = tmp_path / "silent.png"
    diag_silent = canvas.render(str(wav_silent), str(png_silent), "Silent Pass")
    assert png_silent.is_file() and png_silent.stat().st_size > 1000
    assert diag_silent["active_start_sec"] is None
    assert diag_silent["active_end_sec"] is None
    assert diag_silent["crest_factor_db"] == 0.0

    canvas.foley.contact(0.05, 0.02, band=(400, 3500), gain=0.4, modes=((320, 0.12, 0.5),))
    wav_hit = tmp_path / "hit.wav"
    png_hit = tmp_path / "hit.png"
    diag = canvas.render(str(wav_hit), str(png_hit), "Hit Pass")
    assert png_hit.is_file() and png_hit.stat().st_size > 1000
    assert diag["crest_factor_db"] > 0.0
    assert abs(diag["dc_offset"]) < 1e-2
    assert diag["active_start_sec"] is not None and diag["active_end_sec"] is not None
    assert diag["active_start_sec"] < diag["active_end_sec"]
    assert diag["tail_rms_db"] < -60.0
    total_band_pct = sum(diag["band_energy_pct"].values())
    assert 99.8 <= total_band_pct <= 100.2


def test_cli_analyze_preserves_native_sr_and_compare(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    # 14 kHz tone would be lost if analyze resampled to 22050 Hz by default.
    c1 = SpectralCanvas(duration=0.25, sr=44100, seed=1)
    c1.foley.tonal(0.03, 0.18, pitch=14000.0, pressure=((0, 0), (0.1, 1), (0.9, 1), (1, 0)), gain=0.3)
    wav_a = tmp_path / "a.wav"
    c1.render(str(wav_a), target_peak_db=-1.0)

    c2 = SpectralCanvas(duration=0.25, sr=44100, seed=2)
    c2.foley.tonal(0.03, 0.18, pitch=500.0, pressure=((0, 0), (0.1, 1), (0.9, 1), (1, 0)), gain=0.3)
    wav_b = tmp_path / "b.wav"
    c2.render(str(wav_b), target_peak_db=-1.0)

    cli_script = Path(__file__).resolve().parents[1] / "skills/audio-shmaudio/scripts/cli.py"
    out_png = tmp_path / "analyzed.png"
    res_a = subprocess.run(
        [sys.executable, str(cli_script), "analyze", str(wav_a), "--out", str(out_png)],
        capture_output=True,
        text=True,
        check=True,
    )
    analyzed = json.loads(res_a.stdout)
    assert out_png.is_file()
    assert analyzed["duration_sec"] == 0.25
    assert analyzed["weighted_centroid_hz"] > 12500.0

    res_cmp = subprocess.run(
        [sys.executable, str(cli_script), "compare", str(wav_a), str(wav_b)],
        capture_output=True,
        text=True,
        check=True,
    )
    compared = json.loads(res_cmp.stdout)
    assert compared["deltas"]["weighted_centroid_hz"] < -11000.0
    assert "band_energy_pct" in compared["deltas"]


def test_export_progression_video_end_to_end(tmp_path):
    import shutil
    import subprocess

    assert shutil.which("ffmpeg"), "ffmpeg required for progression video export test"
    canvas = SpectralCanvas(duration=0.25, sr=44100, seed=12)
    canvas.foley.contact(0.04, band=(400, 2400), gain=0.3)
    canvas.render(str(tmp_path / "pass_01.wav"), str(tmp_path / "pass_01.png"), "Pass 1")
    canvas.foley.granular(0.06, 0.12, density=300, gain=0.2)
    canvas.render(str(tmp_path / "pass_02.wav"), str(tmp_path / "pass_02.png"), "Pass 2")

    mp4_path = tmp_path / "progression.mp4"
    out = canvas.export_progression_video(str(mp4_path), pad_silence=0.15)
    assert out == str(mp4_path)
    assert mp4_path.is_file() and mp4_path.stat().st_size > 2000
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(mp4_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert float(probe.stdout.strip()) >= 0.6


def test_modal_filter_and_layer_resonate():
    from spectral_foley import modal_filter

    sr = 44100
    impulse = np.zeros(sr // 2, dtype=np.float64)
    impulse[0] = 1.0
    modes = ((420.0, 0.18, 1.0), (1380.0, 0.12, 0.8))
    filtered = modal_filter(impulse, sr, modes, dry=0.0)
    assert filtered.shape == impulse.shape
    spec = np.abs(np.fft.rfft(filtered * np.hanning(len(filtered))))
    freqs = np.fft.rfftfreq(len(filtered), 1.0 / sr)
    mag_420 = float(spec[np.argmin(np.abs(freqs - 420.0))])
    mag_1380 = float(spec[np.argmin(np.abs(freqs - 1380.0))])
    mag_850 = float(spec[np.argmin(np.abs(freqs - 850.0))])
    assert mag_420 > mag_850 * 4.0
    assert mag_1380 > mag_850 * 2.5

    canvas = SpectralCanvas(duration=0.4, sr=sr, seed=7)
    layer_a = canvas.new_layer(seed=7)
    layer_b = canvas.new_layer(seed=7)
    layer_a.friction(0.02, 0.25, band=(200, 4000), gain=0.3)
    layer_b.friction(0.02, 0.25, band=(200, 4000), gain=0.3)
    layer_a.resonate(modes, dry=0.1, gain=0.9)
    layer_b.resonate(modes, dry=0.1, gain=0.9)
    np.testing.assert_array_equal(layer_a.audio, layer_b.audio)
    assert np.count_nonzero(layer_a.audio[: round(0.02 * sr)]) == 0


def test_stick_slip_creak_tracks_body_modes_and_seed():
    modes = ((320.0, 0.05, 0.7), (920.0, 0.04, 1.0), (1950.0, 0.025, 0.5))

    def render_creak(seed):
        canvas = SpectralCanvas(duration=0.65, sr=44100, seed=seed)
        canvas.foley.stick_slip(
            0.05,
            0.40,
            rate=((0.0, 18.0), (0.5, 95.0), (1.0, 22.0)),
            pressure=((0.0, 0.0), (0.15, 1.0), (0.85, 0.9), (1.0, 0.0)),
            modes=modes,
            roughness=0.4,
            grit=0.2,
            gain=0.3,
        )
        return canvas.foley.audio

    audio = render_creak(61)
    np.testing.assert_array_equal(audio, render_creak(61))
    assert not np.array_equal(audio, render_creak(62))
    assert np.max(np.abs(audio)) > 0.05
    assert np.count_nonzero(audio[: round(0.05 * 44100)]) == 0
    # Active creak + 0.05s decay tail fits inside duration and ends cleanly.
    assert np.count_nonzero(audio[round(0.55 * 44100) :]) == 0

    seg = audio[round(0.08 * 44100) : round(0.42 * 44100)]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), 1.0 / 44100)
    mode_band = ((freqs > 820.0) & (freqs < 1020.0)).astype(bool)
    off_band = ((freqs > 520.0) & (freqs < 680.0)).astype(bool)
    assert float(np.mean(spec[mode_band] ** 2)) > float(np.mean(spec[off_band] ** 2)) * 4.0


def test_stick_slip_high_rate_coherent_and_clean():
    hinge_modes = (
        (960.0, 0.08, 0.8),
        (1580.0, 0.09, 1.0),
        (2420.0, 0.07, 0.7),
        (3380.0, 0.05, 0.4),
    )

    def render_squeak(seed):
        canvas = SpectralCanvas(duration=0.65, sr=44100, seed=seed)
        canvas.foley.stick_slip(
            0.04,
            0.45,
            rate=((0.0, 480.0), (0.35, 980.0), (0.75, 760.0), (1.0, 420.0)),
            pressure=((0.0, 0.0), (0.12, 1.0), (0.85, 0.9), (1.0, 0.0)),
            modes=hinge_modes,
            roughness=0.28,
            grit=0.06,
            dry=0.0,
            gain=0.25,
        )
        return canvas.foley.audio

    audio = render_squeak(88)
    np.testing.assert_array_equal(audio, render_squeak(88))
    assert not np.array_equal(audio, render_squeak(89))
    seg = audio[round(0.08 * 44100) : round(0.45 * 44100)]
    power = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1.0 / 44100)
    total = float(np.sum(power))
    assert total > 0.0
    squeak_pct = float(np.sum(power[(freqs >= 650.0) & (freqs <= 3800.0)])) / total
    ultrasonic_pct = float(np.sum(power[freqs > 8000.0])) / total
    assert squeak_pct > 0.85
    assert ultrasonic_pct < 0.03


def test_plucked_string_and_layer_pluck_accuracy_and_seed():
    from spectral_foley import measure_pitch, plucked_string

    sr = 44100

    def render_pluck(seed: int, *, t60_low: float = 6.5, t60_hi: float = 1.5) -> np.ndarray:
        canvas = SpectralCanvas(duration=1.8, sr=sr, seed=seed)
        canvas.foley.pluck(
            0.05,
            1.7,
            pitch=82.4069,
            t60_low=t60_low,
            t60_hi=t60_hi,
            wound_ripple=0.14,
            gain=0.6,
        )
        return canvas.foley.audio

    e2_a = render_pluck(11)
    e2_b = render_pluck(11)
    e2_diff = render_pluck(12)
    np.testing.assert_array_equal(e2_a, e2_b)
    assert not np.array_equal(e2_a, e2_diff)
    assert np.count_nonzero(e2_a[: round(0.05 * sr)]) == 0

    hz_e2, cents_e2 = measure_pitch(e2_a[round(0.05 * sr) :], sr, 82.4069)
    assert hz_e2 is not None and cents_e2 is not None
    assert abs(cents_e2) < 1.5

    muted = render_pluck(11, t60_low=0.45, t60_hi=0.16)
    late_sust = float(np.sqrt(np.mean(e2_a[round(0.7 * sr) : round(1.4 * sr)] ** 2)))
    late_mute = float(np.sqrt(np.mean(muted[round(0.7 * sr) : round(1.4 * sr)] ** 2)))
    assert late_mute < late_sust * 0.25

    e5 = plucked_string(
        1.8,
        sr,
        659.2551,
        t60_low=4.0,
        t60_hi=1.65,
        pickup_pos=((0.24, 0.52), (0.155, 0.48)),
        pickup_comb_depth=0.76,
        seed=5,
    )
    assert np.all(np.isfinite(e5))
    hz_e5, cents_e5 = measure_pitch(e5, sr, 659.2551)
    assert hz_e5 is not None and cents_e5 is not None
    assert abs(cents_e5) < 1.5

    seg = e5[round(0.03 * sr) : round(0.45 * sr)]
    power = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
    p1 = float(np.max(power[np.abs(freqs - 659.2551) < 25.0]))
    p2 = float(np.max(power[np.abs(freqs - 2.0 * 659.2551) < 25.0]))
    p3 = float(np.max(power[np.abs(freqs - 3.0 * 659.2551) < 25.0]))
    assert (p2 + p3) / max(p1, 1e-18) > 0.01

    tail_rms = float(np.sqrt(np.mean(e5[-round(0.05 * sr) :] ** 2)))
    assert tail_rms < 1e-3


def test_saturate_oversampled_harmonics_dc_and_sag():
    from spectral_foley import saturate

    sr = 44100
    t = np.arange(sr // 2, dtype=np.float64) / sr
    sine = 0.65 * np.sin(2.0 * np.pi * 220.0 * t)
    unblocked = saturate(sine, sr, drive=3.0, bias=0.18, sag=0.0, oversample=4, dc_block_hz=0.0)
    out = saturate(sine, sr, drive=3.0, bias=0.18, sag=0.0, oversample=4)
    assert out.shape == sine.shape
    assert np.all(np.isfinite(out))
    assert abs(float(np.mean(unblocked))) > 0.05
    assert abs(float(np.mean(out))) < 5e-3
    assert abs(float(np.mean(out[round(0.1 * sr) : round(0.45 * sr)]))) < 2e-4
    np.testing.assert_array_equal(out, saturate(sine, sr, drive=3.0, bias=0.18, sag=0.0, oversample=4))

    seg = out[round(0.1 * sr) : round(0.45 * sr)]
    power = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
    p1 = float(np.max(power[np.abs(freqs - 220.0) < 12.0]))
    p2 = float(np.max(power[np.abs(freqs - 440.0) < 12.0]))
    p3 = float(np.max(power[np.abs(freqs - 660.0) < 12.0]))
    assert p2 > p1 * 1e-4
    assert p3 > p1 * 1e-4

    hot = np.sin(2.0 * np.pi * 220.0 * (np.arange(round(0.6 * sr), dtype=np.float64) / sr))
    no_sag = saturate(hot, sr, drive=3.2, bias=0.14, sag=0.0, oversample=2)
    with_sag = saturate(hot, sr, drive=3.2, bias=0.14, sag=0.85, sag_release=0.05, oversample=2)
    rms_no_sag = float(np.sqrt(np.mean(no_sag[round(0.25 * sr) : round(0.55 * sr)] ** 2)))
    rms_sag = float(np.sqrt(np.mean(with_sag[round(0.25 * sr) : round(0.55 * sr)] ** 2)))
    assert rms_sag < rms_no_sag * 0.90


def test_cabinet_filter_and_resonant_lowpass():
    from spectral_foley import build_cabinet_ir, cabinet_filter, resonant_lowpass

    sr = 44100
    ir = build_cabinet_ir(sr, preset="guitar_1x12")
    assert np.all(np.isfinite(ir))
    assert int(np.argmax(np.abs(ir))) <= 8

    rng = np.random.default_rng(77)
    noise = rng.normal(0.0, 0.25, size=sr)

    cab = cabinet_filter(noise, sr, preset="guitar_1x12")
    freqs = np.fft.rfftfreq(len(cab), 1.0 / sr)
    power = np.abs(np.fft.rfft(cab * np.hanning(len(cab)))) ** 2
    total = float(np.sum(power))
    assert total > 0.0
    assert float(np.sum(power[freqs > 7000.0])) / total < 0.03
    assert float(np.sum(power[(freqs >= 120.0) & (freqs <= 4500.0)])) / total > 0.85

    radio = cabinet_filter(noise, sr, preset="radio")
    radio_pow = np.abs(np.fft.rfft(radio * np.hanning(len(radio)))) ** 2
    radio_tot = float(np.sum(radio_pow))
    assert float(np.sum(radio_pow[(freqs >= 350.0) & (freqs <= 3200.0)])) / radio_tot > 0.85

    canvas = SpectralCanvas(duration=0.4, sr=sr, seed=9)
    layer = canvas.new_layer(seed=9)
    layer.pluck(0.02, 0.35, pitch=196.0, gain=0.5)
    raw = layer.audio.copy()
    layer.saturate(drive=2.4, sag=0.2)
    saturated = layer.audio.copy()
    layer.cabinet(preset="guitar_4x12")
    assert not np.array_equal(raw, saturated)
    assert not np.array_equal(saturated, layer.audio)

    impulse = np.zeros(sr // 4, dtype=np.float64)
    impulse[0] = 1.0
    res = resonant_lowpass(impulse, sr, freq_hz=3000.0, q=2.5)
    res_spec = np.abs(np.fft.rfft(res))
    res_freqs = np.fft.rfftfreq(len(res), 1.0 / sr)
    mag_3000 = float(res_spec[np.argmin(np.abs(res_freqs - 3000.0))])
    mag_500 = float(res_spec[np.argmin(np.abs(res_freqs - 500.0))])
    mag_9000 = float(res_spec[np.argmin(np.abs(res_freqs - 9000.0))])
    assert mag_3000 > mag_500 * 1.5
    assert mag_3000 > mag_9000 * 8.0


def test_measure_pitch_write_sfz_and_cli_target_hz(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    from spectral_foley import measure_pitch, write_sfz

    sr = 44100
    target = 196.0
    t = np.arange(round(1.6 * sr), dtype=np.float64) / sr
    tone = 0.5 * np.sin(2.0 * np.pi * target * t)

    measured_hz, err_cents = measure_pitch(tone, sr, target)
    assert measured_hz is not None and err_cents is not None
    assert abs(err_cents) < 0.2

    wav_path = tmp_path / "g3.wav"
    sf.write(wav_path, tone, sr, subtype="PCM_24")

    sfz_path = tmp_path / "test_patch.sfz"
    written = write_sfz(
        sfz_path,
        [{"sample": "g3.wav", "pitch_keycenter": 55, "lokey": 53, "hikey": 57, "lovel": 1, "hivel": 127}],
        header_comment="Unit test patch",
    )
    assert written == sfz_path and sfz_path.is_file()
    sfz_text = sfz_path.read_text(encoding="utf-8")
    assert "<global>" in sfz_text
    assert "<region>" in sfz_text
    assert "sample=g3.wav" in sfz_text

    cli_script = Path(__file__).resolve().parents[1] / "skills/audio-shmaudio/scripts/cli.py"
    out_png = tmp_path / "g3.png"
    res = subprocess.run(
        [sys.executable, str(cli_script), "analyze", str(wav_path), "--target-hz", str(target), "--out", str(out_png)],
        env={**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl")},
        capture_output=True,
        text=True,
        check=True,
    )
    analyzed = json.loads(res.stdout)
    assert out_png.is_file()
    assert analyzed["target_hz"] == 196.0
    assert analyzed["measured_hz"] is not None
    assert abs(analyzed["pitch_error_cents"]) < 0.2


def test_write_sf2_structure_and_fluidsynth_playback(tmp_path):
    import shutil
    import struct
    import subprocess
    from spectral_foley import measure_pitch, write_sf2

    sr = 44100
    n = round(1.6 * sr)
    t = np.arange(n, dtype=np.float64) / sr
    env = np.ones(n, dtype=np.float64)
    ramp = round(0.002 * sr)
    env[:ramp] = np.linspace(0.0, 1.0, ramp)
    env[-ramp:] = np.linspace(1.0, 0.0, ramp)

    a3 = (10.0 ** (-3.0 / 20.0)) * np.sin(2.0 * np.pi * 220.0 * t) * env
    a4_quiet = (10.0 ** (-45.0 / 20.0)) * np.sin(2.0 * np.pi * 440.0 * t) * env
    wav_a3 = tmp_path / "a3.wav"
    wav_a4 = tmp_path / "a4_quiet.wav"
    sf.write(wav_a3, a3, sr, subtype="PCM_24")
    sf.write(wav_a4, a4_quiet, sr, subtype="PCM_24")

    presets = [
        {
            "name": "Test Tone",
            "bank": 0,
            "program": 0,
            "attack": 0.001,
            "release": 0.14,
            "amp_veltrack": 45.0,
            "regions": [
                {
                    "sample": "a3.wav",
                    "pitch_keycenter": 57,
                    "lokey": 55,
                    "hikey": 59,
                    "lovel": 1,
                    "hivel": 127,
                }
            ],
        },
        {
            "name": "Quiet High",
            "bank": 0,
            "program": 1,
            "attack": 0.001,
            "release": 0.14,
            "amp_veltrack": 100.0,
            "regions": [
                {
                    "sample": "a4_quiet.wav",
                    "pitch_keycenter": 69,
                    "lokey": 67,
                    "hikey": 71,
                    "lovel": 1,
                    "hivel": 127,
                }
            ],
        },
    ]

    sf2_24 = tmp_path / "test24.sf2"
    sf2_24_copy = tmp_path / "test24_copy.sf2"
    sf2_16 = tmp_path / "test16.sf2"
    write_sf2(sf2_24, presets, bit_depth=24, sample_root=tmp_path)
    write_sf2(sf2_24_copy, presets, bit_depth=24, sample_root=tmp_path)
    write_sf2(sf2_16, presets, bit_depth=16, sample_root=tmp_path)
    assert sf2_24.read_bytes() == sf2_24_copy.read_bytes()

    def _parse_chunks(blob: bytes) -> list:
        items = []
        pos = 0
        while pos < len(blob):
            cid = blob[pos : pos + 4]
            csize = struct.unpack_from("<I", blob, pos + 4)[0]
            payload = blob[pos + 8 : pos + 8 + csize]
            items.append((cid, payload))
            pos += 8 + csize + (csize % 2)
        assert pos == len(blob)
        return items

    def _inspect_sf2(path_obj, expected_depth: int) -> None:
        raw = path_obj.read_bytes()
        assert raw[:4] == b"RIFF" and raw[8:12] == b"sfbk"
        riff_size = struct.unpack_from("<I", raw, 4)[0]
        assert riff_size == len(raw) - 8
        lists = [(p[:4], _parse_chunks(p[4:])) for cid, p in _parse_chunks(raw[12:]) if cid == b"LIST"]
        assert [lt for lt, _ in lists] == [b"INFO", b"sdta", b"pdta"]

        info = dict(lists[0][1])
        sdta = dict(lists[1][1])
        pdta_list = lists[2][1]
        expected_pdta = [b"phdr", b"pbag", b"pmod", b"pgen", b"inst", b"ibag", b"imod", b"igen", b"shdr"]
        assert [cid for cid, _ in pdta_list] == expected_pdta

        if expected_depth == 24:
            assert struct.unpack("<HH", info[b"ifil"]) == (2, 4)
            assert b"sm24" in sdta
            assert len(sdta[b"sm24"]) == len(sdta[b"smpl"]) // 2
            assert len(sdta[b"sm24"]) % 2 == 0
        else:
            assert struct.unpack("<HH", info[b"ifil"]) == (2, 1)
            assert b"sm24" not in sdta

        pdta = dict(pdta_list)
        rec_sizes = {b"phdr": 38, b"pbag": 4, b"pmod": 10, b"pgen": 4, b"inst": 22, b"ibag": 4, b"imod": 10, b"igen": 4, b"shdr": 46}
        for tag, rsz in rec_sizes.items():
            assert len(pdta[tag]) >= rsz and len(pdta[tag]) % rsz == 0
        assert pdta[b"phdr"][-38 : -38 + 4] == b"EOP\x00"
        assert pdta[b"inst"][-22 : -22 + 4] == b"EOI\x00"
        assert pdta[b"shdr"][-46 : -46 + 4] == b"EOS\x00"

    _inspect_sf2(sf2_24, 24)
    _inspect_sf2(sf2_16, 16)

    fluidsynth_bin = shutil.which("fluidsynth")
    if not fluidsynth_bin:
        return

    def _vlq(val: int) -> bytes:
        buf = [val & 0x7F]
        val >>= 7
        while val:
            buf.append((val & 0x7F) | 0x80)
            val >>= 7
        return bytes(reversed(buf))

    def _write_midi(midi_path, program: int, note: int, vel: int, dur_sec: float = 1.35) -> None:
        tpb = 480
        track = bytearray()
        track.extend(_vlq(0) + b"\xFF\x51\x03" + int(500000).to_bytes(3, "big"))
        track.extend(_vlq(0) + bytes([0xC0, program]))
        track.extend(_vlq(round(0.04 * 2 * tpb)) + bytes([0x90, note, vel]))
        track.extend(_vlq(round(dur_sec * 2 * tpb)) + bytes([0x80, note, 0]))
        track.extend(_vlq(round(0.20 * 2 * tpb)) + b"\xFF\x2F\x00")
        hdr = b"MThd" + struct.pack(">IHHH", 6, 0, 1, tpb)
        trk = b"MTrk" + struct.pack(">I", len(track)) + bytes(track)
        midi_path.write_bytes(hdr + trk)

    def _render_fs(sf2_path, mid_path, out_wav) -> np.ndarray:
        proc = subprocess.run(
            [fluidsynth_bin, "-ni", "-a", "file", "-F", str(out_wav), "-r", "44100", "-g", "0.8", "-R", "0", "-C", "0", str(sf2_path), str(mid_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        fs_errors = [ln for ln in (proc.stderr or "").splitlines() if "fluidsynth: error:" in ln.lower()]
        assert not fs_errors, fs_errors
        audio, rsr = sf.read(out_wav, dtype="float64", always_2d=True)
        assert rsr == 44100 and len(audio) > sr
        return np.mean(audio, axis=1)

    mid_p0_hi = tmp_path / "p0_hi.mid"
    mid_p0_lo = tmp_path / "p0_lo.mid"
    mid_p1_quiet = tmp_path / "p1_quiet.mid"
    _write_midi(mid_p0_hi, program=0, note=57, vel=120)
    _write_midi(mid_p0_lo, program=0, note=57, vel=45)
    _write_midi(mid_p1_quiet, program=1, note=69, vel=127)

    ren24_p0_hi = _render_fs(sf2_24, mid_p0_hi, tmp_path / "ren24_p0_hi.wav")
    ren16_p0_hi = _render_fs(sf2_16, mid_p0_hi, tmp_path / "ren16_p0_hi.wav")
    ren24_p0_lo = _render_fs(sf2_24, mid_p0_lo, tmp_path / "ren24_p0_lo.wav")
    ren24_p1_q = _render_fs(sf2_24, mid_p1_quiet, tmp_path / "ren24_p1_q.wav")

    for ren, f_exp in ((ren24_p0_hi, 220.0), (ren16_p0_hi, 220.0), (ren24_p1_q, 440.0)):
        hz, cents = measure_pitch(ren, sr, f_exp, start_sec=0.35, end_sec=1.20)
        assert hz is not None and cents is not None
        assert abs(cents) < 1.5

    seg_q = ren24_p1_q[round(0.35 * sr) : round(1.20 * sr)]
    q_pow = np.abs(np.fft.rfft(seg_q * np.hanning(len(seg_q)))) ** 2
    q_freqs = np.fft.rfftfreq(len(seg_q), 1.0 / sr)
    fund_ratio = float(np.sum(q_pow[np.abs(q_freqs - 440.0) <= 15.0])) / max(float(np.sum(q_pow)), 1e-18)
    assert fund_ratio > 0.95

    rms_hi = float(np.sqrt(np.mean(ren24_p0_hi[round(0.35 * sr) : round(1.20 * sr)] ** 2)))
    rms_lo = float(np.sqrt(np.mean(ren24_p0_lo[round(0.35 * sr) : round(1.20 * sr)] ** 2)))
    assert rms_hi * 0.15 < rms_lo < rms_hi * 0.75


def test_write_sfz_groups_and_round_robin_opcodes(tmp_path):
    import pytest
    from spectral_foley import write_sfz

    groups = [
        {
            "opcodes": {"sw_last": 24, "sw_label": "Lead", "ampeg_release": 0.13},
            "regions": [
                {
                    "sample": f"lead_rr{rr}.wav",
                    "pitch_keycenter": 40,
                    "lokey": 39,
                    "hikey": 41,
                    "seq_length": 4,
                    "seq_position": rr,
                }
                for rr in range(1, 5)
            ],
        },
        {
            "opcodes": {"sw_last": 25, "sw_label": "Palm Mute", "ampeg_release": 0.08},
            "regions": [
                {
                    "sample": f"pm_rr{rr}.wav",
                    "pitch_keycenter": 40,
                    "lokey": 39,
                    "hikey": 41,
                    "seq_length": 4,
                    "seq_position": rr,
                }
                for rr in range(1, 5)
            ],
        },
    ]
    sfz_path = tmp_path / "master.sfz"
    out = write_sfz(
        sfz_path,
        groups=groups,
        attack=0.001,
        release=0.12,
        amp_veltrack=42.0,
        global_opcodes={"sw_lokey": 24, "sw_hikey": 25, "sw_default": 24},
        header_comment="Keyswitched RR test",
    )
    assert out == sfz_path and sfz_path.is_file()
    text = sfz_path.read_text(encoding="utf-8")
    assert "// Keyswitched RR test" in text
    assert "<global>" in text
    assert "sw_lokey=24" in text and "sw_hikey=25" in text and "sw_default=24" in text
    assert text.count("<group>") == 2
    assert text.count("<region>") == 8
    assert "sw_last=24" in text and "sw_last=25" in text
    for rr in range(1, 5):
        assert f"seq_position={rr}" in text
        assert f"sample=lead_rr{rr}.wav" in text
        assert f"sample=pm_rr{rr}.wav" in text
    assert text.count("seq_length=4") == 8

    with pytest.raises(ValueError):
        write_sfz(tmp_path / "bad1.sfz")
    with pytest.raises(ValueError):
        write_sfz(tmp_path / "bad2.sfz", regions=[{"sample": "a.wav"}], groups=groups)
    with pytest.raises(ValueError):
        write_sfz(tmp_path / "bad3.sfz", groups=[])
    with pytest.raises(ValueError):
        write_sfz(tmp_path / "bad4.sfz", groups=[{"opcodes": {"sw_last": 24}, "regions": []}])


