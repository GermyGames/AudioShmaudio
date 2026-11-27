# Procedural primitives and material recipes

Import `SpectralCanvas` from `scripts/spectral_foley.py` (defaults to `sr=44100`, `n_fft=2048`, `hop_length=256`, or `1024`/`128` below `32000` Hz). A canvas has both a direct waveform timeline (`canvas.foley`) and a magnitude spectrogram (`canvas.S`). Waveform primitives are mixed at their authored times and retain phase. Spectral brushes are converted using Griffin–Lim only when `canvas.S` is non-zero and `spectral_gain` is non-zero.

## Waveform primitives

- `contact(time, duration=0.018, band=(250, 4500), gain=0.3, modes=())` places a short burst of band-limited noise with a fast attack and decay. Optional modes are `(frequency_hz, decay_seconds, relative_level)` or 4-tuple pitch-gliding `(start_hz, decay_seconds, relative_level, end_hz)` tuples. Each mode decays smoothly to zero over its decay time (~60 dB drop plus micro-taper) without end-of-buffer clicks.
- `granular(time, duration, density=120, band=(650, 7500), grain_duration=(0.0025, 0.022), pressure=..., gain=0.15)` generates independently timed noise grains with varied levels and bandwidths. `density` is the expected rate per second at full pressure; `pressure` accepts a scalar or `(fraction, level)` pairs. Lower grains tend to last longer.
- `friction(time, duration, band=(450, 5400), end_band=None, pressure=..., roughness=0.45, gain=0.15)` generates continuously colored noise with irregular 5–25 ms fluctuations. `end_band` performs a continuous overlap-add bandpass sweep from `band` to `end_band` across the event. `roughness` in `[0, 1]` varies surface unevenness; `pressure` accepts a scalar or `(fraction, level)` pairs.
- `tonal(time, duration, *, pitch, pressure=..., partials=((1.0, 1.0),), jitter=0.0, vibrato_hz=0.0, vibrato_depth=0.0, gain=0.2)` synthesizes sustained or swept harmonic/inharmonic resonators. `pitch`, `pressure`, and `jitter` accept scalars or `(fraction, value)` pairs. `partials` are `(ratio, level)` tuples with automatic rolloff near Nyquist. Best for pitched chimes, whistles, singing glass, and insects; avoid naked clean `tonal` sweeps in mechanical/sci-fi foley.
- `stick_slip(time, duration, *, rate, pressure=..., modes=..., roughness=0.35, grit=0.15, dry=0.0, gain=0.25)` excites a single stationary `modes` body bank (`(freq_hz, decay_sec, level)` tuples) with sub-sample phase-coherent slip pulses at repetition `rate` (`[2, min(2500, 0.15 * sr)]` Hz). Best for geared tracks, ratchets, and slow dry catches (`15–80 Hz`).
- `pluck(time, duration, *, pitch, t60_low=6.0, t60_hi=1.35, stiffness=5e-5, pluck_pos=0.18, pickup_pos=..., gain=0.35)` synthesizes a coupled dual-polarization stiff waveguide string (`plucked_string` in `instruments.py`). Combine with `resonant_lowpass`, `saturate(drive=..., bias=..., sag=...)`, and `cabinet(preset="guitar_1x12")` for electric guitars/basses; see [physical instruments and DAW sampling](instruments_and_sampling.md).
- `saturate(*, drive=2.0, bias=0.14, sag=0.0, oversample=4, ...)` and `cabinet(*, preset="guitar_1x12", ...)` apply oversampled asymmetric tube saturation and zero-latency speaker cabinet IR filtering in place on a `FoleyLayer` (standalone `saturate` / `cabinet_filter` also available).
- `resonate(modes, *, dry=0.15, gain=1.0)` (or standalone `modal_filter(audio, sr, modes, dry=0.0)`) filters an entire `FoleyLayer` or mono buffer in place through a parallel bank of unity-calibrated 2-pole resonators so scrapes, airflow, and hits share one physical wood, iron, or steel body.
- `add_waveform(time, audio, gain=1.0)` places a custom mono synthesis signal sampled at the canvas rate. Samples outside the canvas are cropped. See [voices and scenes](voices_and_scenes.md) for `vocalize`, `new_layer()`, and `StereoScene` workflows.

Time and duration are in seconds; frequency is in Hz. Pressure and contour pairs use strictly increasing fractions from `0` to `1` interpolated smoothly (PCHIP). Call `canvas.reset(seed=None)` when you want to rebuild a revised pass from scratch while keeping `canvas.history` intact for progression video export.

## Material starting points

| Effect | Construction |
| --- | --- |
| Gravel, soil, or rubble | Start with a brief 250–2500 Hz contact and a second cluster as weight shifts. Use multiple grain populations (e.g., 4–28 ms grains around 450–3200 Hz and quieter 2–10 ms grains around 2500–7600 Hz). Shape an uneven pressure curve with space between load peaks and few quiet grains on release. Increase grain size and lower density for heavier rubble. |
| Wood, tile, or a hollow object | Combine a short noise contact with unevenly spaced resonances (e.g., 160, 370, 710 Hz for a wood panel, with higher modes decaying sooner). Add a nearly inaudible rebound bounce where physical. Give ceramic a shorter, brighter excitation. |
| Heavy drums, thuds, or slams | For flexible skins/membranes, use 4-tuple downward pitch-gliding `contact` modes like `(155, 0.22, 0.9, 62)`. For rigid solid wood/steel door slams, avoid pitch-gliding sines (which sound like a bouncy tom/dowel); instead layer low-passed broadband sub-thud (`35–190 Hz`), double-staged jamb impacts (`10–20 ms` apart), dense damped `resonate` body modes, and loose hardware rattle. |
| Water drops, drips, or bubbles | Model a drop with a tiny bright surface `contact` and an upward pitch-gliding Minnaert bubble mode, e.g., `(950, 0.045, 0.85, 2200)`, plus a few very short high micro-splash grains. Vary drop pitch, spacing, distance, and room reflections across a cavern or pipe. |
| Metal tap, latch, or sliding part | Use short `2–4 ms` cold clicks and rough `friction` routed through dense damped metallic `resonate` modes (`0.015–0.11 s` decays). For heavy machinery or airlocks, combine relay snaps, roaring low-passed duct `friction` through `resonate`, geared `stick_slip` clatter, and broadband impacts rather than pure sine sweeps or beeps. |
| Mechanical catches, gears, or ratchets | Use `stick_slip` (`15–80 Hz`) through dense damped `modes` or `resonate` for geared tracks, ratchets, and slow dry catches. Note that realistic high-pitched rusty hinge squeaks require self-excited nonlinear friction: linear impulse trains (`20–130 Hz`) read like zippers, swept periodic pulses into multi-formant banks read vocal/creature-like, and noise-fed narrowband bandpass filters sound like whistling into a bottle. |
| Bell, chime, or resonant vessel | Model strike and response separately with a short contact and distinctive group of modes. Give selected low modes longer decays and let brighter modes disappear sooner. Split one or two modes slightly in frequency for slow beating. |
| Plucked strings, electric guitar, or bass | Use `pluck` / `plucked_string` with physical fretboard positions (`pluck_pos`, `pickup_pos`), plectrum-timed `pulse_ms` (`0.12–0.46 ms`), and passive symmetric bridge coupling. Layer pre-pluck wound `friction`, pick `contact`, solid-body `modal_filter`, and sympathetic open strings before `resonant_lowpass` + oversampled `saturate` + `cabinet_filter`. See [instruments_and_sampling.md](instruments_and_sampling.md). |
| Cloth, paper, or foliage | Build swells of rough band-limited `friction` around 300–5500 Hz, varying speed and `end_band` through the gesture. Add sparse short grains for stiff creases or denser bursts for crumpling. Keep sub-180 Hz light unless moving heavy fabric or a body. |
| Wind, fire, rain, or moving air | Layer irregular crackles or droplets using multiple `granular` and sparse sharp `contact` layers. Use sweeping low-roughness `friction` or a spectral cloud for continuous air or ember beds, fading scene edges smoothly. |

### Long, pitched rings

For a tuned church bell, start with five low partials near `0.5`, `1`, `1.2`, `1.5`, and `2` times the prime frequency and adjust ratios for other vessels. Split one or two modes slightly for gentle beating. [Measured church-bell modes](https://www.hibberts.co.uk/wp-content/uploads/2022/01/perrin_charnley_depont_1982.pdf)

Build the lower body first, then add shorter upper modes and sharper strike detail in a later pass. Confirm the low body audibly outlasts high partials, no single peak dominates normalization, and tails fade cleanly before canvas end.

## Spectral brushes and export

- `paint_oval(t_center, f_center, t_radius, f_radius_oct=0.5, gain=1.0)` paints a smooth Gaussian tonal or low-frequency envelope.
- `paint_cloud(t_start, t_end, f_low, f_high, gain=0.5, roughness=0.6, seed=None)` paints a bounded noise bed.
- `erase_region(t_start, t_end, f_low=0, f_high=20000, strength=0.7)` softly attenuates spectral paint with smooth time and frequency transitions.
- `add_modal_strike`, `add_formant_noise`, `scatter_micro_impacts`, and `add_impulse` remain available for hybrid spectral workflows; prefer `canvas.foley` primitives for phase-sensitive transients.

`render(wav_path, spec_path=None, pass_label=None, n_iter=64, target_peak_db=-1.0, *, spectral_gain=0.85, subtype="PCM_16")` renders and saves diagnostics plus dual waveform + spectrogram PNGs. Start `spectral_gain` around `0.03–0.15` when layering spectral texture under direct waveform events. Use `target_peak_db=None` for fixed-level A/B comparisons, `subtype="PCM_24"` for final delivery, and `export_progression_video(output_mp4)` to stitch recorded passes.
