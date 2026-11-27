# Voices and layered scenes

## Shaping a voiced sound

`canvas.foley.vocalize(time, duration, *, pitch, pressure=..., opening=0.55, strain=0.25, band=(150, 7300), formants=..., gain=0.25)` synthesizes a pulse source and routes it through smoothly sweeping resonant formants. Use it for calls, cries, hoots, grunts, laughs, and creature vocalizations. `pitch` is in Hz; `pressure`, `opening`, and `strain` are in `[0, 1]`. Each control accepts a scalar or `(fraction, value)` knots starting at `0` and ending at `1`.

Formants are `(closed_hz, open_hz, Q, level)` tuples. When `opening` moves across the event, each formant sweeps smoothly in frequency between `closed_hz` and `open_hz` via overlap-add filtering. Use defaults for speech-like placement, custom tuples for bird/animal calls, or `()` for a band-shaped pulse source. Keep `pitch < sr / 5` and formants below Nyquist. Layer `friction` for breathy onsets or releases.

```python
from spectral_foley import SpectralCanvas

canvas = SpectralCanvas(duration=1.4, sr=44100, seed=122)
canvas.foley.vocalize(
    0.12, 0.37,
    pitch=((0, 790), (0.18, 1160), (0.53, 1040), (1, 710)),
    pressure=((0, 0), (0.12, 0.55), (0.31, 1), (0.78, 0.44), (1, 0)),
    opening=((0, 0.3), (0.28, 0.94), (0.75, 0.64), (1, 0.23)),
    strain=((0, 0.2), (0.24, 0.12), (0.62, 0.66), (1, 0.44)),
    formants=((1450, 1850, 2.4, 0.57), (2290, 2940, 3.1, 0.86), (4270, 4900, 3.6, 0.40)),
)
canvas.render("voiced_call.wav", "voiced_call_spec.png", "Voiced call", subtype="PCM_24")
```

Shape each syllable independently, varying pitch contour, onset speed, duration, and spacing. Confine stronger `strain` to the gritty part of a phrase and taper `pressure` toward zero so endings read like natural breath.

## Building a spatial stereo scene

1. **Author separate layers.** Use `layer = canvas.new_layer()` to build distinct elements (background bed, foreground impacts, moving mechanisms, distant calls) on isolated mono `FoleyLayer` timelines without clobbering each other's RNG streams or mixing prematurely.
2. **Place sources in space.** Create a `scene = canvas.new_scene()` (`StereoScene` from `spatial.py`, also re-exported by `spectral_foley.py`), and place each layer with `scene.add(layer, pan=..., gain=1.0, distance=0.0, delay=0.0)`.
   - `pan` accepts a scalar or `(fraction, pan)` contour in `[-1.0, 1.0]` for equal-power panning trajectories.
   - `distance` in `[0.0, 1.0]` applies air-absorption lowpass filtering (`distance_filter`) from ~18 kHz down to ~900 Hz so distant elements sit behind foreground transients.
   - `delay` offsets the source in seconds when reusing or staggering layers.
3. **Place the room around the mix.** Call `scene.apply_room(size=0.45, damping=0.55, wet=0.22, predelay=0.012, early_reflections=6)` to add deterministic, decorrelated stereo early reflections and a natural decaying tail. Keep `wet` light (`0.08–0.18`) for outdoors, moderate (`0.20–0.32`) for rooms/corridors, and richer (`0.35–0.50`) for large stone caverns or halls.
4. **Master cleanly.** Call `scene.fade_edges(fade_in=..., fade_out=...)` on continuous ambience beds and `scene.soft_limit(ceiling_db=-1.0, drive=1.0)` when occasional transient peaks overshadow the body. Export via `canvas.render_mix(scene.audio, wav_path, spec_path, pass_label, target_peak_db=..., subtype="PCM_24")`.

```python
from spectral_foley import SpectralCanvas

canvas = SpectralCanvas(duration=2.0, sr=44100, seed=42)
bed = canvas.new_layer()
bed.friction(0.0, 2.0, band=(280, 2600), roughness=0.35, pressure=((0, 0), (0.15, 1), (0.85, 1), (1, 0)), gain=0.14)

move = canvas.new_layer()
move.tonal(0.35, 0.85, pitch=((0, 340), (0.5, 510), (1, 290)), partials=((1, 1.0), (2.7, 0.45)), jitter=0.55, gain=0.2)

scene = canvas.new_scene()
scene.add(bed, pan=0.0, distance=0.25)
scene.add(move, pan=((0, -0.6), (1, 0.55)), distance=0.05)
scene.apply_room(size=0.5, damping=0.5, wet=0.22)
scene.fade_edges(fade_in=0.03, fade_out=0.08)
canvas.render_mix(scene.audio, "scene.wav", "scene_spec.png", "Spatial scene", subtype="PCM_24")
```

Standalone helpers `pan_mono`, `distance_filter`, `add_room`, and `soft_limit` are also importable directly from `spectral_foley` or `spatial` for custom mixing pipelines. Check `mono_rms_db` against `rms_db` and keep `stereo_correlation` healthy (typically `+0.15` to `+0.95` for natural stereo scenes) so important layers survive mono fold-down.
