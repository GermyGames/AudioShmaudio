---
name: audio-shmaudio
description: Generate original sound effects and procedural foley from a description, including impacts, moving materials, resonant objects, vocalizations, and layered ambience. Synthesize locally and refine with waveform and spectrogram feedback.
---

# AudioShmaudio

Synthesize the effect from scratch with [the engine](scripts/spectral_foley.py). Use procedural events, spectral brushes, spatial scenes, or newly generated local signals. The analysis CLI accepts a local reference WAV when the user explicitly requests reference analysis.

## Build and refine an effect

1. **Plan the source and its perspective.** Sketch how it starts, changes under load, and stops. Identify what makes the sound recognizable, such as a burst of grains, pitch-gliding modes, sustained tonal friction, or the phrasing of a voice. Pick a listening perspective and room feel for layered scenes. Leave room for the lead-in and tail.
2. **Build the sound from reusable parts.** Use `canvas.foley.contact` for impacts and damped or pitch-gliding modes, `granular` for clustered micro-events, `friction` for continuous or band-swept motion, `stick_slip` for geared tracks, ratchets, and slow mechanical catches exciting a coherent body, `tonal` for whistles, chimes, and insects, and `resonate` (or `modal_filter`) to route any layer through a stationary physical object body. See the [primitive API and material recipes](references/acoustic_recipes.md). Use `vocalize` for voiced calls and `canvas.new_layer()` / `canvas.new_scene()` (`StereoScene`) with `render_mix` for spatialized mono or stereo scenes. See [voices and scenes](references/voices_and_scenes.md). Use spectral brushes for broad noise beds and tonal sweeps, setting `spectral_gain` explicitly when mixing with waveform events.
3. **Render and inspect.** Start with timing and recognizable source behavior. Inspect the WAV directly (listen when audio inspection is available) alongside the dual waveform + spectrogram PNG and diagnostics (`crest_factor_db`, `band_energy_pct`, `tail_rms_db`). Call `canvas.reset()` between passes when revising earlier events while preserving progression history, and use `scripts/cli.py compare` to measure changes between passes. Keep the seed and playback level comparable while tuning. The [critique checklist](references/critique_checklist.md) suggests edits for common problems.
4. **Export.** Deliver the final WAV (`PCM_24` when helpful), its feedback spectrogram, and a short account of the meaningful revisions. Export a progression MP4 when `ffmpeg` is available. Save each pass under a distinct name and render through the same canvas to retain pass order. Check final timing, headroom, tail decay, and stereo fold-down.

Run scripts locally with NumPy, SciPy, librosa, SoundFile, and Matplotlib (`SpectralCanvas` defaults to `sr=44100`, `n_fft=2048`, `hop_length=256`). Author differences in timing, contour, and strength separately across passes.

```python
from spectral_foley import SpectralCanvas  # add the skill's scripts directory to sys.path

canvas = SpectralCanvas(duration=0.65, sr=44100, seed=23)
canvas.foley.contact(0.055, 0.018, band=(350, 3400), gain=0.22, modes=((190, 0.14, 0.45, 110),))
canvas.foley.granular(0.059, 0.22, density=330, band=(600, 6400), gain=0.13)
canvas.render("effect.wav", "effect_spec.png", "Initial pass", subtype="PCM_24")
```
