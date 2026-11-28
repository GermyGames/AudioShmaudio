# Physical Instruments, Amps, and DAW Sampling

Use [instruments.py](../scripts/instruments.py) (re-exported by [spectral_foley.py](../scripts/spectral_foley.py) and exposed on `FoleyLayer`) when building realistic plucked strings, tube amplification, speaker cabinets, and DAW-ready multisample packs.

## Core instrument & amp API

- `canvas.foley.pluck(time, duration, *, pitch, t60_low=6.0, t60_hi=1.35, stiffness=5e-5, pluck_pos=0.18, pickup_pos=((0.24, 0.52), (0.155, 0.48)), pickup_comb_depth=0.76, pulse_ms=0.22, pick_lp_hz=5600.0, noise_amount=0.24, wound_ripple=0.0, pitch_drop_cents=10.0, pitch_drop_tau=0.06, gain=0.35)` synthesizes a coupled dual-polarization stiff digital waveguide string with exact loop phase-delay compensation, passive symmetric bridge coupling, plectrum release shaping, and optional pickup comb readout (`pickup_pos=None` for acoustic bridge readout).
- `plucked_string(duration, sr, pitch_hz, ..., attack_time=0.004, return_polarizations=False)` is the standalone buffer function; pass `return_polarizations=True` to access vertical/horizontal displacement (`buf_v`, `buf_h`) for velocity-gated fret buzz before pickup/amp processing.
- `resonant_lowpass(audio, sr, freq_hz, q=1.8)` models 2-pole magnetic pickup RLC coil peaks (`~2.8–3.4 kHz` for single coils, `~2.1–2.7 kHz` for humbuckers).
- `canvas.foley.saturate(*, drive=2.0, bias=0.14, sag=0.0, sag_release=0.12, oversample=4, dc_block_hz=45.0, gain=1.0)` (standalone `saturate(...)`) applies alias-free oversampled $C^\infty$ asymmetric tube saturation, inter-stage coupling-cap DC removal, and envelope-following power-amp sag.
- `canvas.foley.cabinet(*, preset="guitar_1x12", lp_hz=None, hp_hz=None, micro_reflections=True, gain=1.0)` (standalone `cabinet_filter(...)` / `build_cabinet_ir(...)`) convolves with a zero-latency speaker cabinet IR (`"guitar_1x12"`, `"guitar_4x12"`, `"bass_1x15"`, or `"radio"`).
- `measure_pitch(audio, sr, target_hz, *, start_sec=0.45, end_sec=1.45)` measures steady-state fundamental and cents error after the initial tension pitch-drop settles (also exposed via `cli.py analyze --target-hz ...`).
- `write_sfz(path, regions=None, *, groups=None, attack=0.001, release=0.14, amp_veltrack=45.0, global_opcodes=None)` writes DAW/sampler-ready `.sfz` instruments, including `<group>` keyswitch blocks (`sw_lokey`/`sw_hikey`/`sw_last`) and round-robin `seq_length`/`seq_position` opcodes.
- `write_sf2(path, presets, *, bank_name="AudioShmaudio", bit_depth=24, sample_root=None)` packs multi-preset SoundFont 2.04 (`.sf2`) banks directly from WAV region lists (full 24-bit via `smpl` + `sm24` or 16-bit via `smpl`).

```python
from spectral_foley import SpectralCanvas, resonant_lowpass

canvas = SpectralCanvas(duration=4.0, sr=44100, seed=42)
layer = canvas.new_layer()
layer.pluck(
    0.004, 3.99,
    pitch=164.8138,  # E3
    t60_low=7.2, t60_hi=1.28, stiffness=5.7e-5,
    pluck_pos=0.20, pickup_pos=((0.27, 0.48), (0.17, 0.52)),
    pulse_ms=0.19, pick_lp_hz=5860.0, noise_amount=0.28, wound_ripple=0.14,
    gain=0.9,
)
layer.audio = resonant_lowpass(layer.audio, canvas.sr, 3350.0, q=1.95).astype("float32")
layer.saturate(drive=3.15, bias=0.18, sag=0.55)
layer.cabinet(preset="guitar_1x12")
canvas.render_mix(layer.audio, "e3_edge.wav", "e3_edge_spec.png", "E3 Edge", subtype="PCM_24")
```

## Acoustic realism principles (avoiding cheap MIDI timbre)

1. **Model real fretboard geometry.** Compute effective speaking length `L_eff = L_scale * 2**(-fret / 12)` (e.g., `L_scale = 647.7 mm` for 25.5" Strat scale), scale stiffness `B = min(B0 * 2**(fret / 6), 8e-4)` (`B0 ~ 3e-5` to `8e-5`), and derive fractional `pluck_pos = d_pick / L_eff` and `pickup_pos = d_pickup / L_eff` from fixed millimeter bridge distances. Apply mild finger-fret damping (`~0.90x` `t60_low`, `~0.92x` `t60_hi`) when `fret > 0`.
2. **Decouple pick-release width from string period.** Never scale the pick-slip pulse duration inversely with `f0` on bass strings: a `1.8 ms` Gaussian pulse plus a low-cutoff leaky integrator crushes `1–4 kHz` energy and turns `E2` into a rubber thump. Keep `pulse_ms` in `0.12–0.46 ms` across the neck (`vel_cutoff_hz ~ 145–540 Hz`) so wound low strings keep steel snap.
3. **Use passive symmetric bridge coupling, never rotation matrices.** Skew-symmetric 2x2 rotation between vertical and horizontal polarizations shifts loop eigenfrequencies by `±theta/(2*pi)` and causes `-2` to `-3.5` cent pitch drift plus deep phase-cancellation notches on high strings. Keep `bridge_coupling` symmetric (`0.001–0.002`) with `cross_pol_mix ~ 0.18–0.25` and slight positive detuning (`0.4–0.75 cents`).
4. **Preserve upper-register (`G4–E5`) harmonics.** A neck-heavy pickup blend near `mu ~ 0.48` at fret 12 notches out even harmonics (`2nd, 4th`), making upper notes sound flute- or bell-like. Balance neck + middle blends (`~50/50`, `pickup_comb_depth ~ 0.72–0.78`), keep plain-string `t60_hi` generous (`1.35–1.75 s`), and scale pre-amp drive slightly upward across MIDI `40–76` so high notes push warm tube overtones.
5. **Layer mechanical foley before the pickup and amp.** Place pre-pluck `friction` wound scrapes (`1900–5600 Hz`), pick-edge bevel `contact` transients, fret-finger thumps, solid-body wood `modal_filter` (~10% wet), sympathetic open-string modes (~1.5% wet), and subtle velocity-gated fret buzz *before* `resonant_lowpass`, `saturate`, and `cabinet_filter` so string and mechanics share one amp and speaker cone.

## Starting parameter recipes

| Instrument voice | `t60_low` / `t60_hi` | `pulse_ms` / `pick_lp_hz` | `pickup_pos` | `saturate` / `cabinet` |
| --- | --- | --- | --- | --- |
| **Clean Electric Guitar (`v1`)** | `4.0–8.5 s` / `1.1–1.75 s` | `0.24–0.46 ms` / `3300–4900 Hz` | Neck+Mid `((mu_neck, 0.54), (mu_mid, 0.46))` | `drive=1.28, bias=0.11, sag=0.45` + `"guitar_1x12"` |
| **Edge-of-Breakup Guitar (`v2`)** | `3.9–8.5 s` / `1.28–1.75 s` | `0.12–0.24 ms` / `5400–7600 Hz` | Neck+Mid `((mu_neck, 0.48), (mu_mid, 0.52))` | `drive=3.15, bias=0.18, sag=0.55` + `"guitar_1x12"` |
| **Palm-Muted Guitar (`pm`)** | `0.28–0.52 s` / `0.13–0.19 s` | `0.14–0.26 ms` / `4000–5600 Hz` | Bridge+Mid `((mu_mid, 0.50), (mu_brg, 0.50))` | `drive=3.35, bias=0.17, sag=0.55` + `"guitar_1x12"` |
| **Acoustic / Finger Bass** | `5.0–9.0 s` / `0.85–1.50 s` | `0.35–0.75 ms` / `2400–4200 Hz` | `pickup_pos=None` (or P-bass `0.28`) | Acoustic body `modal_filter` (or `drive=1.4` + `"bass_1x15"`) |

## Building DAW-ready multisample packs

1. **Multisample grid & format.** Export `44.1 kHz` / `PCM_24` mono WAVs spaced every minor third (3 semitones across 3 octaves = 13 roots) so samplers pitch-shift at most ±1 semitone. Keep onset tight (`~4 ms` pre-attack room) for crisp grid alignment and fade tails cleanly (`tail_rms_db < -60 dB`).
2. **Velocity staging & SFZ / SF2 mapping.** Separate dynamic layers in both timbre and headroom (e.g., `v1` at `-4.2 dB` peak, `v2` at `-1.2 dB` peak, `pm` at `-2.0 dB` peak) and map them with `write_sfz(..., release=0.14, amp_veltrack=45)` (optionally adding `seq_length`/`seq_position` round-robin variations and `groups` keyswitches) or bundle multiple articulations into one SoundFont bank with `write_sf2(..., bit_depth=24)`.
3. **Verify across registers.** Check every exported WAV with `measure_pitch` (`|pitch_error_cents| < 2.0`), verify `above_7000_hz < 3%` for cab-filtered electric tones, inspect spectrograms and audio directly across low, mid, and high roots, and assemble a short stereo musical demo built *strictly* from the exported WAVs to verify real-world playability.
