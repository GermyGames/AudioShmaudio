import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/audio-shmaudio/scripts"))
from spectral_foley import SpectralCanvas


def build_campfire_layers(canvas: SpectralCanvas, *, stage: int) -> dict:
    fire_bed = canvas.new_layer()
    crackle_l = canvas.new_layer()
    crackle_r = canvas.new_layer()
    log_pops = canvas.new_layer()
    owl = canvas.new_layer()
    crickets = canvas.new_layer()

    # Stage 1+: Warm ember bed & flickering flame convection swells
    fire_bed.friction(
        0.05, 5.85, band=(65, 320), end_band=(85, 370), roughness=0.45,
        pressure=((0, 0), (0.08, 0.82), (0.38, 0.68), (0.68, 0.94), (0.90, 0.76), (1, 0)),
        gain=0.125,
    )
    fire_bed.friction(
        0.12, 3.30, band=(180, 1250), end_band=(280, 1750), roughness=0.52,
        pressure=((0, 0), (0.18, 0.72), (0.48, 0.45), (0.78, 0.88), (1, 0)),
        gain=0.062,
    )
    fire_bed.friction(
        2.65, 3.20, band=(210, 1400), end_band=(170, 1100), roughness=0.50,
        pressure=((0, 0), (0.22, 0.85), (0.58, 0.55), (0.86, 0.75), (1, 0)),
        gain=0.058,
    )

    if stage >= 2:
        # Stage 2+: Asymmetric crackles, fine ember ticks & burning log sap pops
        crackle_l.granular(
            0.15, 5.55, density=28, band=(900, 4400), grain_duration=(0.0018, 0.0065),
            pressure=((0, 0), (0.10, 0.85), (0.45, 0.55), (0.78, 1.0), (1, 0)),
            gain=0.082,
        )
        crackle_l.granular(
            0.20, 5.45, density=42, band=(2200, 7200), grain_duration=(0.0010, 0.0038),
            pressure=((0, 0), (0.15, 0.7), (0.60, 1.0), (1, 0)),
            gain=0.052,
        )

        crackle_r.granular(
            0.25, 5.40, density=24, band=(1100, 5100), grain_duration=(0.0016, 0.0055),
            pressure=((0, 0), (0.12, 0.65), (0.52, 0.95), (0.85, 0.70), (1, 0)),
            gain=0.078,
        )
        sap_pops = [
            (0.44, 0.007, (1300, 6200), 0.14),
            (1.38, 0.006, (1600, 6800), 0.11),
            (2.12, 0.008, (1100, 5600), 0.15),
            (3.24, 0.006, (1800, 7100), 0.12),
            (4.46, 0.007, (1400, 6400), 0.13),
            (5.16, 0.006, (1500, 5900), 0.095),
        ]
        for t, dur, bnd, gn in sap_pops:
            crackle_r.contact(t, dur, band=bnd, gain=gn, modes=((470, 0.015, 0.25),))

        for t, gn, f_wood in ((0.94, 0.145, 205), (2.68, 0.16, 188), (4.14, 0.135, 224)):
            log_pops.contact(
                t, 0.014, band=(260, 2600), gain=gn,
                modes=((f_wood, 0.032, 0.34), (f_wood * 2.25, 0.018, 0.18)),
            )

    if stage >= 3:
        # Stage 3: Distant nocturnal great-horned owl hoot phrase (soft & hollow behind the fire)
        owl_formants = ((340.0, 405.0, 5.2, 1.5),)
        hoots = [
            (1.35, 0.25, ((0, 355), (0.35, 388), (1, 368)), 0.80),
            (1.86, 0.15, ((0, 348), (0.5, 382), (1, 368)), 0.72),
            (2.08, 0.38, ((0, 372), (0.32, 405), (0.8, 378), (1, 356)), 1.00),
            (2.68, 0.48, ((0, 368), (0.28, 392), (0.75, 355), (1, 332)), 0.90),
        ]
        for t, dur, p_curve, lvl in hoots:
            owl.vocalize(
                t, dur, pitch=p_curve,
                pressure=((0, 0), (0.18, 0.85), (0.45, 1.0), (0.82, 0.45), (1, 0)),
                opening=((0, 0.15), (0.38, 0.82), (1, 0.20)),
                strain=0.05,
                band=(200, 1050),
                formants=owl_formants,
                gain=0.042 * lvl,
            )
            owl.tonal(
                t, dur, pitch=p_curve,
                pressure=((0, 0), (0.20, 0.82), (0.50, 1.0), (0.85, 0.40), (1, 0)),
                partials=((1.0, 1.0), (2.0, 0.08)),
                jitter=0.14,
                gain=0.032 * lvl,
            )

        # Delicate nocturnal cricket stridulation bursts in the far right grass
        for chirp_t in (0.52, 1.48, 3.56, 4.62, 5.28):
            for p_idx in range(3):
                crickets.tonal(
                    chirp_t + p_idx * 0.036, 0.023,
                    pitch=((0, 4340), (1, 4210)),
                    pressure=((0, 0), (0.25, 1.0), (0.75, 0.85), (1, 0)),
                    partials=((1.0, 1.0), (2.0, 0.12)),
                    jitter=0.08,
                    gain=0.026,
                )

    return {
        "fire_bed": fire_bed,
        "crackle_l": crackle_l,
        "crackle_r": crackle_r,
        "log_pops": log_pops,
        "owl": owl,
        "crickets": crickets,
    }


def main() -> None:
    out = ROOT / "output/campfire_night"
    canvas = SpectralCanvas(duration=6.0, sr=44100, seed=409)

    # Pass 1: Warm ember bed & flickering flame swells
    p1 = build_campfire_layers(canvas, stage=1)
    print("Pass 1:", json.dumps(canvas.render_mix(
        p1["fire_bed"].audio,
        out / "pass_01.wav", out / "pass_01.png",
        "Pass 1: Fire Bed & Flame Swells",
    )))

    # Pass 2: Layered ember crackles, fine ticks & sap pops
    canvas.reset()
    p2 = build_campfire_layers(canvas, stage=2)
    mono_p2 = (
        p2["fire_bed"].audio
        + p2["crackle_l"].audio
        + p2["crackle_r"].audio
        + p2["log_pops"].audio
    )
    print("Pass 2:", json.dumps(canvas.render_mix(
        mono_p2,
        out / "pass_02.wav", out / "pass_02.png",
        "Pass 2: Ember Crackles & Sap Pops",
    )))

    # Pass 3: Stereo firepit, distant owl hoots, subtle crickets & forest acoustics
    canvas.reset()
    p3 = build_campfire_layers(canvas, stage=3)
    scene = canvas.new_scene()
    scene.add(p3["fire_bed"], pan=0.0, distance=0.14, gain=0.82)
    scene.add(p3["crackle_l"], pan=-0.42, distance=0.06, gain=1.25)
    scene.add(p3["crackle_r"], pan=0.46, distance=0.10, gain=1.25)
    scene.add(p3["log_pops"], pan=0.06, distance=0.05, gain=1.15)
    scene.add(p3["owl"], pan=-0.65, distance=0.72, gain=0.82)
    scene.add(p3["crickets"], pan=0.68, distance=0.44, gain=0.85)
    scene.apply_room(size=0.52, damping=0.74, wet=0.14, predelay=0.016, early_reflections=5)
    scene.soft_limit(ceiling_db=-1.0, drive=1.95)
    scene.fade_edges(fade_in=0.18, fade_out=0.35)

    print("Pass 3:", json.dumps(canvas.render_mix(
        scene.audio, out / "campfire_night.wav", out / "campfire_night_spec.png",
        "Campfire Night with Distant Owl (Stereo)", subtype="PCM_24",
    )))

    if shutil.which("ffmpeg"):
        canvas.export_progression_video(out / "campfire_night_progression.mp4")


if __name__ == "__main__":
    main()
