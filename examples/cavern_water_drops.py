import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/audio-shmaudio/scripts"))
from spectral_foley import SpectralCanvas


def build_drops(canvas: SpectralCanvas, *, with_bubbles: bool) -> dict:
    near_l = canvas.new_layer()
    mid_r = canvas.new_layer()
    far_l = canvas.new_layer()
    trickle = canvas.new_layer()

    # Foreground left pair (close stalactite drips)
    near_specs = [
        (0.18, 760.0, 1480.0, 0.044, 0.90),
        (0.45, 980.0, 1860.0, 0.036, 0.66),
        (2.74, 700.0, 1390.0, 0.048, 0.96),
    ]
    for t, f0, f1, dec, lvl in near_specs:
        if with_bubbles:
            # Gentle liquid break followed immediately by dominant Minnaert bubble resonance
            near_l.contact(t, 0.0028, band=(2200, 6400), gain=0.085 * lvl)
            near_l.contact(
                t + 0.0012, 0.0025, band=(1200, 3200), gain=0.23 * lvl,
                modes=((f0, dec, 3.2, f1),),
            )
            near_l.granular(
                t + 0.005, 0.045, density=110, band=(2400, 7200),
                grain_duration=(0.002, 0.007), pressure=((0, 1), (0.35, 0.35), (1, 0)),
                gain=0.032 * lvl,
            )
        else:
            near_l.contact(t, 0.004, band=(1600, 5600), gain=0.20 * lvl, modes=((f0, 0.025, 0.45),))

    # Mid-distance right drops
    mid_specs = [
        (1.06, 890.0, 1710.0, 0.040, 0.86),
        (2.18, 1180.0, 2120.0, 0.032, 0.74),
        (3.32, 820.0, 1560.0, 0.042, 0.80),
    ]
    for t, f0, f1, dec, lvl in mid_specs:
        if with_bubbles:
            mid_r.contact(t, 0.0028, band=(2000, 6000), gain=0.075 * lvl)
            mid_r.contact(
                t + 0.0012, 0.0025, band=(1100, 3000), gain=0.21 * lvl,
                modes=((f0, dec, 3.0, f1),),
            )
            mid_r.granular(
                t + 0.004, 0.040, density=95, band=(2200, 6600),
                grain_duration=(0.002, 0.006), pressure=((0, 1), (0.4, 0.3), (1, 0)),
                gain=0.028 * lvl,
            )
        else:
            mid_r.contact(t, 0.004, band=(1500, 5200), gain=0.18 * lvl, modes=((f0, 0.022, 0.42),))

    # Far left rapid ledge cluster
    far_specs = [
        (1.62, 1320.0, 2360.0, 0.028, 0.68),
        (1.77, 1510.0, 2650.0, 0.026, 0.56),
        (1.98, 1240.0, 2190.0, 0.030, 0.52),
    ]
    for t, f0, f1, dec, lvl in far_specs:
        if with_bubbles:
            far_l.contact(
                t, 0.0025, band=(1400, 4200), gain=0.16 * lvl,
                modes=((f0, dec, 2.8, f1),),
            )
        else:
            far_l.contact(t, 0.0035, band=(1400, 5000), gain=0.15 * lvl, modes=((f0, 0.020, 0.38),))

    # Subtle distant subterranean water trickle bed
    if with_bubbles:
        trickle.friction(
            0.05, 4.25, band=(550, 2600), end_band=(720, 3100),
            pressure=((0, 0), (0.12, 0.65), (0.55, 0.85), (0.85, 0.55), (1, 0)),
            roughness=0.60, gain=0.028,
        )
        trickle.granular(
            0.10, 4.10, density=24, band=(1400, 5200),
            grain_duration=(0.002, 0.0075),
            pressure=((0, 0), (0.15, 0.7), (0.6, 1.0), (0.9, 0.5), (1, 0)),
            gain=0.035,
        )

    return {"near_l": near_l, "mid_r": mid_r, "far_l": far_l, "trickle": trickle}


def main() -> None:
    out = ROOT / "output/cavern_water_drops"
    canvas = SpectralCanvas(duration=4.6, sr=44100, seed=107)

    # Pass 1: Timing and rhythm skeleton (dry drop contacts)
    layers_p1 = build_drops(canvas, with_bubbles=False)
    dry_mono = (
        layers_p1["near_l"].audio
        + layers_p1["mid_r"].audio
        + layers_p1["far_l"].audio
    )
    print("Pass 1:", json.dumps(canvas.render_mix(
        dry_mono, out / "pass_01.wav", out / "pass_01.png",
        "Pass 1: Drop Timing Skeleton",
    )))

    # Pass 2: Liquid Minnaert bubble modes, splash micro-grains & distant trickle
    canvas.reset()
    layers_p2 = build_drops(canvas, with_bubbles=True)
    mono_p2 = (
        layers_p2["near_l"].audio
        + layers_p2["mid_r"].audio
        + layers_p2["far_l"].audio
        + layers_p2["trickle"].audio
    )
    print("Pass 2:", json.dumps(canvas.render_mix(
        mono_p2, out / "pass_02.wav", out / "pass_02.png",
        "Pass 2: Liquid Bubble Modes & Trickle Bed",
    )))

    # Pass 3: Stereo spatial placement, distance air absorption & dampened stone cavern reverb
    canvas.reset()
    layers_p3 = build_drops(canvas, with_bubbles=True)
    scene = canvas.new_scene()
    scene.add(layers_p3["near_l"], pan=-0.38, distance=0.06, gain=1.0)
    scene.add(layers_p3["mid_r"], pan=0.42, distance=0.28, gain=0.90)
    scene.add(layers_p3["far_l"], pan=-0.65, distance=0.56, gain=0.75)
    scene.add(layers_p3["trickle"], pan=0.55, distance=0.68, gain=0.70)
    scene.apply_room(size=0.84, damping=0.46, wet=0.46, predelay=0.018, early_reflections=9)
    scene.soft_limit(ceiling_db=-1.0, drive=2.2)
    scene.fade_edges(fade_in=0.04, fade_out=0.25)

    print("Pass 3:", json.dumps(canvas.render_mix(
        scene.audio, out / "cavern_water_drops.wav", out / "cavern_water_drops_spec.png",
        "Cavern Water Drops (Stereo)", subtype="PCM_24",
    )))

    if shutil.which("ffmpeg"):
        canvas.export_progression_video(out / "cavern_water_drops_progression.mp4")


if __name__ == "__main__":
    main()
