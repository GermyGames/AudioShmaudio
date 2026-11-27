import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/audio-shmaudio/scripts"))
from spectral_foley import SpectralCanvas


def main() -> None:
    out = ROOT / "output/gravel_footstep"
    canvas = SpectralCanvas(duration=0.5, sr=44100, seed=23)

    # A short heel strike with an uneven roll into the forefoot.
    canvas.foley.contact(.054, .031, band=(250, 2250), gain=.21)
    canvas.foley.contact(.146, .024, band=(340, 2900), gain=.11)
    canvas.foley.friction(
        .056, .224, band=(330, 1750), end_band=(580, 2800), roughness=.77,
        pressure=((0, 0), (.08, .71), (.30, .25), (.47, 1), (.72, .56), (1, 0)),
        gain=.15,
    )
    print(json.dumps(canvas.render(out / "pass_01.wav", out / "pass_01.png", "Contact and roll")))

    # Variable-size grains cluster under pressure and leave space between peaks.
    canvas.foley.granular(
        .052, .235, density=475, band=(440, 3200), grain_duration=(.004, .028),
        pressure=((0, .26), (.12, .94), (.33, .32), (.52, 1), (.71, .68), (1, 0)),
        gain=.22,
    )
    canvas.foley.granular(
        .063, .211, density=275, band=(2450, 7600), grain_duration=(.0017, .010),
        pressure=((0, .10), (.18, .72), (.38, .24), (.58, .93), (.81, .37), (1, 0)),
        gain=.11,
    )
    print(json.dumps(canvas.render(out / "pass_02.wav", out / "pass_02.png", "Gravel under pressure")))

    # A few grains continue moving as the sole leaves the ground.
    canvas.foley.granular(
        .247, .123, density=150, band=(950, 5500), grain_duration=(.002, .017),
        pressure=((0, .24), (.16, .63), (.48, .35), (1, 0)), gain=.064,
    )
    print(json.dumps(canvas.render(
        out / "gravel_footstep.wav", out / "gravel_footstep_spec.png", "Gravel footstep",
        subtype="PCM_24",
    )))

    if shutil.which("ffmpeg"):
        canvas.export_progression_video(out / "gravel_progression.mp4")


if __name__ == "__main__":
    main()
