import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/audio-shmaudio/scripts"))
from spectral_foley import SpectralCanvas


def main() -> None:
    out = ROOT / "output/wood_knock"
    canvas = SpectralCanvas(duration=.75, sr=44100, seed=7)

    for start, weight in ((.038, 1.0), (.295, .86)):
        canvas.foley.contact(start, .011, band=(430, 3700), gain=.25 * weight)
    print(json.dumps(canvas.render(out / "pass_01.wav", out / "pass_01.png", "Knuckle contacts")))

    modes = ((161, .131, .45), (373, .093, .34), (714, .071, .19), (1170, .041, .085))
    for start, weight in ((.039, 1.0), (.296, .86)):
        canvas.foley.contact(start, .018, band=(240, 1250), gain=.23 * weight, modes=modes)
    print(json.dumps(canvas.render(out / "pass_02.wav", out / "pass_02.png", "Wood panel response")))

    canvas.foley.contact(.31, .012, band=(360, 1900), gain=.037)
    print(json.dumps(canvas.render(
        out / "wood_knock.wav", out / "wood_knock_spec.png", "Double knock",
        subtype="PCM_24",
    )))

    if shutil.which("ffmpeg"):
        canvas.export_progression_video(out / "wood_knock_progression.mp4")


if __name__ == "__main__":
    main()
