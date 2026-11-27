import json
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../skills/audio-shmaudio/scripts")))
from spectral_foley import SpectralCanvas


def main() -> None:
    out_dir = os.path.join(os.path.dirname(__file__), "../output/sword_unsheath")
    os.makedirs(out_dir, exist_ok=True)

    canvas = SpectralCanvas(duration=0.95, sr=22050)

    # Pass 1: Rising metallic friction scrape
    canvas.add_formant_noise(
        t_start=0.05,
        t_end=0.62,
        formants=[(1200, 3200, 220), (3400, 6800, 350)],
        gain=0.75,
        seed=7,
    )
    print("Pass 1:", json.dumps(canvas.render(f"{out_dir}/pass_01.wav", f"{out_dir}/pass_01.png", pass_label="Pass 1: Blade Scrape")))

    # Pass 2: Free metallic ring tail as blade clears scabbard
    canvas.add_modal_strike(t_start=0.48, base_hz=920.0, material="metal", decay=0.22, gain=0.9)
    print("Pass 2:", json.dumps(canvas.render(f"{out_dir}/pass_02.wav", f"{out_dir}/pass_02.png", pass_label="Pass 2: Metal Ring")))

    # Pass 3: Guard release click + high-end polish
    canvas.add_impulse(time=0.035, sharpness=0.92, gain=0.28, seed=9)
    print("Pass 3:", json.dumps(canvas.render(f"{out_dir}/sword_unsheath.wav", f"{out_dir}/sword_unsheath_spec.png", pass_label="Pass 3: Guard Click")))

    if shutil.which("ffmpeg"):
        canvas.export_progression_video(f"{out_dir}/sword_unsheath_progression.mp4", pad_silence=0.35)


if __name__ == "__main__":
    main()
