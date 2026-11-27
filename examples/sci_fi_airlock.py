import json
import shutil
import sys
import tempfile
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/audio-shmaudio/scripts"))
from spectral_foley import SpectralCanvas

RELAY_MODES = (
    (1920.0, 0.018, 0.70),
    (3450.0, 0.013, 1.00),
    (5400.0, 0.009, 0.52),
)

STEEL_MODES = (
    (192.0, 0.105, 0.75),
    (348.0, 0.082, 0.90),
    (585.0, 0.065, 1.00),
    (920.0, 0.048, 0.85),
    (1410.0, 0.036, 0.68),
    (2080.0, 0.028, 0.50),
    (2940.0, 0.020, 0.34),
    (4120.0, 0.015, 0.20),
)

DUCT_MODES = (
    (138.0, 0.045, 0.80),
    (295.0, 0.035, 1.00),
    (560.0, 0.024, 0.70),
    (1020.0, 0.018, 0.40),
)


def paint_chamber_wash(canvas: SpectralCanvas) -> None:
    """Paint a soft downward-sweeping chamber turbulence bed."""
    canvas.paint_cloud(0.80, 2.95, 180.0, 4600.0, gain=0.45, roughness=0.55)
    canvas.erase_region(1.90, 3.05, 1800.0, 12000.0, strength=0.80)


def build_airlock_layers(canvas: SpectralCanvas, *, full_detail: bool) -> dict:
    relay = canvas.new_layer()
    bolts_l = canvas.new_layer()
    bolts_r = canvas.new_layer()
    hiss_hi = canvas.new_layer()
    duct_roar = canvas.new_layer()
    mechanism = canvas.new_layer()
    seal_sub = canvas.new_layer()
    seal_body = canvas.new_layer()
    clamp_l = canvas.new_layer()
    clamp_r = canvas.new_layer()

    # 1. Electromechanical relay snap & solenoid pilot valve hiss (no arcade beeps)
    relay.contact(0.080, 0.0022, band=(2600, 11000), gain=0.20)
    relay.contact(0.104, 0.0020, band=(2200, 9600), gain=0.14)
    if full_detail:
        relay.friction(
            0.115, 0.15, band=(2800, 11500), end_band=(1500, 6200),
            roughness=0.35, pressure=((0, 0), (0.08, 1.0), (0.55, 0.45), (1, 0)),
            gain=0.055,
        )
        relay.resonate(RELAY_MODES, dry=0.45, gain=1.0)

    # 2. Heavy steel locking bolts retracting left & right
    if full_detail:
        bolts_l.friction(
            0.36, 0.16, band=(420, 5200), end_band=(680, 6800),
            roughness=0.68, pressure=((0, 0), (0.2, 0.8), (0.8, 1.0), (1, 0)),
            gain=0.095,
        )
        bolts_r.friction(
            0.48, 0.16, band=(460, 5400), end_band=(720, 7100),
            roughness=0.68, pressure=((0, 0), (0.2, 0.8), (0.8, 1.0), (1, 0)),
            gain=0.090,
        )
    bolts_l.contact(0.495, 0.030, band=(42, 220), gain=0.28)
    bolts_l.contact(0.495, 0.008, band=(950, 8800), gain=0.24)
    bolts_r.contact(0.615, 0.028, band=(45, 240), gain=0.26)
    bolts_r.contact(0.615, 0.008, band=(1050, 9200), gain=0.22)
    if full_detail:
        bolts_l.resonate(STEEL_MODES, dry=0.35, gain=1.05)
        bolts_r.resonate(STEEL_MODES, dry=0.35, gain=1.05)

    # 3. Gasket break pop, multi-layered turbulent air depressurization & geared bulkhead slide
    if full_detail:
        hiss_hi.contact(0.780, 0.014, band=(450, 6800), gain=0.18)
        hiss_hi.contact(0.780, 0.026, band=(55, 240), gain=0.20)
        hiss_hi.friction(
            0.790, 2.15, band=(2600, 12800), end_band=(380, 1850), roughness=0.28,
            pressure=((0, 0), (0.04, 1.0), (0.28, 0.82), (0.65, 0.45), (0.92, 0.12), (1, 0)),
            gain=0.22,
        )
        hiss_hi.granular(
            0.800, 0.65, density=170, band=(2200, 9200), grain_duration=(0.002, 0.007),
            pressure=((0, 0), (0.10, 1.0), (0.55, 0.45), (1, 0)), gain=0.055,
        )

        duct_roar.friction(
            0.800, 2.20, band=(65, 580), end_band=(45, 230), roughness=0.45,
            pressure=((0, 0), (0.06, 0.85), (0.35, 1.0), (0.75, 0.55), (1, 0)),
            gain=0.20,
        )
        duct_roar.resonate(DUCT_MODES, dry=0.68, gain=1.0)

        mechanism.friction(
            1.050, 1.88, band=(220, 2800), end_band=(165, 1950), roughness=0.72,
            pressure=((0, 0), (0.12, 0.78), (0.55, 1.0), (0.88, 0.65), (1, 0)),
            gain=0.12,
        )
        mechanism.granular(
            1.060, 1.85, density=105, band=(280, 3600), grain_duration=(0.0025, 0.012),
            pressure=((0, 0), (0.14, 0.75), (0.60, 1.0), (0.90, 0.55), (1, 0)),
            gain=0.085,
        )
        mechanism.stick_slip(
            1.080, 1.80,
            rate=((0.0, 24.0), (0.35, 58.0), (0.75, 52.0), (1.0, 20.0)),
            pressure=((0.0, 0.0), (0.12, 0.80), (0.55, 1.0), (0.88, 0.60), (1.0, 0.0)),
            modes=STEEL_MODES, roughness=0.45, grit=0.35, gain=0.13,
        )
        mechanism.resonate(STEEL_MODES, dry=0.45, gain=1.0)
    else:
        hiss_hi.friction(
            0.790, 2.12, band=(500, 4800), roughness=0.35,
            pressure=((0, 0), (0.06, 1.0), (0.55, 0.55), (1, 0)), gain=0.16,
        )

    # 4. Heavy bulkhead compression seal & staggered pneumatic lock clamps
    if full_detail:
        seal_sub.friction(
            2.918, 0.095, band=(32, 165), roughness=0.18,
            pressure=((0, 0), (0.05, 1.0), (0.35, 0.45), (1, 0)), gain=0.40,
        )
        seal_sub.contact(2.920, 0.042, band=(38, 195), gain=0.34)
        seal_sub.contact(2.942, 0.032, band=(44, 220), gain=0.24)

        seal_body.contact(2.920, 0.016, band=(140, 2400), gain=0.32)
        seal_body.contact(2.942, 0.012, band=(220, 3800), gain=0.26)
        seal_body.resonate(STEEL_MODES, dry=0.40, gain=1.0)

        clamp_l.contact(3.18, 0.005, band=(1600, 9200), gain=0.19)
        clamp_l.resonate(STEEL_MODES, dry=0.40, gain=1.0)
        clamp_r.contact(3.27, 0.005, band=(1800, 9800), gain=0.18)
        clamp_r.friction(
            3.28, 0.16, band=(2200, 9500), end_band=(900, 3600), roughness=0.26,
            pressure=((0, 0), (0.08, 1.0), (0.55, 0.35), (1, 0)), gain=0.065,
        )
        clamp_r.resonate(STEEL_MODES, dry=0.45, gain=1.0)
    else:
        seal_body.contact(2.92, 0.022, band=(100, 3200), gain=0.34)

    return {
        "relay": relay,
        "bolts_l": bolts_l,
        "bolts_r": bolts_r,
        "hiss_hi": hiss_hi,
        "duct_roar": duct_roar,
        "mechanism": mechanism,
        "seal_sub": seal_sub,
        "seal_body": seal_body,
        "clamp_l": clamp_l,
        "clamp_r": clamp_r,
    }


def main() -> None:
    out = ROOT / "output/sci_fi_airlock"
    canvas = SpectralCanvas(duration=4.6, sr=44100, seed=512)

    # Pass 1: Mechanical timing skeleton (relay, bolts, air release, seal)
    p1 = build_airlock_layers(canvas, full_detail=False)
    print("Pass 1:", json.dumps(canvas.render_mix(
        p1["relay"].audio + p1["bolts_l"].audio + p1["bolts_r"].audio
        + p1["hiss_hi"].audio + p1["seal_body"].audio,
        out / "pass_01.wav", out / "pass_01.png",
        "Pass 1: Airlock Mechanical Skeleton",
    )))

    # Pass 2: Resonant steel bolts, sweeping turbulent vent, geared track & spectral wash
    canvas.reset()
    p2 = build_airlock_layers(canvas, full_detail=True)
    canvas.foley.add_waveform(
        0.0,
        p2["relay"].audio + p2["bolts_l"].audio + p2["bolts_r"].audio
        + p2["hiss_hi"].audio + p2["duct_roar"].audio + p2["mechanism"].audio
        + p2["seal_sub"].audio + p2["seal_body"].audio
        + p2["clamp_l"].audio + p2["clamp_r"].audio,
    )
    paint_chamber_wash(canvas)
    print("Pass 2:", json.dumps(canvas.render(
        out / "pass_02.wav", out / "pass_02.png",
        "Pass 2: Steel Resonance, Duct Roar & Gear Track",
        spectral_gain=0.045,
    )))

    # Pass 3: Stereo hydraulic bulkhead travel, corridor reflections & mastered export
    canvas.reset()
    p3 = build_airlock_layers(canvas, full_detail=True)

    wash_canvas = SpectralCanvas(duration=canvas.duration, sr=canvas.sr, seed=canvas.seed)
    paint_chamber_wash(wash_canvas)
    with tempfile.TemporaryDirectory() as tmpdir:
        wash_wav = Path(tmpdir) / "wash.wav"
        wash_canvas.render(wash_wav, target_peak_db=None, spectral_gain=0.045)
        wash_audio, _ = sf.read(wash_wav, dtype="float32")

    scene = canvas.new_scene()
    scene.add(p3["relay"], pan=-0.18, distance=0.04, gain=1.0)
    scene.add(p3["bolts_l"], pan=-0.52, distance=0.05, gain=1.05)
    scene.add(p3["bolts_r"], pan=0.52, distance=0.06, gain=1.05)
    scene.add(p3["hiss_hi"], pan=0.0, distance=0.05, gain=1.0)
    scene.add(p3["duct_roar"], pan=0.0, distance=0.10, gain=1.0)
    scene.add(wash_audio, pan=0.0, distance=0.14, gain=1.0)
    scene.add(p3["mechanism"], pan=((0.0, -0.50), (0.55, 0.0), (1.0, 0.50)), distance=0.06, gain=1.08)
    scene.add(p3["seal_sub"], pan=0.0, distance=0.03, gain=1.08)
    scene.add(p3["seal_body"], pan=0.0, distance=0.05, gain=1.0)
    scene.add(p3["clamp_l"], pan=-0.42, distance=0.05, gain=0.95)
    scene.add(p3["clamp_r"], pan=0.42, distance=0.05, gain=0.95)
    scene.apply_room(size=0.38, damping=0.34, wet=0.22, predelay=0.008, early_reflections=8)
    scene.soft_limit(ceiling_db=-1.0, drive=1.55)
    scene.fade_edges(fade_in=0.02, fade_out=0.20)

    print("Pass 3:", json.dumps(canvas.render_mix(
        scene.audio, out / "sci_fi_airlock.wav", out / "sci_fi_airlock_spec.png",
        "Sci-Fi Airlock Sequence (Stereo)", subtype="PCM_24",
    )))

    if shutil.which("ffmpeg"):
        canvas.export_progression_video(out / "sci_fi_airlock_progression.mp4")


if __name__ == "__main__":
    main()
