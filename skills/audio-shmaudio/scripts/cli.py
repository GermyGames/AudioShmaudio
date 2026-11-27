import argparse
import glob
import json
import os

import librosa

from spectral_foley import (
    compute_diagnostics,
    save_feedback_spectrogram,
    stitch_progression_video,
)


def cmd_analyze(args: argparse.Namespace) -> None:
    audio, sr = librosa.load(args.wav, sr=args.sr, mono=False)
    audio = audio.T
    out_png = args.out or os.path.splitext(args.wav)[0] + "_spec.png"
    diag = compute_diagnostics(audio, sr, args.wav, out_png)
    save_feedback_spectrogram(
        audio=audio,
        sr=sr,
        spec_path=out_png,
        diagnostics=diag,
        pass_label=args.label or "Reference",
    )
    print(json.dumps(diag, indent=2))


def cmd_compare(args: argparse.Namespace) -> None:
    audio_a, sr_a = librosa.load(args.wav_a, sr=args.sr, mono=False)
    audio_b, sr_b = librosa.load(args.wav_b, sr=args.sr, mono=False)
    diag_a = compute_diagnostics(audio_a.T, sr_a, args.wav_a)
    diag_b = compute_diagnostics(audio_b.T, sr_b, args.wav_b)
    deltas = {
        "active_rms_db": round(diag_b["active_rms_db"] - diag_a["active_rms_db"], 2),
        "weighted_centroid_hz": round(
            diag_b["weighted_centroid_hz"] - diag_a["weighted_centroid_hz"], 1
        ),
        "crest_factor_db": round(diag_b["crest_factor_db"] - diag_a["crest_factor_db"], 2),
        "tail_rms_db": round(diag_b["tail_rms_db"] - diag_a["tail_rms_db"], 2),
        "energy_below_180_hz_pct": round(
            diag_b["energy_below_180_hz_pct"] - diag_a["energy_below_180_hz_pct"], 2
        ),
        "band_energy_pct": {
            band: round(
                diag_b["band_energy_pct"][band] - diag_a["band_energy_pct"][band], 2
            )
            for band in diag_a["band_energy_pct"]
        },
    }
    print(json.dumps({"a": diag_a, "b": diag_b, "deltas": deltas}, indent=2))


def cmd_stitch(args: argparse.Namespace) -> None:
    wavs = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    pairs = []
    for w in wavs:
        p = os.path.splitext(w)[0] + ".png"
        if os.path.exists(p):
            pairs.append((w, p))
    if not pairs:
        raise SystemExit(
            f"No matching {args.pattern} / .png pairs found in {args.dir}"
        )
    out = stitch_progression_video(pairs, args.out, pad_silence=args.pad)
    print(f"Exported {out} ({len(pairs)} passes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="AudioShmaudio CLI utility.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser(
        "analyze", help="Render a feedback spectrogram + diagnostics for an existing WAV."
    )
    p_analyze.add_argument("wav", help="Path to input audio file.")
    p_analyze.add_argument("--out", help="Output PNG path.")
    p_analyze.add_argument(
        "--sr", type=int, default=None, help="Target sample rate (defaults to native)."
    )
    p_analyze.add_argument("--label", help="Optional title label.")
    p_analyze.set_defaults(func=cmd_analyze)

    p_compare = sub.add_parser(
        "compare", help="Compare diagnostics and spectral balance between two WAV files."
    )
    p_compare.add_argument("wav_a", help="Baseline WAV file.")
    p_compare.add_argument("wav_b", help="Revised WAV file.")
    p_compare.add_argument(
        "--sr", type=int, default=None, help="Target sample rate (defaults to native)."
    )
    p_compare.set_defaults(func=cmd_compare)

    p_stitch = sub.add_parser(
        "stitch", help="Stitch matching WAV/PNG pass pairs in a folder into an MP4."
    )
    p_stitch.add_argument("dir", help="Directory containing pass WAV and PNG pairs.")
    p_stitch.add_argument(
        "--pattern", default="pass_*.wav", help="Glob pattern for WAV files."
    )
    p_stitch.add_argument("--out", default="progression.mp4", help="Output MP4 path.")
    p_stitch.add_argument(
        "--pad", type=float, default=0.35, help="Silence padding between passes (sec)."
    )
    p_stitch.set_defaults(func=cmd_stitch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
