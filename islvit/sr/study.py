"""Helpers for run_gan_study.sh: path lists, guarded clean-up, detection metrics.

Subcommands::

    python -m islvit.sr.study train-paths OUT.csv     # the 2,845 training clips, leak-checked
    python -m islvit.sr.study cislr-paths OUT.csv     # the 609 CISLR cross-corpus test clips
    python -m islvit.sr.study detection CACHE LM_CACHE OUT.json [--reference CACHE LM_CACHE]
    python -m islvit.sr.study drop CACHE              # delete crops.npy from a study cache

``drop`` exists because this project has lost 54 GB twice to clean-up steps that
did not check what they were deleting. It removes only ``crops.npy``, only inside
the study work root (GAN_WORK, default C:/isl_gan_work), and only from a
directory carrying a FRONTEND.json, i.e. one the study's own extraction wrote.
Everything small enough to keep for later analysis -- sources, geometry, index,
landmarks -- stays.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from islvit.sr.patches import split_paths

DIRECT, ROI, INTERP, MISSING = 1, 2, 3, 0
STREAMS = ("left_hand", "right_hand", "face")
# Frames are letterboxed to 1280x720, so normalised x units are 16/9 of y units.
ASPECT = 1280 / 720


def write_paths(paths, out: str) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with Path(out).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_path"])
        writer.writerows([p] for p in paths)
    print(f"wrote {len(paths)} paths to {out}")


def cislr_paths() -> list[str]:
    with Path("splits/crosscorpus__cislr-test.csv").open(encoding="utf-8") as handle:
        return sorted(r["video_path"] for r in csv.DictReader(handle) if r.get("corpus") == "cislr" and r["split"] == "test")


def drop(cache: str) -> None:
    work = Path(os.environ.get("GAN_WORK", "C:/isl_gan_work")).resolve()
    target = Path(cache).resolve()
    if work not in target.parents:
        raise SystemExit(f"refusing: {target} is not inside the study work root {work}")
    if not (target / "FRONTEND.json").exists():
        raise SystemExit(f"refusing: {target} has no FRONTEND.json, so the study did not build it")
    crops = target / "crops.npy"
    if crops.exists():
        size = crops.stat().st_size / 2**30
        crops.unlink()
        print(f"dropped {crops} ({size:.1f} GiB); sources, index and landmarks kept")


def video_order(cache: Path) -> list[str]:
    with (cache / "index.csv").open(encoding="utf-8") as handle:
        return [r["video_path"] for r in csv.DictReader(handle)]


def detection(cache: str, lm_cache: str, out: str, reference: tuple[str, str] | None) -> None:
    """Detection and landmark statistics for one front end, optionally against a reference."""
    cache, lm_cache = Path(cache), Path(lm_cache)
    sources = np.load(cache / "sources.npy")
    present = np.load(lm_cache / "hand_src.npy") > 0
    hands = np.load(lm_cache / "hands.npy").astype(np.float32)
    result = {"cache": cache.as_posix(), "clips": int(len(sources)), "frontend": json.loads((cache / "FRONTEND.json").read_text())
              if (cache / "FRONTEND.json").exists() else None, "streams": {}}
    for s, name in enumerate(STREAMS):
        column = sources[:, :, s]
        result["streams"][name] = {
            "direct": float((column == DIRECT).mean()), "roi": float((column == ROI).mean()),
            "interp": float((column == INTERP).mean()), "missing": float((column == MISSING).mean()),
            "genuine": float(np.isin(column, (DIRECT, ROI)).mean()),
        }
    result["hand_landmarks_present"] = {"left": float(present[:, :, 0].mean()), "right": float(present[:, :, 1].mean())}

    if reference:
        ref_cache, ref_lm = Path(reference[0]), Path(reference[1])
        if video_order(ref_cache) != video_order(cache):
            raise SystemExit(f"{cache} and {ref_cache} hold different clips or a different order")
        ref_present = np.load(ref_lm / "hand_src.npy") > 0
        ref_hands = np.load(ref_lm / "hands.npy").astype(np.float32)
        both = present & ref_present
        delta = hands[..., 0, :2] - ref_hands[..., 0, :2]          # wrist, (N, T, 2)
        error = np.hypot(delta[..., 0] * ASPECT, delta[..., 1])     # isotropic, in frame heights
        result["against_clean"] = {
            "reference": ref_lm.as_posix(),
            "presence_agreement": float((present == ref_present).mean()),
            # Found here but not in the clean video: the hallucination a GAN can cause.
            "extra_hands": float((present & ~ref_present).mean()),
            "missed_hands": float((~present & ref_present).mean()),
            "wrist_error_median": float(np.median(error[both])) if both.any() else None,
            "wrist_error_p90": float(np.percentile(error[both], 90)) if both.any() else None,
            # Present in both but more than 5 % of the frame height apart: a hand in the wrong place.
            "misplaced_hands": float(((error > 0.05) & both).mean()),
        }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GAN study helpers")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train-paths").add_argument("out")
    sub.add_parser("cislr-paths").add_argument("out")
    d = sub.add_parser("detection")
    d.add_argument("cache")
    d.add_argument("lm_cache")
    d.add_argument("out")
    d.add_argument("--reference", nargs=2, default=None, metavar=("CACHE", "LM_CACHE"))
    sub.add_parser("drop").add_argument("cache")
    args = parser.parse_args()

    if args.command == "train-paths":
        write_paths(split_paths("train"), args.out)
    elif args.command == "cislr-paths":
        write_paths(cislr_paths(), args.out)
    elif args.command == "detection":
        detection(args.cache, args.lm_cache, args.out, tuple(args.reference) if args.reference else None)
    elif args.command == "drop":
        drop(args.cache)


if __name__ == "__main__":
    main()
