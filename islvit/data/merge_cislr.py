"""Fold CISLR's labelled clips into an INCLUDE split as extra training data.

218 of INCLUDE's 262 classes also appear in CISLR, worth 609 clips. Against a
session-disjoint training set of 1,957 clips that is a 31% increase, but the size
is not the point -- the *provenance* is. INCLUDE is one studio in one city, and
the model's measured failure is generalising across recording sessions. CISLR
clips come from different signers, rooms, clothing and lighting, which is the
exact axis the model cannot currently cross.

They are also safe by construction: a different corpus cannot share a recording
session with INCLUDE's held-out sessions, so adding them to train cannot leak into
test. They go to ``train`` only -- never val or test -- so every reported number
stays measured on INCLUDE alone and remains comparable to every earlier run.

Usage::

    python -m islvit.data.merge_cislr --split splits/full263__session-disjoint.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from islvit.data.crops import SOURCE_DIRECT, SOURCE_ROI

CISLR_INDEX = Path("CISLR/dataset.csv")
CISLR_CACHE = Path("cache_cislr")


def normalise(gloss: str) -> str:
    """INCLUDE labels look like '94. good'; CISLR glosses are bare words."""
    text = str(gloss).lower().strip()
    text = re.sub(r"^[0-9]+\.\s*", "", text)
    text = re.sub(r"[^a-z ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def cislr_rows(min_genuine: float) -> list[tuple[str, str, float]]:
    """(video_path, normalised gloss, genuine-detection rate) for cached clips."""
    with (CISLR_CACHE / "index.csv").open(encoding="utf-8") as handle:
        index = {row["video_path"]: int(row["row"]) for row in csv.DictReader(handle) if row["cached"] == "1"}

    sources = np.load(CISLR_CACHE / "sources.npy", mmap_mode="r")
    with CISLR_INDEX.open(encoding="utf-8") as handle:
        catalogue = list(csv.DictReader(handle))

    rows: list[tuple[str, str, float]] = []
    for entry in catalogue:
        video = f"{entry['uid']}.mp4"
        if video not in index:
            continue
        hands = np.asarray(sources[index[video]][:, :2])
        genuine = float(((hands == SOURCE_DIRECT) | (hands == SOURCE_ROI)).mean())
        if genuine >= min_genuine:
            rows.append((video, normalise(entry["gloss"]), genuine))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge CISLR clips into an INCLUDE split")
    parser.add_argument("--split", type=str, default="splits/full263__session-disjoint.csv")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--min-genuine",
        type=float,
        default=0.25,
        help="drop CISLR clips whose hands are genuinely detected in fewer than this "
        "fraction of frames; at 300x300 many boxes are stale interpolations",
    )
    args = parser.parse_args()

    split_path = Path(args.split)
    with split_path.open(encoding="utf-8") as handle:
        include = list(csv.DictReader(handle))
    fieldnames = list(include[0].keys())

    label_of = {normalise(row["label"]): row["label"] for row in include}
    candidates = cislr_rows(args.min_genuine)
    matched = [(video, label_of[gloss], rate) for video, gloss, rate in candidates if gloss in label_of]

    out_path = Path(args.out) if args.out else split_path.with_name(
        split_path.stem + "+cislr" + split_path.suffix
    )
    out_fields = fieldnames + ["corpus"]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields)
        writer.writeheader()
        for row in include:
            writer.writerow({**row, "corpus": "include"})
        for video, label, _ in matched:
            writer.writerow(
                {
                    **{name: "" for name in fieldnames},
                    "video_path": video,
                    "label": label,
                    # Its own take-group namespace: a CISLR clip shares no capture
                    # session with anything in INCLUDE, so it must never be grouped
                    # with an INCLUDE take-run.
                    "take_group": f"cislr#{video}",
                    "split": "train",
                    "corpus": "cislr",
                }
            )

    covered = len({label for _, label, _ in matched})
    train_before = sum(1 for row in include if row["split"] == "train")
    print(f"CISLR clips cached and above {args.min_genuine:.0%} genuine: {len(candidates)}")
    print(f"  matched to INCLUDE classes: {len(matched)} clips over {covered} classes")
    print(f"  train {train_before} -> {train_before + len(matched)} (+{len(matched) / train_before:.0%})")
    print(f"  wrote {out_path}")

    # The whole value of this merge is that the added clips are NOT from INCLUDE's
    # sessions, so assert the only way that could break: a video path collision.
    include_paths = {row["video_path"] for row in include}
    collisions = include_paths & {video for video, _, _ in matched}
    assert not collisions, f"CISLR path collides with INCLUDE: {sorted(collisions)[:5]}"
    held_out = {row["video_path"] for row in include if row["split"] in ("val", "test")}
    assert not (held_out & {video for video, _, _ in matched}), "CISLR clip landed in val/test"
    print("  leakage assertions passed (no shared paths with val/test)")


if __name__ == "__main__":
    main()
