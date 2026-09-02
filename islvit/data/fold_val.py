"""Fold the validation clips into train for a final model.

The val set in ``session-disjoint`` shares every recording session with train --
verified directly: train and val both draw on blocks {1,2,3,4,7,8,9,10,11,12}
while test holds {0,5,6} alone. Val therefore measures *in-session* recognition,
which is not what the reported number measures, so selecting a checkpoint on it
optimises the wrong quantity. Two runs showed this plainly: self-supervised
pretraining moved val by -0.8 points and test by +7.6.

Given val cannot steer selection usefully, its 888 clips are worth far more as
training data -- a 45% increase on 1,957. The checkpoint is then taken as the
final EMA weights under a cosine schedule that ends at zero learning rate, rather
than by early stopping.

**Test is untouched**, so every number stays comparable with earlier runs. Only
the train/val boundary moves, and it moves entirely within the training sessions.

Usage::

    python -m islvit.data.fold_val --split splits/full263__session-disjoint.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge val into train for a final model")
    parser.add_argument("--split", type=str, default="splits/full263__session-disjoint.csv")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    split_path = Path(args.split)
    with split_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fieldnames = list(rows[0].keys())

    before = Counter(row["split"] for row in rows)
    for row in rows:
        if row["split"] == "val":
            row["split"] = "train"
    after = Counter(row["split"] for row in rows)

    out_path = Path(args.out) if args.out else split_path.with_name(
        split_path.stem.replace("session-disjoint", "session-disjoint-foldval") + split_path.suffix
    )
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # The entire justification for this file is that only the train/val boundary
    # moved. If test changed at all, the new numbers are not comparable to the old
    # ones and the whole exercise is void.
    assert before["test"] == after["test"], "test set changed; runs are no longer comparable"
    assert after["val"] == 0, f"{after['val']} rows still marked val"

    print(f"train {before['train']} -> {after['train']} (+{after['train'] / before['train'] - 1:.0%})")
    print(f"val   {before['val']} -> 0   (folded in)")
    print(f"test  {after['test']} (unchanged)")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
