"""Leak audit of INCLUDE's own (shipped) train/test split, using the same two
checks as audit_results.py: take-groups that put clips on both sides of the
train/test boundary, and capture-counter session blocks on both sides.

    python report/final/scripts/official_split_leak.py
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPLITS = ROOT / "splits"
OUT = ROOT / "report/final/data/official_split_leak.json"


def mvi(path: str) -> int:
    m = re.search(r"MVI_(\d+)", path)
    return int(m.group(1)) if m else -1


def rows_of(name: str) -> list[dict]:
    with (SPLITS / name).open(encoding="utf-8") as h:
        return list(csv.DictReader(h))


def main() -> None:
    # Session blocks rebuilt from the whole corpus exactly as audit_results.py does.
    ids = sorted({mvi(r["video_path"]) for r in rows_of("full263__session-disjoint-foldval.csv")} - {-1})
    edges, start, prev = [], ids[0], ids[0]
    for cur in ids[1:]:
        if cur - prev > 5:
            edges.append((start, prev))
            start = cur
        prev = cur
    edges.append((start, prev))

    def block_of(i: int) -> int:
        return next(k for k, (lo, hi) in enumerate(edges) if lo <= i <= hi)

    res = {"n_blocks_corpus": len(edges)}
    for name in ("include50__official.csv", "full263__official.csv"):
        rows = rows_of(name)
        groups, blocks = defaultdict(set), defaultdict(set)
        for r in rows:
            if r["split"] not in ("train", "test"):
                continue
            groups[(r["label"], r["take_group"])].add(r["split"])
            if mvi(r["video_path"]) >= 0:
                blocks[block_of(mvi(r["video_path"]))].add(r["split"])
        straddling = {g for g, s in groups.items() if len(s) > 1}
        test = [r for r in rows if r["split"] == "test"]
        test_in_straddling = sum(1 for r in test if (r["label"], r["take_group"]) in straddling)
        res[name] = {
            "counts": dict(Counter(r["split"] for r in rows)),
            "words": len({r["label"] for r in rows}),
            "take_groups_straddling_train_test": len(straddling),
            "test_clips_with_a_train_take_in_same_group": test_in_straddling,
            "test_clips": len(test),
            "blocks_with_rows": len(blocks),
            "blocks_on_both_sides": sum(1 for s in blocks.values() if len(s) > 1),
        }
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
