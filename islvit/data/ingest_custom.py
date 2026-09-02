"""Build a **signer-disjoint** split from newly recorded clips.

This is the protocol the report has been unable to run. INCLUDE ships no signer
identifiers, so `take-group` and `session-disjoint` are both *proxies* for
signer-independence -- they hold out recording runs and recording sessions and
hope those track the person signing. Clips recorded as
``custom/<signer>/<label>/<take>.mp4`` carry the signer explicitly, so the
boundary can be drawn where it actually belongs.

Two modes, because new footage answers two different questions:

``--mode eval``
    New signers become **test only**, INCLUDE stays as train. This measures how
    the existing model generalises to people it has never seen, which is the
    number the whole project has been approximating. Run this first: it is the
    validation of every session-disjoint figure reported so far.

``--mode train``
    New signers are split signer-disjointly among train/val/test. This is what
    actually raises accuracy, but it can only be judged once ``eval`` has
    established the baseline it is supposed to improve.

Usage::

    python -m islvit.data.ingest_custom --mode eval
    python -m islvit.data.ingest_custom --mode train --test-signers priya
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

CUSTOM_CACHE = Path("cache_custom")
BASE_SPLIT = Path("splits/full263__session-disjoint-foldval.csv")


def normalise(label: str) -> str:
    """'94. good' and 'good' must resolve to the same class."""
    text = re.sub(r"^[0-9]+\.\s*", "", str(label).lower().strip())
    text = re.sub(r"[^a-z ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def cached_clips() -> list[tuple[str, str, str]]:
    """(video_path, signer, label) for every clip present in the custom cache."""
    index_path = CUSTOM_CACHE / "index.csv"
    if not index_path.exists():
        raise SystemExit(
            f"{index_path} missing -- extract first:\n"
            f"  python -m islvit.data.crops --corpus custom --cache-dir {CUSTOM_CACHE} --crop-size 128"
        )
    with index_path.open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["cached"] == "1"]

    clips = []
    for row in rows:
        parts = Path(row["video_path"]).parts
        if len(parts) < 3:
            raise SystemExit(
                f"{row['video_path']} is not custom/<signer>/<label>/<take> -- "
                "the signer directory is what makes this split meaningful"
            )
        clips.append((row["video_path"], parts[0], parts[1]))
    return clips


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a signer-disjoint split from new recordings")
    parser.add_argument("--mode", choices=("eval", "train"), default="eval")
    parser.add_argument("--base", type=str, default=str(BASE_SPLIT), help="INCLUDE split to build on")
    parser.add_argument(
        "--test-signers",
        type=str,
        default=None,
        help="comma-separated signers held out (mode=train); default: the last one alphabetically",
    )
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    clips = cached_clips()
    signers = sorted({signer for _, signer, _ in clips})
    by_signer = Counter(signer for _, signer, _ in clips)
    print(f"{len(clips)} clips from {len(signers)} signers: {dict(by_signer)}")

    with Path(args.base).open(encoding="utf-8") as handle:
        include = list(csv.DictReader(handle))
    fieldnames = list(include[0].keys())
    label_of = {normalise(row["label"]): row["label"] for row in include}

    # A recorded word the model was never trained on cannot be scored; report it
    # rather than dropping it silently, since it usually means a typo in a folder
    # name rather than a genuinely new sign.
    unknown = sorted({label for _, _, label in clips if normalise(label) not in label_of})
    if unknown:
        print(f"  WARNING: {len(unknown)} recorded labels are not INCLUDE classes: {unknown[:8]}")
    usable = [(path, signer, label) for path, signer, label in clips if normalise(label) in label_of]
    print(f"  {len(usable)} clips map to INCLUDE classes ({len({l for _, _, l in usable})} distinct)")

    if args.mode == "eval":
        test_signers = set(signers)
    else:
        test_signers = set(
            args.test_signers.split(",") if args.test_signers else signers[-1:]
        )
        missing = test_signers - set(signers)
        if missing:
            raise SystemExit(f"unknown signers: {sorted(missing)}")
    print(f"  held out as test: {sorted(test_signers)}")

    out_path = Path(args.out) if args.out else Path("splits") / (
        f"custom__signer-disjoint-{args.mode}.csv"
    )
    rows_out = []
    for row in include:
        # In eval mode the INCLUDE test clips are dead weight -- the point is to
        # score on new signers -- but they are demoted to train rather than
        # dropped, so no data is wasted and the class list stays identical.
        split = "train" if args.mode == "eval" and row["split"] == "test" else row["split"]
        rows_out.append({**row, "split": split, "corpus": "include", "signer": ""})

    counts: dict[str, int] = defaultdict(int)
    for path, signer, label in usable:
        split = "test" if signer in test_signers else "train"
        counts[split] += 1
        rows_out.append(
            {
                **{name: "" for name in fieldnames},
                "video_path": path,
                "label": label_of[normalise(label)],
                # Its own namespace: a new recording shares no take-run with
                # INCLUDE, so it must never be grouped with one.
                "take_group": f"custom#{signer}#{label}",
                "split": split,
                "corpus": "custom",
                "signer": signer,
            }
        )

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames + ["corpus", "signer"])
        writer.writeheader()
        writer.writerows(rows_out)

    # The whole value of this split is that no signer appears on both sides.
    train_signers = {r["signer"] for r in rows_out if r["split"] == "train" and r["signer"]}
    test_signers_out = {r["signer"] for r in rows_out if r["split"] == "test" and r["signer"]}
    overlap = train_signers & test_signers_out
    assert not overlap, f"signer on both sides of the split: {sorted(overlap)}"
    assert counts["test"] > 0, "no clips landed in test"

    print(f"  custom clips -> train {counts['train']}, test {counts['test']}")
    print(f"  signer disjointness verified (train {sorted(train_signers)} vs test {sorted(test_signers_out)})")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
