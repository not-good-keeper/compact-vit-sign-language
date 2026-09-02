"""Build a vocabulary-restricted split: keep the N best-supported words.

Vocabulary size is a free variable that has been treated as fixed. 262 words is
INCLUDE's choice, not a requirement, and the effect is large -- on an older recipe
262 words scored 29.3 % against 50 words at 46.5 % on the same protocol. For a
product, a small vocabulary recognised reliably beats a large one recognised
badly.

Words are ranked by TOTAL clips and must clear a minimum of held-out clips to be
scoreable at all. Ranking by *training* clips alone is the intuitive choice and is
wrong: INCLUDE gives each word ~16 clips split across sessions, so the words with
the most training clips are precisely those with the fewest test clips, and a
30-word tier built that way had 22 test clips in total.

The session-disjoint structure is preserved exactly -- the same held-out sessions
stay held out. Only rows whose label falls outside the chosen vocabulary are
dropped. Per-word support stays as it was (each kept word keeps all its clips), so
across tiers the variable that changes is the number of classes to separate, not
how much evidence each class has.

Usage::

    python -m islvit.data.build_vocab_tier --words 50
    python -m islvit.data.build_vocab_tier --words 30 --with-cislr
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from islvit.data.merge_cislr import cislr_rows, normalise

DEFAULT_BASE = Path("splits/full263__session-disjoint-foldval.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Restrict a split to the N best-supported words")
    parser.add_argument("--words", type=int, required=True)
    parser.add_argument("--base", type=str, default=str(DEFAULT_BASE))
    parser.add_argument(
        "--with-cislr",
        action="store_true",
        help="also fold in the overlapping CISLR clips as extra training rows; §11.6 "
        "measured this as free (INCLUDE 41.9 %% vs 42.1 %%) and it doubles cross-corpus top-5",
    )
    parser.add_argument(
        "--min-test",
        type=int,
        default=3,
        help="minimum held-out clips a word needs to be scoreable at all; words below "
        "this contribute classes the model can be wrong about but never measured on",
    )
    parser.add_argument("--min-genuine", type=float, default=0.25)
    parser.add_argument(
        "--tb-split",
        type=str,
        default="splits/crosscorpus__cislr-test.csv",
        help="cross-corpus held-out clips to keep OUT of training; \"\" disables the check",
    )
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    with Path(args.base).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fieldnames = list(rows[0].keys())

    # Clips reserved for the cross-corpus test set must never reach training.
    # The first version of this script had no such check, and every tier it built
    # drew its CISLR training rows from inside T-B -- 76/76 at 50 words, 256/256 at
    # 137. No published number was affected, because T-B was only ever scored on
    # INCLUDE-only checkpoints, but the tiers on disk are unusable for cross-corpus
    # evaluation and nothing in the artefacts said so.
    withheld: set[str] = set()
    if args.tb_split:
        tb_path = Path(args.tb_split)
        if not tb_path.exists():
            raise SystemExit(f"{tb_path} not found; pass --tb-split '' to build without the leak check")
        with tb_path.open(encoding="utf-8") as handle:
            withheld = {row["video_path"] for row in csv.DictReader(handle) if row["split"] == "test"}
        print(f"holding out {len(withheld)} cross-corpus test clips from training")

    # Rank by TOTAL support, then require a usable number of test clips.
    #
    # Ranking by training support alone looks right and is actively wrong here:
    # INCLUDE gives each word ~16 clips split across sessions, so the words with
    # the most TRAIN clips are exactly those with the fewest TEST clips. Doing it
    # that way produced a 30-word tier with 22 test clips total -- unmeasurable.
    train_support = Counter(row["label"] for row in rows if row["split"] == "train")
    test_support = Counter(row["label"] for row in rows if row["split"] == "test")
    total = Counter({label: train_support[label] + test_support[label] for label in train_support})

    eligible = [
        label for label, _ in total.most_common()
        if test_support[label] >= args.min_test and train_support[label] >= 4
    ]
    if len(eligible) < args.words:
        raise SystemExit(
            f"only {len(eligible)} words have >={args.min_test} test clips and >=4 train clips; "
            f"asked for {args.words}. Lower --words or --min-test."
        )
    chosen = eligible[: args.words]
    keep = set(chosen)
    print(f"{args.words} words kept, from {len(eligible)} eligible ({len(total)} total)")
    print(f"  train clips/word: {min(train_support[w] for w in keep)}-{max(train_support[w] for w in keep)}")
    print(f"  test  clips/word: {min(test_support[w] for w in keep)}-{max(test_support[w] for w in keep)}")

    out_path = Path(args.out) if args.out else Path("splits") / (
        f"vocab{args.words}{'+cislr' if args.with_cislr else ''}__session-disjoint.csv"
    )
    out_fields = fieldnames + ["corpus"]
    counts: Counter[str] = Counter()
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields)
        writer.writeheader()
        for row in rows:
            if row["label"] not in keep or row["split"] == "unused":
                continue
            writer.writerow({**row, "corpus": "include"})
            counts[row["split"]] += 1

        if args.with_cislr:
            label_of = {normalise(label): label for label in keep}
            added, withheld_hits = 0, 0
            for video, gloss, _rate in cislr_rows(args.min_genuine):
                label = label_of.get(gloss)
                if label is None:
                    continue
                if video in withheld:
                    withheld_hits += 1
                    continue
                writer.writerow(
                    {
                        **{name: "" for name in fieldnames},
                        "video_path": video,
                        "label": label,
                        "take_group": f"cislr#{video}",
                        "split": "train",
                        "corpus": "cislr",
                    }
                )
                added += 1
                counts["train"] += 1
            print(f"  CISLR clips folded into train: {added}")
            if withheld_hits:
                print(f"  CISLR clips skipped because they are held out for T-B: {withheld_hits}")

    # A tier must never quietly lose its test set, and every kept word needs both
    # training and test clips or the class is unscoreable.
    assert counts["test"] > 0, "no test clips survived the vocabulary filter"
    # Re-read what was actually written rather than trusting the loop above: the
    # leak this guards against was introduced by a filter that looked correct at
    # the call site and was never checked against the file it produced.
    if withheld:
        with out_path.open(encoding="utf-8") as handle:
            leaked = [row["video_path"] for row in csv.DictReader(handle)
                      if row["split"] == "train" and row["video_path"] in withheld]
        assert not leaked, f"{len(leaked)} held-out T-B clips reached train, e.g. {leaked[:3]}"
        print(f"  verified: no T-B clip appears in train")
    test_labels = {row["label"] for row in rows if row["split"] == "test" and row["label"] in keep}
    missing = keep - test_labels
    if missing:
        print(f"  note: {len(missing)} kept words have no test clips (unscoreable)")
    print(f"  train {counts['train']}  test {counts['test']}  ({len(keep)} classes)")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
