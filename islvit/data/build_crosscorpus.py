"""Build the cross-corpus test set: train on INCLUDE, test on CISLR.

Every number this project has reported is measured inside INCLUDE -- one studio,
one city, one recording setup. ``session-disjoint`` holds out recording sessions
as a *proxy* for unseen signers, because INCLUDE records no signer identity at
all. That proxy has never been checked against actual different people.

CISLR was recorded independently: different signers, rooms, cameras, lighting.
218 of INCLUDE's 262 words also appear there, covered by ~609 cached clips. Those
clips are a genuine cross-corpus, cross-signer test set -- disjoint from INCLUDE
by construction, no clustering heuristic required. It is the closest thing to a
real-world generalisation measurement available without recording new footage.

**This costs the supervised CISLR merge.** The same 609 clips are the only ones
the merge could add to training (they are the overlap, by definition), so the
corpus can serve as held-out test *or* as extra training rows, not both. Test
wins: the merge was already measured at p=0.45 and is structurally thin anyway
(~2.4 clips per class over 202 classes), whereas nothing else can tell us whether
the session-disjoint numbers survive contact with new signers.

**Leak warning.** ``runs/cislr_mim`` was pretrained on *all* CISLR clips,
including these. Checkpoints from it cannot honestly be scored here. Use an
iSign-pretrained checkpoint (never saw CISLR) or re-pretrain with
``--restrict-to`` and the allow-list this script writes.

Usage::

    python -m islvit.data.build_crosscorpus
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from islvit.data.merge_cislr import CISLR_CACHE, CISLR_INDEX, cislr_rows, normalise

DEFAULT_BASE = Path("splits/full263__session-disjoint-foldval.csv")
TEST_OUT = Path("splits/crosscorpus__cislr-test.csv")
ALLOW_OUT = Path("splits/cislr__pretrain-allowed.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the INCLUDE-train / CISLR-test split")
    parser.add_argument("--base", type=str, default=str(DEFAULT_BASE))
    parser.add_argument(
        "--train-word-frac",
        type=float,
        default=0.0,
        help="fraction of the overlapping WORDS whose CISLR clips go to TRAIN instead of "
        "test. 0.0 (default) holds all of CISLR out -- zero-shot corpus transfer. 0.5 "
        "splits words in half, so the model sees CISLR's domain during training but is "
        "still tested on words it never saw in CISLR. Splitting by word rather than by "
        "clip is deliberate: a clip-level split would let the model memorise the exact "
        "signs it is tested on, measuring recall instead of transfer.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-genuine",
        type=float,
        default=0.0,
        help="detection-quality floor for TEST clips; default 0 keeps every cached clip, "
        "because filtering the test set to clips our own detector handles well is "
        "optimistic bias. Quality is recorded per row so a filtered subset can be "
        "scored separately.",
    )
    args = parser.parse_args()

    with Path(args.base).open(encoding="utf-8") as handle:
        include = list(csv.DictReader(handle))
    fieldnames = list(include[0].keys())
    label_of = {normalise(row["label"]): row["label"] for row in include}

    candidates = cislr_rows(args.min_genuine)
    matched = [(video, label_of[gloss], rate) for video, gloss, rate in candidates if gloss in label_of]
    matched.sort()

    covered = sorted({label for _, label, _ in matched})
    print(f"CISLR cached clips considered: {len(candidates)}")
    print(f"  matched to an INCLUDE class: {len(matched)} clips over {len(covered)} words")
    good = sum(1 for _, _, rate in matched if rate >= 0.25)
    print(f"  of those, {good} have >=25% genuine hand detection ({good/max(1,len(matched)):.0%})")

    # Which overlapping WORDS contribute their CISLR clips to training.
    train_words: set[str] = set()
    if args.train_word_frac > 0:
        shuffled = list(covered)
        random.Random(args.seed).shuffle(shuffled)
        train_words = set(shuffled[: int(len(shuffled) * args.train_word_frac)])
        print(f"  CISLR words routed to TRAIN: {len(train_words)} / {len(covered)} "
              f"(their clips join INCLUDE; the rest stay held out)")

    # INCLUDE's own train/test boundary is PRESERVED, not collapsed into train.
    #
    # Promoting every INCLUDE clip to train is tempting -- the test corpus is
    # CISLR, so nothing needs holding back -- and the first version did exactly
    # that. It made the split unusable for training: a model fitted on it scored
    # 100.0% on INCLUDE session-disjoint, because all 1,010 of those test clips
    # had been in its training set. It also silently changed training-set size
    # (2,845 -> 3,855), confounding any comparison against a baseline trained on
    # the foldval split.
    #
    # Held-out INCLUDE rows become "unused" here: excluded from training, and
    # still scoreable via the foldval split file afterwards.
    # Distinct filenames per mode, or the zero-shot and mixed experiments
    # silently overwrite each other's split.
    suffix = "" if args.train_word_frac == 0 else f"-mix{int(args.train_word_frac*100)}"
    test_out = TEST_OUT.with_name(TEST_OUT.stem + suffix + TEST_OUT.suffix)
    out_fields = fieldnames + ["corpus", "genuine"]
    with test_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields)
        writer.writeheader()
        include_train = 0
        for row in include:
            if row["split"] == "unused":
                continue
            # Keep INCLUDE's own held-out sessions out of training so the model
            # stays scoreable on them; "unused" rows are dropped by the dataset.
            split = "train" if row["split"] == "train" else "unused"
            writer.writerow({**row, "split": split, "corpus": "include", "genuine": ""})
            include_train += split == "train"
        cislr_train = cislr_test = 0
        for video, label, rate in matched:
            to_train = label in train_words
            cislr_train += to_train
            cislr_test += not to_train
            writer.writerow(
                {
                    **{name: "" for name in fieldnames},
                    "video_path": video,
                    "label": label,
                    "take_group": f"cislr#{video}",
                    "split": "train" if to_train else "test",
                    "corpus": "cislr",
                    "genuine": f"{rate:.4f}",
                }
            )

    # Allow-list for pretraining: every cached CISLR clip EXCEPT the test ones.
    # pretrain.py --restrict-to keeps rows whose split == "train".
    with (CISLR_CACHE / "index.csv").open(encoding="utf-8") as handle:
        cached = [row["video_path"] for row in csv.DictReader(handle) if row["cached"] == "1"]
    # Only the held-out clips need withholding from pretraining. Clips promoted
    # to supervised training are already seen, so excluding them buys nothing.
    held_out = {video for video, label, _ in matched if label not in train_words}
    with ALLOW_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_path", "split"])
        writer.writeheader()
        allowed = 0
        for video in cached:
            if video in held_out:
                continue
            writer.writerow({"video_path": video, "split": "train"})
            allowed += 1

    # The entire value of this split is that test shares nothing with train or
    # with the pretraining pool. Assert both, loudly.
    train_paths = {row["video_path"] for row in include}
    assert not (train_paths & held_out), "a CISLR test clip also appears in the INCLUDE training rows"
    with ALLOW_OUT.open(encoding="utf-8") as handle:
        allow_paths = {row["video_path"] for row in csv.DictReader(handle)}
    assert not (allow_paths & held_out), "a CISLR test clip survived into the pretraining allow-list"

    # A word must never have CISLR clips on both sides -- that turns a transfer
    # test into a memorisation test for that word.
    test_words = {label for video, label, _ in matched if video in held_out}
    assert not (train_words & test_words), "a word has CISLR clips in both train and test"

    print(f"\n  train {include_train} INCLUDE + {cislr_train} CISLR  ->  test {cislr_test} CISLR clips")
    print(f"  wrote {test_out}")
    print(f"  pretraining allow-list: {allowed} of {len(cached)} cached CISLR clips ({len(held_out)} withheld)")
    print(f"  wrote {ALLOW_OUT}")
    print("  leak assertions passed (test disjoint from train AND from the pretraining pool)")


if __name__ == "__main__":
    main()
