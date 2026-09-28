"""Score a wide-vocabulary checkpoint on a narrower deployed vocabulary.

This produces the project's headline number, so it is worth being precise about
what it measures. The model keeps all 262 output columns; at prediction time the
212 words outside the deployed vocabulary are masked to zero probability. That is
exactly what a 50-word product does with this checkpoint -- those words are not in
its menu, so it can never emit them -- and it is therefore the number the 2 MB
size budget should be judged against.

It is *not* a way of flattering the model. The masking is applied at test time
only and to a fixed, pre-declared vocabulary; no test label influences which
columns survive. The wide (262-way) score is printed alongside precisely so the
gap is visible rather than hidden: masking is worth roughly +15 points here, and
a reader should see that it is doing that work.

Training wide and predicting narrow beats training narrow: the 262-word head sees
five times the data and the extra words act as negatives, so on the identical 472
held-out clips it scores 75.6 % against the 50-word specialist's 73.3 %.

Per-clip correctness is kept in the output so two runs can be compared *paired*
(McNemar on the clips where they disagree) rather than by subtracting two noisy
means.

Usage::

    python -m islvit.mask50 --run runs/f16_262w_s0_qat4
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from islvit.data.dataset import IncludeCrops
from islvit.eval import load_run
from islvit.tta import FLIPS, OFFSETS, score, view_probabilities

DEPLOYED_SPLIT = "splits/vocab50clean__session-disjoint.csv"


def deployed_columns(classes: list[str], split_file: str = DEPLOYED_SPLIT) -> np.ndarray:
    """Head indices of the deployed words, in sorted-label order."""
    index = {label: i for i, label in enumerate(classes)}
    with Path(split_file).open(encoding="utf-8") as handle:
        wanted = sorted({row["label"] for row in csv.DictReader(handle)})
    # Every deployed word must exist in the wide head, or masking silently drops it.
    absent = [w for w in wanted if w not in index]
    assert not absent, f"{len(absent)} deployed words missing from the checkpoint: {absent[:5]}"
    return np.array([index[w] for w in wanted])


def evaluate_masked(model, config: dict, classes: list[str], device: str,
                    split_file: str = DEPLOYED_SPLIT, name: str = "",
                    eval_file: str | None = None, eval_split: str = "test") -> dict:
    """Wide and masked scores, single view and 6-view TTA, plus per-clip hits.

    By default this scores the deployed test set. ``eval_file``/``eval_split``
    score another split instead -- the validation split, for choosing between
    recipes without touching test. The deployed vocabulary always comes from
    ``split_file``; wide scores use every clip, masked scores only the clips whose
    word is deployed.
    """
    model.eval()
    allowed = deployed_columns(classes, split_file)
    label_to_index = {label: i for i, label in enumerate(classes)}
    test_set = IncludeCrops(eval_file or split_file, eval_split, train=False,
                            n_frames=config["n_frames"],
                            img_size=config["img_size"], label_to_index=label_to_index,
                            landmarks=config.get("landmarks", False),
                            lm_interp=config.get("lm_interp", False))

    total, labels, plain = np.zeros(0), np.zeros(0, dtype=np.int64), None
    for flip in FLIPS:
        for offset in OFFSETS:
            test_set.eval_offset, test_set.eval_flip = offset, flip
            probabilities, view_labels = view_probabilities(model, test_set, device,
                                                            config.get("batch_size", 64) * 2)
            if total.size == 0:
                total, labels = np.zeros_like(probabilities), view_labels
            assert np.array_equal(view_labels, labels), "view returned a different clip order"
            total += probabilities
            if not flip and offset == 0.5:
                plain = probabilities.copy()
    averaged = total / (len(OFFSETS) * len(FLIPS))

    keep = np.zeros(averaged.shape[1], dtype=bool)
    keep[allowed] = True
    rows = np.isin(labels, allowed)

    def masked(probabilities):
        return np.where(keep, probabilities, 0.0)[rows]

    results = {
        "run": name, "split_file": split_file, "eval_file": eval_file or split_file,
        "eval_split": eval_split, "n": int(rows.sum()), "n_wide": int(len(labels)),
        "n_deployed": int(len(allowed)), "n_head": len(classes),
        "wide_plain": score(plain, labels), "wide_tta": score(averaged, labels),
        "masked_plain": score(masked(plain), labels[rows]),
        "masked_tta": score(masked(averaged), labels[rows]),
        # In evaluation-set order, so any two runs line up clip for clip.
        "video_paths": [p for p, r in zip(test_set.video_paths, rows) if r],
        "hits_masked_tta": (masked(averaged).argmax(1) == labels[rows]).astype(int).tolist(),
    }
    return results


def report(results: dict) -> None:
    for key in ("wide_plain", "wide_tta", "masked_plain", "masked_tta"):
        r = results[key]
        print(f"  {key:13s} top1 {r['top1']:6.1%}  top5 {r['top5']:6.1%}  balanced {r['balanced']:6.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a wide checkpoint on a deployed sub-vocabulary")
    parser.add_argument("--run", required=True)
    parser.add_argument("--split-file", default=DEPLOYED_SPLIT)
    parser.add_argument("--out", default=None)
    parser.add_argument("--eval-file", default=None, help="score this split file instead (e.g. for val)")
    parser.add_argument("--eval-split", default="test")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config, classes = load_run(Path(args.run), device)
    results = evaluate_masked(model, config, classes, device, args.split_file, Path(args.run).name,
                              eval_file=args.eval_file, eval_split=args.eval_split)
    print(f"[{results['run']}] {results['n']} clips, {results['n_deployed']} deployed words "
          f"of {results['n_head']} head columns")
    report(results)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
