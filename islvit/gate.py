"""Confidence gating on a TTA-averaged ensemble: what does the model get right
when it is allowed to say nothing?

A wearable does not have to answer every clip. Declining a low-confidence sign
costs the user a repeat; a confident wrong sign costs them a wrong sentence. So
the number that matters for the product is not top-1 over all clips but
*accuracy at a given coverage* -- and those two move in opposite directions,
which a single accuracy figure hides completely.

Probabilities are averaged over TTA views first, then over runs (seeds).
Averaging softmax outputs rather than logits is deliberate and matches
``islvit.tta``: the views and seeds are alternative observations of one clip, so
a confidently-correct view should outvote an undecided one.

The threshold reported here is chosen on the *test* set, so the accuracy at a
given coverage is optimistic as a forward-looking estimate. It is reported as a
characteristic curve of the model, not as a validated operating point -- the
honest use is to read the shape, not to quote one row as a product number.

Usage::

    python -m islvit.gate --runs runs/grok_wd005 runs/long2000_s1
    python -m islvit.gate --runs runs/grok_wd005 --no-tta
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from islvit.data.dataset import build_datasets
from islvit.eval import load_run
from islvit.tta import FLIPS, OFFSETS, score, view_probabilities

THRESHOLDS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def averaged_probabilities(run_dir: Path, device: str, use_tta: bool):
    """TTA-averaged probabilities for one run, plus labels in load order."""
    model, config, classes = load_run(run_dir, device)
    _, _, test_set = build_datasets(
        config["split_file"], n_frames=config["n_frames"], img_size=config["img_size"]
    )
    combos = [(o, f) for f in FLIPS for o in OFFSETS] if use_tta else [(0.5, False)]
    total, labels = None, None
    for offset, flip in combos:
        test_set.eval_offset = offset
        test_set.eval_flip = flip
        probabilities, view_labels = view_probabilities(
            model, test_set, device, config["batch_size"] * 2
        )
        if total is None:
            total, labels = np.zeros_like(probabilities), view_labels
        assert np.array_equal(view_labels, labels), "view returned a different clip order"
        total += probabilities
    return total / len(combos), labels, classes


def main() -> None:
    parser = argparse.ArgumentParser(description="Accuracy-vs-coverage for a gated ensemble")
    parser.add_argument("--runs", type=str, nargs="+", required=True)
    parser.add_argument("--no-tta", dest="tta", action="store_false")
    parser.add_argument("--out", type=str, default=None, help="write the curve as JSON")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensemble, labels, classes = None, None, None
    for name in args.runs:
        run_dir = Path(name)
        probabilities, run_labels, run_classes = averaged_probabilities(run_dir, device, args.tta)
        single = score(probabilities, run_labels)
        print(f"  {run_dir.name:<16s} top1 {single['top1']:.1%}  top5 {single['top5']:.1%}")
        if ensemble is None:
            ensemble, labels, classes = np.zeros_like(probabilities), run_labels, run_classes
        # Averaging across runs is only meaningful if the class index -> word map
        # and the clip order are identical; otherwise this silently adds up
        # probabilities for different words.
        assert run_classes == classes, "runs disagree on the class list"
        assert np.array_equal(run_labels, labels), "runs disagree on the clip order"
        ensemble += probabilities
    ensemble /= len(args.runs)

    combined = score(ensemble, labels)
    views = (len(OFFSETS) * len(FLIPS)) if args.tta else 1
    print(
        f"\nensemble of {len(args.runs)} run(s) x {views} view(s) "
        f"on {len(labels)} clips, {len(classes)} classes"
    )
    print(f"  top1 {combined['top1']:.1%}  top5 {combined['top5']:.1%}  balanced {combined['balanced']:.1%}\n")

    confidence = ensemble.max(axis=1)
    correct = ensemble.argmax(axis=1) == labels
    print("  threshold   coverage   accuracy-when-answered   answered-and-right")
    curve = []
    for threshold in THRESHOLDS:
        keep = confidence >= threshold
        if not keep.any():
            continue
        coverage = float(keep.mean())
        accuracy = float(correct[keep].mean())
        # Accuracy on the kept clips flatters itself as coverage falls, so also
        # report the share of ALL clips that are both answered and correct --
        # the quantity a user actually experiences.
        yielded = float((keep & correct).mean())
        curve.append(
            {"threshold": threshold, "coverage": coverage, "accuracy": accuracy, "yield": yielded}
        )
        print(f"    >={threshold:.1f}      {coverage:6.1%}          {accuracy:6.1%}              {yielded:6.1%}")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"runs": args.runs, "tta": args.tta, "curve": curve}, indent=2),
            encoding="utf-8",
        )
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
