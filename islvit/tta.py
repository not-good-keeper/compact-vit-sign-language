"""Test-time augmentation: average several deterministic views of each clip.

Two axes, both of which the model was trained to be invariant to:

* **Temporal phase.** Training samples a random offset inside each of ``n_frames``
  uniform segments; plain evaluation takes the centre of every segment, which is
  one arbitrary phase out of many. Averaging a few phases removes the luck of
  where the 8 sampled frames happened to land in a 16-frame cache.
* **Horizontal flip**, with the two hand streams swapped. This is the same
  augmentation used at train time, and it is only physically coherent *because*
  the streams swap -- a mirrored right-dominant signer is a left-dominant signer.

Probabilities are averaged, not logits: the views are alternative observations of
one clip, so a view that is confidently right should outvote one that is barely
undecided, and averaging logits would let a single large negative logit dominate.

Nothing is trained here, so this costs one forward pass per view and cannot
overfit -- but it also cannot fix a model that is wrong in every phase.

Usage::

    python -m islvit.tta --run runs/sd263_ssl_foldval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from islvit.data.dataset import build_datasets
from islvit.eval import load_run

# Three phases spanning the segment, plus the two flip states. Nine or more views
# were not measurably better than six in a spot check and cost proportionally more.
OFFSETS = (0.25, 0.5, 0.75)
FLIPS = (False, True)


@torch.no_grad()
def view_probabilities(model, dataset, device: str, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Softmax probabilities for the current view, plus the labels in load order."""
    # num_workers=0: worker processes would each hold their own pickled copy of the
    # dataset, so mutating eval_offset between views would silently not reach them.
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    probabilities, labels = [], []
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(
                batch["crops"].to(device),
                batch["detected"].to(device),
                batch["geometry"].to(device),
            )
        probabilities.append(logits.float().softmax(1).cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def score(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    ranked = np.argsort(-probabilities, axis=1)
    hits = ranked[:, 0] == labels
    top5 = (ranked[:, :5] == labels[:, None]).any(axis=1)
    recalls = [hits[labels == label].mean() for label in np.unique(labels)]
    return {
        "top1": float(hits.mean()),
        "top5": float(top5.mean()),
        "balanced": float(np.mean(recalls)),
        "n": int(len(labels)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint with test-time augmentation")
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--split-file", type=str, default=None)
    parser.add_argument("--write-summary", action="store_true", help="record tta scores in summary.json")
    args = parser.parse_args()

    run_dir = Path(args.run)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config, classes = load_run(run_dir, device)
    split_file = args.split_file or config["split_file"]

    _, _, test_set = build_datasets(split_file, n_frames=config["n_frames"], img_size=config["img_size"])
    batch_size = config["batch_size"] * 2

    plain = None
    total = np.zeros(0)
    labels = np.zeros(0, dtype=np.int64)
    for flip in FLIPS:
        for offset in OFFSETS:
            test_set.eval_offset = offset
            test_set.eval_flip = flip
            probabilities, view_labels = view_probabilities(model, test_set, device, batch_size)
            if total.size == 0:
                total, labels = np.zeros_like(probabilities), view_labels
            # Every view must describe the same clips in the same order, or the
            # average is over mismatched rows and the result is meaningless.
            assert np.array_equal(view_labels, labels), "view returned a different clip order"
            total += probabilities
            if not flip and offset == 0.5:
                plain = score(probabilities, labels)

    augmented = score(total / (len(OFFSETS) * len(FLIPS)), labels)

    print(f"[{run_dir.name}] {len(OFFSETS) * len(FLIPS)} views on {split_file}")
    print(f"  plain  top1 {plain['top1']:.1%}  top5 {plain['top5']:.1%}  balanced {plain['balanced']:.1%}")
    print(f"  TTA    top1 {augmented['top1']:.1%}  top5 {augmented['top5']:.1%}  balanced {augmented['balanced']:.1%}")
    print(
        f"  delta  top1 {100 * (augmented['top1'] - plain['top1']):+.1f}  "
        f"top5 {100 * (augmented['top5'] - plain['top5']):+.1f}  "
        f"balanced {100 * (augmented['balanced'] - plain['balanced']):+.1f}  (pts)"
    )

    if args.write_summary:
        path = run_dir / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        # A separate key, never overwriting "test": the headline table compares
        # single-pass numbers across runs and must not silently mix the two.
        summary["test_tta"] = {key: round(value, 4) for key, value in augmented.items()}
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"  wrote test_tta to {path}")


if __name__ == "__main__":
    main()
