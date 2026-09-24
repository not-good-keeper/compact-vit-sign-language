"""Post-training INT4: pack a checkpoint, measure the file, score the deployed metric.

Nothing is trained and nothing is selected, so the number this prints cannot have
been tuned on the test set -- which makes it the honest floor for any INT4 claim.
Quantisation-aware training can only be judged against it.

Usage::

    python -m islvit.int4_eval --run runs/f16_262w_lm_s0
"""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path

import torch

from islvit.eval import load_run
from islvit.export import int4_targets, pack_int4
from islvit.mask50 import evaluate_masked, report
from islvit.qat import quantised_copy


def packed_size_mb(model, group: int) -> float:
    packed, _ = pack_int4(copy.deepcopy(model).cpu(), group=group)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        path = Path(handle.name)
    try:
        torch.save(packed, path)
        return path.stat().st_size / 2**20
    finally:
        path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-training INT4 size and accuracy")
    parser.add_argument("--run", required=True)
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config, classes = load_run(Path(args.run), device)
    quantised = quantised_copy(model, int4_targets(model), args.group)
    results = evaluate_masked(quantised, config, classes, device, name=f"{Path(args.run).name}_ptq4")
    size = packed_size_mb(quantised, args.group)
    print(f"[{results['run']}] INT4 group {args.group}, packed {size:.3f} MB")
    report(results)
    results["packed_int4_mb"] = round(size, 3)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
