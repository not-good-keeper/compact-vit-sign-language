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

Usage::

    python -m islvit.mask50 --run runs/f16_262w_s0_qat4
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np, torch
from islvit.data.dataset import IncludeCrops
from islvit.eval import load_run
from islvit.tta import OFFSETS, FLIPS, view_probabilities, score

def main() -> None:
    parser = argparse.ArgumentParser(description="Score a wide checkpoint on a deployed sub-vocabulary")
    parser.add_argument("--run", required=True)
    parser.add_argument("--split-file", default="splits/vocab50clean__session-disjoint.csv")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config, classes = load_run(Path(args.run), device)
    label_to_index = {label: i for i, label in enumerate(classes)}

    with Path(args.split_file).open(encoding="utf-8") as handle:
        wanted = sorted({row["label"] for row in csv.DictReader(handle)})
    # Every deployed word must exist in the wide head, or masking silently drops it.
    absent = [w for w in wanted if w not in label_to_index]
    assert not absent, f"{len(absent)} deployed words missing from the checkpoint: {absent[:5]}"
    allowed = np.array([label_to_index[w] for w in wanted])

    test_set = IncludeCrops(args.split_file, "test", train=False, n_frames=config["n_frames"],
                            img_size=config["img_size"], label_to_index=label_to_index)
    print(f"[{Path(args.run).name}] {len(test_set)} clips, {len(wanted)} deployed words "
          f"of {len(classes)} head columns")

    total, labels = np.zeros(0), np.zeros(0, dtype=np.int64)
    plain = None
    for flip in FLIPS:
        for offset in OFFSETS:
            test_set.eval_offset, test_set.eval_flip = offset, flip
            probabilities, view_labels = view_probabilities(model, test_set, device,
                                                            config["batch_size"] * 2)
            if total.size == 0:
                total, labels = np.zeros_like(probabilities), view_labels
            assert np.array_equal(view_labels, labels), "view returned a different clip order"
            total += probabilities
            if not flip and offset == 0.5:
                plain = probabilities.copy()
    averaged = total / (len(OFFSETS) * len(FLIPS))

    def masked(probabilities):
        """Zero every column outside the deployed vocabulary, then score."""
        keep = np.zeros(probabilities.shape[1], dtype=bool)
        keep[allowed] = True
        return score(np.where(keep, probabilities, 0.0), labels)

    results = {
        "run": Path(args.run).name, "split_file": args.split_file, "n": int(len(labels)),
        "n_deployed": len(wanted), "n_head": len(classes),
        "wide_plain": score(plain, labels), "wide_tta": score(averaged, labels),
        "masked_plain": masked(plain), "masked_tta": masked(averaged),
    }
    for name in ("wide_plain", "wide_tta", "masked_plain", "masked_tta"):
        r = results[name]
        print(f"  {name:13s} top1 {r['top1']:6.1%}  top5 {r['top5']:6.1%}  balanced {r['balanced']:6.1%}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
