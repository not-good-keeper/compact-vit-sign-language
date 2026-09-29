"""How much does averaging train-only models gain on validation?

Every model here was trained on the 1,957-clip train split only, so validation is
honest for all of them. Probabilities are 6-view TTA, cached to disk.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from islvit.data.dataset import IncludeCrops
from islvit.eval import load_run
from islvit.mask50 import deployed_columns
from islvit.tta import FLIPS, OFFSETS, view_probabilities

SPLIT = "splits/full263__session-disjoint.csv"
RUNS = ["dev_lm_base_s0", "dev_lm_base_s1", "dev_lm_aug_s0", "dev_lm_aug_s1", "dev_lm_aug2_s0",
        "dev_lm_interp_s0", "dev_lm_vel_s0", "dev_lm_base1000_s0", "dev_lm_aug_pair_s0",
        "dev_lm_aug_wrist_s0"]
OUT = Path("runs/_val_probs")
OUT.mkdir(exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

probs, labels, allowed = {}, None, None
for run in RUNS:
    cache = OUT / f"{run}.npz"
    if cache.exists():
        z = np.load(cache)
        probs[run], labels, allowed = z["p"], z["y"], z["allowed"]
        continue
    model, config, classes = load_run(Path("runs") / run, device)
    ds = IncludeCrops(SPLIT, "val", n_frames=16, img_size=64,
                      label_to_index={c: i for i, c in enumerate(classes)}, landmarks=True,
                      lm_interp=config.get("lm_interp", False))
    total = 0
    for flip in FLIPS:
        for off in OFFSETS:
            ds.eval_offset, ds.eval_flip = off, flip
            p, labels = view_probabilities(model, ds, device, 128)
            total = total + p
    allowed = deployed_columns(classes)
    probs[run] = total / 6
    np.savez(cache, p=probs[run], y=labels, allowed=allowed)
    del model
    torch.cuda.empty_cache()

rows = np.isin(labels, allowed)
keep = np.zeros(next(iter(probs.values())).shape[1], bool)
keep[allowed] = True


def masked_acc(p):
    return (np.where(keep, p, 0)[rows].argmax(1) == labels[rows]).mean()


def wide_acc(p):
    return (p.argmax(1) == labels).mean()


singles = {r: (masked_acc(p), wide_acc(p)) for r, p in probs.items()}
for r, (m, w) in singles.items():
    print(f"  {r:<22s} masked {m:.1%}  wide {w:.1%}")
ms = np.array([v[0] for v in singles.values()])
print(f"single models: masked mean {ms.mean():.1%} (range {ms.min():.1%}-{ms.max():.1%})")

for k in (2, 3, 5, len(RUNS)):
    accs = [masked_acc(np.mean([probs[r] for r in combo], axis=0))
            for combo in itertools.combinations(RUNS, k)]
    widest = [wide_acc(np.mean([probs[r] for r in combo], axis=0))
              for combo in itertools.combinations(RUNS, k)]
    print(f"ensembles of {k:2d}: masked mean {np.mean(accs):.1%} (range {min(accs):.1%}-{max(accs):.1%}), "
          f"wide mean {np.mean(widest):.1%}  [{len(accs)} combinations]")
