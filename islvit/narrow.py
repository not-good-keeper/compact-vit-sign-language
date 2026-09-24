"""Fine-tune a wide-vocabulary model on the objective it is deployed with.

The 262-word model is trained with a 262-way softmax and deployed with a 50-way
one: at inference the other 212 columns are masked out. Those are different
objectives. Training pushes probability mass away from all 261 wrong words; the
deployed model only ever has to separate the 50 it can say. Capacity spent keeping
"thirsty" apart from a word that is not in the menu is wasted at deployment.

This runs a short second stage that trains exactly what is deployed: the 50-word
training clips, with the softmax restricted to the 50 deployed columns. The wide
stage has already learned from five times the data; this stage only re-weights
what it learned toward the decision the product actually makes.

Two things keep it honest, because there is no validation set:

* **No checkpoint selection.** The epoch count is fixed before the run and the
  final EMA weights are the result. Picking the best epoch by test accuracy would
  be tuning on the test set.
* **Paired comparison.** Each seed is scored against its *own* starting
  checkpoint on the same 472 clips, so seed-to-seed spread (3.6 points here)
  cancels instead of swamping the effect.

``--int4`` runs the same stage with 4-bit fake quantisation, so the deployed
objective and the deployed precision can be trained together.

Usage::

    python -m islvit.narrow --run runs/f16_262w_s0 --epochs 150
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from islvit.data.dataset import IncludeCrops, model_inputs
from islvit.eval import load_run
from islvit.export import int4_targets
from islvit.mask50 import DEPLOYED_SPLIT, deployed_columns, evaluate_masked, report
from islvit.qat import Int4Faker, quantised_copy
from islvit.train import ModelEma, build_param_groups, cosine_schedule, mixup_batch, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune on the deployed sub-vocabulary")
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", default="configs/full263_v2.yaml")
    parser.add_argument("--split-file", default=DEPLOYED_SPLIT)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--int4", action="store_true", help="train through 4-bit fake quantisation")
    parser.add_argument("--group", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    source = Path(args.run)
    tag = args.tag or f"{source.name}_narrow" + ("_int4" if args.int4 else "")
    output_dir = Path("runs") / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config, classes = load_run(source, device)
    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    allowed = torch.as_tensor(deployed_columns(classes, args.split_file), device=device)
    # Wide label index -> position among the deployed columns.
    remap = torch.full((len(classes),), -1, dtype=torch.long, device=device)
    remap[allowed] = torch.arange(len(allowed), device=device)

    train_set = IncludeCrops(
        args.split_file, "train", train=True, n_frames=config["n_frames"],
        img_size=config["img_size"], label_to_index={c: i for i, c in enumerate(classes)},
        flip_prob=base.get("flip_prob", 0.5), color_jitter=base.get("color_jitter", 0.2),
        stream_dropout=base.get("stream_dropout", 0.1),
        grayscale_prob=base.get("grayscale_prob", 0.0), crop_scale=base.get("crop_scale", 0.8),
        resolution_jitter=base.get("resolution_jitter", 0.5),
        landmarks=config.get("landmarks", False),
    )
    workers = base.get("num_workers", 4)
    loader = DataLoader(train_set, batch_size=base["batch_size"], shuffle=True, drop_last=True,
                        num_workers=workers, pin_memory=device == "cuda",
                        persistent_workers=workers > 0)

    before = evaluate_masked(model, config, classes, device, args.split_file, source.name)
    print(f"[{tag}] {len(train_set)} training clips, {len(allowed)} deployed columns, "
          f"{args.epochs} epochs fixed in advance")
    report(before)

    faker = Int4Faker(model, int4_targets(model), args.group) if args.int4 else None
    optimizer = torch.optim.AdamW(
        build_param_groups(model, args.lr, base["weight_decay"], base.get("backbone_lr_scale", 0.1)),
        betas=(0.9, 0.999))
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    ema = ModelEma(model, base.get("ema_decay", 0.99))
    smoothing = base.get("label_smoothing", 0.1)
    mixup_alpha, mixup_prob = base.get("mixup", 0.0) or 0.0, base.get("mixup_prob", 0.5)
    total_steps, warmup, step = args.epochs * len(loader), len(loader), 0

    history = (output_dir / "history.csv").open("w", newline="", encoding="utf-8")
    writer = csv.writer(history)
    writer.writerow(["epoch", "lr", "train_loss"])
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in loader:
            crops = batch["crops"].to(device, non_blocking=True)
            detected = batch["detected"].to(device, non_blocking=True)
            geometry = batch["geometry"].to(device, non_blocking=True)
            labels = remap[batch["label"].to(device, non_blocking=True)]
            extras = model_inputs(batch, device)
            assert (labels >= 0).all(), "a training label is outside the deployed vocabulary"
            # Soft targets over the 50 deployed columns only: mixed if mixup fires,
            # then label-smoothed, matching the wide stage's recipe.
            if mixup_alpha and np.random.rand() < mixup_prob:
                crops, geometry, targets = mixup_batch(crops, geometry, labels, len(allowed), mixup_alpha,
                                                       extras=extras)
            else:
                targets = torch.nn.functional.one_hot(labels, len(allowed)).float()
            targets = targets * (1 - smoothing) + smoothing / len(allowed)

            for group, lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = lr * cosine_schedule(step, total_steps, warmup)
            optimizer.zero_grad(set_to_none=True)
            if faker:
                faker.quantise_()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(crops, detected, geometry, **extras)[:, allowed]
            loss = torch.sum(-targets * logits.float().log_softmax(1), dim=1).mean()
            loss.backward()
            if faker:
                faker.restore_()
            nn.utils.clip_grad_norm_(model.parameters(), base.get("grad_clip", 1.0))
            optimizer.step()
            ema.update(model)
            running += loss.item()
            step += 1
        writer.writerow([epoch, f"{optimizer.param_groups[0]['lr']:.2e}", f"{running / len(loader):.4f}"])
        history.flush()
        if epoch % 25 == 0 or epoch == args.epochs - 1:
            print(f"  ep {epoch:4d}  loss {running / len(loader):.3f}", flush=True)
    history.close()

    # The final EMA weights are the result -- no epoch was chosen by looking at test.
    final = quantised_copy(ema.module, int4_targets(ema.module), args.group) if args.int4 \
        else copy.deepcopy(ema.module)
    torch.save({"model": final.state_dict(), "config": config, "classes": classes,
                "epoch": args.epochs - 1}, output_dir / "best.pt")
    after = evaluate_masked(final, config, classes, device, args.split_file, tag)
    print(f"[{tag}] after")
    report(after)

    delta = after["masked_tta"]["top1"] - before["masked_tta"]["top1"]
    b, a = np.array(before["hits_masked_tta"]), np.array(after["hits_masked_tta"])
    summary = {
        "tag": tag, "source": source.name, "split_file": config["split_file"],
        "narrow_split": args.split_file, "n_frames": config["n_frames"],
        "img_size": config["img_size"], "seed": args.seed,
        "recipe": {"epochs": args.epochs, "lr": args.lr, "int4": args.int4, "group": args.group,
                   "selection": "final epoch, fixed in advance"},
        "masked50_before": {k: before[k] for k in ("wide_tta", "masked_plain", "masked_tta")},
        "masked50": {k: after[k] for k in ("wide_tta", "masked_plain", "masked_tta",
                                            "video_paths", "hits_masked_tta")},
        "paired": {"delta_top1": round(delta, 4), "gained": int(((a == 1) & (b == 0)).sum()),
                   "lost": int(((a == 0) & (b == 1)).sum())},
        "val": {"top1": 0.0, "top5": 0.0, "balanced": 0.0, "n": 0},
        "test": after["masked_tta"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[{tag}] masked-50 TTA {before['masked_tta']['top1']:.1%} -> {after['masked_tta']['top1']:.1%} "
          f"({100 * delta:+.1f}); clips gained {summary['paired']['gained']}, "
          f"lost {summary['paired']['lost']}")


if __name__ == "__main__":
    main()
