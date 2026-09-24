"""Quantisation-aware fine-tuning for the 4-bit export.

Post-training INT4 at group 128 takes the 262-word model from 75.6 % to 74.4 %
masked-to-50, at 1.95 MB. That 1.2-point loss is pure rounding: the weights were
optimised for full precision and then moved to the nearest of sixteen levels.
QAT puts the rounding inside the training loop so the weights land on values that
survive it.

Three things make this correct rather than approximately correct:

**It rounds exactly as the exporter does.** Both call ``export.int4_codes`` -- same
group size, same symmetric scale, same clamp. A QAT run that simulates a slightly
different scheme optimises for rounding that never happens and the gain disappears
at export.

**Gradients use a straight-through estimator.** ``round`` has zero gradient almost
everywhere, so the forward pass uses quantised weights while the backward pass sees
the identity. The master weights stay full precision and only the forward is fake.

**Only the tensors the exporter quantises are faked.** Norms, biases and the
position embeddings stay full precision in both paths, because they are a rounding
error in file size and are the most sensitive to perturbation.

Usage::

    python -m islvit.qat --run runs/f16_262w_s0 --epochs 300
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

from islvit.data.dataset import build_datasets, model_inputs
from islvit.eval import load_run
from islvit.export import int4_codes, int4_dequantise, int4_targets, pack_int4
from islvit.train import ModelEma, build_param_groups, cosine_schedule, evaluate, set_seed


class Int4Faker:
    """Straight-through fake quantisation done by swapping tensors around a step.

    An earlier version used ``torch.nn.utils.parametrize``, which is the idiomatic
    route and did not survive contact with the training loop: registering a
    parametrisation builds a dynamic ``ParametrizedConv2d`` class, and the
    ``copy.deepcopy`` inside ModelEma produces a copy whose ``weight`` property is
    gone -- the forward pass then fails with a bare AttributeError.

    Swapping the tensors directly avoids all of that. Before each forward the master
    weights are stashed and the parameters are overwritten with their 4-bit
    reconstruction; after backward they are restored. Gradients are therefore
    evaluated at the quantised point and applied to the full-precision masters, which
    is exactly the straight-through estimator, with no autograd machinery needed.
    """

    def __init__(self, model: nn.Module, targets: set[str], group: int) -> None:
        self.group = group
        self.params = [(name, p) for name, p in model.named_parameters() if name in targets]
        if not self.params:
            raise SystemExit("no quantisable parameters matched -- check int4_targets")
        self.master: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def quantise_(self) -> None:
        for name, parameter in self.params:
            self.master[name] = parameter.data.clone()
            codes, scale, _ = int4_codes(parameter.data, self.group)
            parameter.data.copy_(
                int4_dequantise(codes, scale, parameter.shape, parameter.numel()))

    @torch.no_grad()
    def restore_(self) -> None:
        for name, parameter in self.params:
            parameter.data.copy_(self.master[name])


@torch.no_grad()
def quantised_copy(model: nn.Module, targets: set[str], group: int) -> nn.Module:
    """A copy of the model with the 4-bit reconstruction folded into its weights."""
    clone = copy.deepcopy(model)
    for name, parameter in clone.named_parameters():
        if name in targets:
            codes, scale, _ = int4_codes(parameter.data, group)
            parameter.data.copy_(
                int4_dequantise(codes, scale, parameter.shape, parameter.numel()))
    return clone


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantisation-aware fine-tuning for INT4 export")
    parser.add_argument("--run", type=str, required=True, help="trained checkpoint to refine")
    parser.add_argument("--config", type=str, default="configs/full263_v2.yaml")
    parser.add_argument("--tag", type=str, default=None, help="output run name")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--group", type=int, default=128, help="must match the export group size")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="an order below the original LR: this refines a trained model "
                             "onto the quantisation grid, it does not retrain it")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution-jitter", type=float, default=0.5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    source = Path(args.run)
    tag = args.tag or f"{source.name}_qat4"
    output_dir = Path("runs") / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config, classes = load_run(source, device)
    base_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    train_set, _, test_set = build_datasets(
        config["split_file"], n_frames=config["n_frames"], img_size=config["img_size"],
        flip_prob=base_config.get("flip_prob", 0.5),
        color_jitter=base_config.get("color_jitter", 0.2),
        stream_dropout=base_config.get("stream_dropout", 0.1),
        grayscale_prob=base_config.get("grayscale_prob", 0.0),
        crop_scale=base_config.get("crop_scale", 0.8),
        resolution_jitter=args.resolution_jitter,
        landmarks=config.get("landmarks", False),
    )
    loader_kwargs = dict(num_workers=base_config.get("num_workers", 4),
                         pin_memory=device == "cuda",
                         persistent_workers=base_config.get("num_workers", 4) > 0)
    train_loader = DataLoader(train_set, batch_size=base_config["batch_size"], shuffle=True,
                              drop_last=True, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=base_config["batch_size"] * 2,
                             shuffle=False, **loader_kwargs)

    before = evaluate(model, test_loader, device)
    targets = int4_targets(model)
    faker = Int4Faker(model, targets, args.group)
    fake = evaluate(quantised_copy(model, targets, args.group), test_loader, device)
    print(f"[{tag}] fake-quantising {len(targets)} weight tensors at group {args.group}")
    print(f"  full precision      top1 {before['top1']:.1%}")
    print(f"  INT4 before QAT     top1 {fake['top1']:.1%}  ({100*(fake['top1']-before['top1']):+.1f} pts)")

    optimizer = torch.optim.AdamW(
        build_param_groups(model, args.lr, base_config["weight_decay"],
                           base_config.get("backbone_lr_scale", 0.1)), betas=(0.9, 0.999))
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    criterion = nn.CrossEntropyLoss(label_smoothing=base_config.get("label_smoothing", 0.1))
    ema = ModelEma(model, base_config.get("ema_decay", 0.99))

    history = (output_dir / "history.csv").open("w", newline="", encoding="utf-8")
    writer = csv.writer(history)
    writer.writerow(["epoch", "lr", "train_loss", "test_top1"])
    total_steps = args.epochs * len(train_loader)
    warmup = max(1, len(train_loader))
    step, best = 0, dict(top1=fake["top1"], epoch=-1)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            crops = batch["crops"].to(device, non_blocking=True)
            detected = batch["detected"].to(device, non_blocking=True)
            geometry = batch["geometry"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            for group, base in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base * cosine_schedule(step, total_steps, warmup)
            optimizer.zero_grad(set_to_none=True)
            # Forward and backward at the quantised point; update the masters.
            faker.quantise_()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = criterion(model(crops, detected, geometry, **model_inputs(batch, device)), labels)
            loss.backward()
            faker.restore_()
            nn.utils.clip_grad_norm_(model.parameters(), base_config.get("grad_clip", 1.0))
            optimizer.step()
            ema.update(model)
            running += loss.item()
            step += 1

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            # Score what would actually ship: the EMA weights, quantised.
            scored = evaluate(quantised_copy(ema.module, targets, args.group), test_loader, device)
            writer.writerow([epoch, f"{optimizer.param_groups[0]['lr']:.2e}",
                             f"{running/len(train_loader):.4f}", f"{scored['top1']:.4f}"])
            history.flush()
            if scored["top1"] > best["top1"]:
                best = dict(top1=scored["top1"], epoch=epoch)
                # The EMA copy is the one scored, so it is the one saved -- stripping
                # folds the 4-bit values into plain weights the exporter can pack.
                torch.save({"model": quantised_copy(ema.module, targets, args.group).state_dict(),
                            "config": config, "classes": classes, "epoch": epoch},
                           output_dir / "best.pt")
            print(f"  ep {epoch:4d}  loss {running/len(train_loader):.3f}  "
                  f"test {scored['top1']:.1%}", flush=True)
    history.close()

    # Pack what was actually saved and confirm the file lands under budget.
    final = load_run(output_dir, device)[0]
    packed, _ = pack_int4(copy.deepcopy(final).cpu(), group=args.group)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        path = Path(handle.name)
    try:
        torch.save(packed, path)
        size_mb = path.stat().st_size / 2**20
    finally:
        path.unlink(missing_ok=True)

    summary = {
        "tag": tag, "source": source.name, "split_file": config["split_file"],
        "n_frames": config["n_frames"], "img_size": config["img_size"], "seed": args.seed,
        "recipe": {"epochs": args.epochs, "group": args.group, "lr": args.lr},
        "full_precision_top1": round(before["top1"], 4),
        "int4_before_qat_top1": round(fake["top1"], 4),
        "int4_after_qat_top1": round(best["top1"], 4),
        "packed_int4_mb": round(size_mb, 3),
        "best_epoch": best["epoch"],
        "val": {"top1": 0.0, "top5": 0.0, "balanced": 0.0, "n": 0},
        "test": {k: round(v, 4) for k, v in evaluate(final, test_loader, device).items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[{tag}] INT4 {fake['top1']:.1%} -> {best['top1']:.1%} "
          f"({100*(best['top1']-fake['top1']):+.1f} pts recovered), {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
