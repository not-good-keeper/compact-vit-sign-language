"""Distil the ensemble into a model small enough to ship.

The accuracy target and the size target pull against each other: 75.8 % top-1 is a
five-member ensemble weighing ~20 MB, and the sub-2 MB budget allows one model of
about 2 M parameters. Distillation is the only mechanism that addresses both -- the
student learns the ensemble's *output distribution*, which carries information the
hard labels do not (which words this ensemble confuses, and how much).

Three decisions worth stating, because each has a cheaper wrong version:

**Depth, not width.** The student is 192-wide like the teacher and shallower.
Width 192 is what DeiT-Tiny and the iSign SSL checkpoint were trained at, so a
narrower student forfeits both initialisations -- worth roughly +6 and +10 points
respectively, far more than the parameters saved. Cutting blocks keeps the
survivors loadable. ``192 / 2+2`` measures at 1.95 MB packed (islvit.export).

**Evenly spaced blocks, not the first N.** Initialising a 2-block student from a
4-block checkpoint takes blocks 0 and 2, not 0 and 1: transformer blocks
specialise by depth, and taking a contiguous prefix gives the student two early
feature extractors and no late ones.

**The teacher sees the same augmented batch as the student.** Precomputing teacher
logits once would be far cheaper, but augmentation changes the input every epoch,
so precomputed targets would describe a different image than the student is looking
at -- silently teaching it to match the wrong distribution.

Usage::

    python -m islvit.distill --teacher runs/f16_clean_s0 runs/f32_clean_s0 \
        --student-depth 2 2 --tag distil_2x2_s0 --epochs 2000
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from islvit.data.dataset import build_datasets
from islvit.eval import load_run
from islvit.models.isl_vit import ISLViT, count_parameters
from islvit.train import (
    ModelEma,
    build_param_groups,
    cosine_schedule,
    evaluate,
    mixup_batch,
    resize_position_embeddings,
    set_seed,
)


def select_blocks(state: dict, prefix: str, teacher_depth: int, student_depth: int) -> dict:
    """Remap a deeper checkpoint's blocks onto a shallower student, evenly spaced.

    ``prefix`` is the literal key prefix, which differs between the two stages: the
    spatial blocks are ``spatial.blocks.N.`` but the temporal ones sit at the root as
    ``blocks.N.``. Guessing a symmetric ``temporal.blocks.`` matches nothing, and the
    shape filter downstream then quietly keeps blocks 0 and 1 -- handing the student a
    contiguous prefix, the exact failure this function exists to avoid. It failed that
    way once; hence the assertion below rather than a print.
    """
    if student_depth >= teacher_depth:
        return state
    matched = [key for key in state if key.startswith(prefix)]
    assert matched, f"prefix {prefix!r} matched no keys -- block selection would silently no-op"

    picks = np.linspace(0, teacher_depth - 1, student_depth).round().astype(int).tolist()
    remapped, dropped = {}, 0
    for key, value in state.items():
        if not key.startswith(prefix):
            remapped[key] = value
            continue
        index = int(key[len(prefix):].split(".")[0])
        if index in picks:
            new_index = picks.index(index)
            remapped[f"{prefix}{new_index}." + key[len(prefix):].split(".", 1)[1]] = value
        else:
            dropped += 1
    kept = len(matched) - dropped
    assert dropped > 0 and kept > 0, (
        f"{prefix}: {len(matched)} matched, {kept} kept, {dropped} dropped -- "
        f"expected to drop {teacher_depth - student_depth} of {teacher_depth} blocks"
    )
    print(f"  {prefix[:-1]}: kept blocks {picks} of {teacher_depth} "
          f"({kept} tensors, dropped {dropped})")
    return remapped


def distillation_loss(student_logits, teacher_logits, targets, criterion, alpha, temperature):
    """Soft KL against the teacher plus the ordinary supervised term.

    The T^2 factor restores the gradient magnitude that softening by T removes, so
    alpha keeps meaning the same thing as temperature is varied (Hinton et al.).
    ``targets`` may be hard labels or mixup-blended soft targets.
    """
    soft = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.log_softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
        log_target=True,
    ) * (temperature ** 2)
    hard = criterion(student_logits, targets)
    return alpha * soft + (1 - alpha) * hard, soft.item(), hard.item()


@torch.no_grad()
def teacher_logits(teachers, crops, detected, geometry, device):
    """Mean softmax over the ensemble, returned as logits for the KL term.

    Probabilities are averaged and then re-logged rather than averaging logits:
    the members are alternative opinions about one clip, and averaging logits lets
    a single large negative value from one member dominate the consensus.
    """
    total = None
    for model in teachers:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out = model(crops, detected, geometry)
        probability = out.float().softmax(1)
        total = probability if total is None else total + probability
    return (total / len(teachers)).clamp_min(1e-8).log()


def main() -> None:
    parser = argparse.ArgumentParser(description="Distil an ensemble into a small student")
    parser.add_argument("--teacher", type=str, nargs="+", required=True)
    parser.add_argument("--config", type=str, default="configs/full263_v2.yaml")
    parser.add_argument("--split-file", type=str, default=None)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--student-depth", type=int, nargs=2, default=(2, 2),
                        metavar=("SPATIAL", "TEMPORAL"))
    parser.add_argument("--student-dim", type=int, default=192,
                        help="leave at 192 unless deliberately forfeiting the pretrained init")
    parser.add_argument("--init-from", type=str, default="runs/isign_mim/pretrained.pt")
    parser.add_argument("--alpha", type=float, default=0.7, help="weight on the teacher term")
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution-jitter", type=float, default=0.5)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config["n_frames"], config["img_size"], config["seed"] = args.n_frames, args.img_size, args.seed
    config["epochs"] = args.epochs
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Teachers first: they fix the split and the class list the student must match.
    teachers, split_file, classes = [], args.split_file, None
    for name in args.teacher:
        model, tconfig, tclasses = load_run(Path(name), device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        teachers.append(model)
        split_file = split_file or tconfig["split_file"]
        if classes is None:
            classes = tclasses
        # Averaging over teachers is only meaningful if index k means the same word
        # in every member; a silent mismatch would train the student on noise.
        assert tclasses == classes, f"{name} disagrees on the class list"
    print(f"[{args.tag}] {len(teachers)} teacher(s), {len(classes)} classes, split {split_file}")

    train_set, val_set, test_set = build_datasets(
        split_file, n_frames=args.n_frames, img_size=args.img_size,
        flip_prob=config.get("flip_prob", 0.5), color_jitter=config.get("color_jitter", 0.2),
        stream_dropout=config.get("stream_dropout", 0.1),
        grayscale_prob=config.get("grayscale_prob", 0.0),
        crop_scale=config.get("crop_scale", 0.8), resolution_jitter=args.resolution_jitter,
    )

    spatial_depth, temporal_depth = args.student_depth
    student = ISLViT(
        n_classes=train_set.n_classes, n_frames=args.n_frames, img_size=args.img_size,
        patch_size=config.get("patch_size", 16), dim=args.student_dim,
        spatial_depth=spatial_depth, temporal_depth=temporal_depth,
        heads=config.get("heads", 3), drop_path=config.get("drop_path", 0.1),
    )
    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location="cpu", weights_only=False)
        state = {k: v for k, v in checkpoint["backbone"].items() if not k.startswith("head.")}
        state = select_blocks(state, "spatial.blocks.", config.get("spatial_depth", 4), spatial_depth)
        state = select_blocks(state, "blocks.", config.get("temporal_depth", 4), temporal_depth)
        state = resize_position_embeddings(state, student)
        target = student.state_dict()
        # Anything filtered here is a shape or name mismatch the remap did not handle.
        # Report it rather than swallowing it -- a silently dropped block is how the
        # student ended up with a contiguous prefix the first time round.
        skipped = [k for k in state if k not in target or target[k].shape != state[k].shape]
        state = {k: v for k, v in state.items() if k not in skipped}
        missing, _ = student.load_state_dict(state, strict=False)
        print(f"  init {len(state)} tensors from {args.init_from}, "
              f"{len(missing)} left random, {len(skipped)} skipped")
        if skipped:
            print(f"    skipped: {skipped[:4]}{' ...' if len(skipped) > 4 else ''}")
    student = student.to(device)

    stats = count_parameters(student)
    teacher_params = count_parameters(teachers[0])["total_M"] * len(teachers)
    print(f"  student {stats['total_M']:.2f} M vs teachers {teacher_params:.2f} M "
          f"({teacher_params / stats['total_M']:.1f}x compression)")

    loader_kwargs = dict(num_workers=config.get("num_workers", 4), pin_memory=device == "cuda",
                         persistent_workers=config.get("num_workers", 4) > 0)
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True,
                              drop_last=True, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"] * 2, shuffle=False, **loader_kwargs)

    optimizer = torch.optim.AdamW(
        build_param_groups(student, config["lr"], config["weight_decay"],
                           config.get("backbone_lr_scale", 0.1)), betas=(0.9, 0.999))
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    hard_criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.1))
    soft_criterion = nn.CrossEntropyLoss()
    ema = ModelEma(student, config.get("ema_decay", 0.99))

    output_dir = Path("runs") / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    history = (output_dir / "history.csv").open("w", newline="", encoding="utf-8")
    writer = csv.writer(history)
    writer.writerow(["epoch", "lr", "loss", "kl", "ce", "test_top1"])

    total_steps = args.epochs * len(train_loader)
    warmup_steps = config.get("warmup_epochs", 5) * len(train_loader)
    mixup_alpha, step = config.get("mixup", 0.0), 0

    for epoch in range(args.epochs):
        student.train()
        running = kl_sum = ce_sum = 0.0
        for batch in train_loader:
            crops = batch["crops"].to(device, non_blocking=True)
            detected = batch["detected"].to(device, non_blocking=True)
            geometry = batch["geometry"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            targets, criterion = labels, hard_criterion
            if mixup_alpha > 0 and np.random.rand() < config.get("mixup_prob", 0.5):
                crops, geometry, targets = mixup_batch(
                    crops, geometry, labels, train_set.n_classes, mixup_alpha)
                criterion = soft_criterion

            # Teacher runs on the post-augmentation, post-mixup batch -- the exact
            # tensor the student sees, or the two are being asked about different images.
            with torch.no_grad():
                t_logits = teacher_logits(teachers, crops, detected, geometry, device)

            for group, base in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base * cosine_schedule(step, total_steps, warmup_steps)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = student(crops, detected, geometry)
            loss, kl, ce = distillation_loss(
                logits.float(), t_logits, targets, criterion, args.alpha, args.temperature)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), config.get("grad_clip", 1.0))
            optimizer.step()
            ema.update(student)
            running, kl_sum, ce_sum, step = running + loss.item(), kl_sum + kl, ce_sum + ce, step + 1

        n = len(train_loader)
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            test = evaluate(ema.module, test_loader, device)
            writer.writerow([epoch, f"{optimizer.param_groups[0]['lr']:.2e}",
                             f"{running/n:.4f}", f"{kl_sum/n:.4f}", f"{ce_sum/n:.4f}",
                             f"{test['top1']:.4f}"])
            history.flush()
            print(f"  ep {epoch:4d}  loss {running/n:.3f} (kl {kl_sum/n:.3f} ce {ce_sum/n:.3f})"
                  f"  test {test['top1']:.1%}", flush=True)

        torch.save({"model": ema.module.state_dict(), "config": config,
                    "classes": train_set.classes, "epoch": epoch}, output_dir / "best.pt")

    history.close()
    final = evaluate(ema.module, test_loader, device)
    summary = {
        "tag": args.tag, "split_file": split_file, "cache": __import__("os").environ.get("ISLVIT_CACHE", "cache"),
        "img_size": args.img_size, "n_frames": args.n_frames, "seed": args.seed,
        "pretrained": f"distil:{'+'.join(Path(t).name for t in args.teacher)}",
        "params_M": round(stats["total_M"], 3), "best_epoch": args.epochs - 1,
        "recipe": {"alpha": args.alpha, "temperature": args.temperature,
                   "student_depth": list(args.student_depth), "student_dim": args.student_dim,
                   "epochs": args.epochs, "resolution_jitter": args.resolution_jitter},
        "val": {"top1": 0.0, "top5": 0.0, "balanced": 0.0, "n": 0},
        "test": {k: round(v, 4) for k, v in final.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[{args.tag}] TEST top1 {final['top1']:.1%}  top5 {final['top5']:.1%}  "
          f"balanced {final['balanced']:.1%}  (n={final['n']})")


if __name__ == "__main__":
    main()
