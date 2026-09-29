"""Train ISL-ViT-Tiny on cached INCLUDE crops.

Everything here is shaped by one fact: INCLUDE gives ~16 clips per class. That is
far too little to train a ViT from scratch, so the recipe leans hard on transfer
and regularisation --

* the spatial encoder starts from ImageNet DeiT-Tiny and trains at a fraction of
  the head's learning rate, so pretrained features are refined rather than erased;
* stochastic depth, label smoothing, mixup and weight EMA all fight memorisation;
* every number is reported on both the leaky and the take-group split, because on
  this dataset the difference between them is the whole story.

Usage::

    python -m islvit.train --config configs/include50.yaml
    python -m islvit.train --config configs/include50.yaml --overfit-batch
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from islvit.data.dataset import build_datasets, model_inputs
from islvit.models.isl_vit import ISLViT, count_parameters, load_deit_tiny_weights


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_param_groups(model: ISLViT, lr: float, weight_decay: float, backbone_lr_scale: float):
    """Lower LR for the pretrained encoder; no weight decay on norms and biases."""
    groups: dict[tuple[str, bool], dict] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_backbone = name.startswith("spatial.")
        decays = parameter.ndim > 1
        key = ("spatial" if is_backbone else "head", decays)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": lr * (backbone_lr_scale if is_backbone else 1.0),
                "weight_decay": weight_decay if decays else 0.0,
            }
        groups[key]["params"].append(parameter)
    return list(groups.values())


def cosine_schedule(step: int, total: int, warmup: int, min_ratio: float = 0.01) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def resize_position_embeddings(state: dict, model: nn.Module) -> dict:
    """Interpolate a checkpoint's position/time embeddings to this model's shape.

    Everything else in the encoder is shape-agnostic, but ``spatial.pos_embed``
    and ``time_embed`` have one entry per patch and per frame, so a checkpoint
    pretrained at 8 frames x 64 px cannot load into a 16-frame or 96 px model at
    all. Without this the SSL initialisation -- worth ~+10 points -- is simply
    unavailable at any other input geometry, which would confound a capacity
    ablation with a pretraining ablation.

    The spatial grid is interpolated bicubically as a 2-D map (the standard ViT
    treatment; the embeddings are a raster, not a sequence) with the class token
    passed through untouched. Time is 1-D and linear.
    """
    target = model.state_dict()
    resized = dict(state)

    key = "spatial.pos_embed"
    if key in state and key in target and state[key].shape != target[key].shape:
        source, want = state[key], target[key]
        extra = want.shape[1] - int(round(math.sqrt(want.shape[1] - 1)) ** 2)
        old_side = int(round(math.sqrt(source.shape[1] - extra)))
        new_side = int(round(math.sqrt(want.shape[1] - extra)))
        if old_side ** 2 + extra != source.shape[1] or new_side ** 2 + extra != want.shape[1]:
            raise SystemExit(f"{key}: cannot read {source.shape[1]} entries as a square grid + {extra} tokens")
        prefix, grid = source[:, :extra], source[:, extra:]
        grid = grid.reshape(1, old_side, old_side, -1).permute(0, 3, 1, 2)
        grid = F.interpolate(grid.float(), size=(new_side, new_side), mode="bicubic", align_corners=False)
        grid = grid.permute(0, 2, 3, 1).reshape(1, new_side * new_side, -1)
        resized[key] = torch.cat([prefix, grid], dim=1).to(source.dtype)
        print(f"  resized {key}: {old_side}x{old_side} -> {new_side}x{new_side} grid")

    key = "time_embed"
    if key in state and key in target and state[key].shape != target[key].shape:
        source, want = state[key], target[key]
        grid = F.interpolate(
            source.float().permute(0, 2, 1), size=want.shape[1], mode="linear", align_corners=False
        )
        resized[key] = grid.permute(0, 2, 1).to(source.dtype)
        print(f"  resized {key}: {source.shape[1]} -> {want.shape[1]} frames")

    return resized


def _soft_targets(targets, perm, n_classes, lam):
    onehot = torch.zeros(targets.size(0), n_classes, device=targets.device)
    onehot.scatter_(1, targets.unsqueeze(1), 1.0)
    return lam * onehot + (1 - lam) * onehot[perm]


def mixup_batch(crops, geometry, targets, n_classes, alpha, extras=None):
    """Mixup over whole clips. Geometry is mixed with the pixels it describes.

    ``extras`` (landmark tensors) is mixed in place with the same lambda and
    permutation, for the same reason geometry is.
    """
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(crops.size(0), device=crops.device)
    crops = lam * crops + (1 - lam) * crops[perm]
    geometry = lam * geometry + (1 - lam) * geometry[perm]
    if extras:
        for key, value in extras.items():
            value = value.float()
            extras[key] = lam * value + (1 - lam) * value[perm]
    return crops, geometry, _soft_targets(targets, perm, n_classes, lam)


def cutmix_batch(crops, geometry, targets, n_classes, alpha):
    """Paste a spatial patch of one clip into another; mix labels by area.

    Complements mixup rather than duplicating it. Mixup blends two clips
    everywhere, so every pixel stays a plausible-looking average; cutmix leaves
    both sources intact and forces the model to classify from a *partial* view,
    which is closer to the failure mode that matters here -- a hand half out of
    frame or occluded by the other hand.

    The box is shared across frames and streams. A box that moved per frame would
    give the temporal stage a moving edge to track, and one applied to a single
    stream would be indistinguishable from that stream simply being cropped
    differently.

    Geometry is mixed by the same ratio as the labels rather than cut: the box
    coordinates describe where crops came from in the source frame, and there is
    no coherent way to splice two of those, so the smooth interpolation mixup
    already uses is the honest fallback.
    """
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(crops.size(0), device=crops.device)
    height, width = crops.shape[-2:]
    box_h, box_w = int(height * np.sqrt(1 - lam)), int(width * np.sqrt(1 - lam))
    if box_h < 1 or box_w < 1:
        return crops, geometry, _soft_targets(targets, perm, n_classes, 1.0)
    top = np.random.randint(0, height - box_h + 1)
    left = np.random.randint(0, width - box_w + 1)
    crops = crops.clone()
    crops[..., top : top + box_h, left : left + box_w] = crops[perm][
        ..., top : top + box_h, left : left + box_w
    ]
    # lam is recomputed from the box actually pasted, not the sampled value:
    # integer rounding of the box size shifts the true area by a few percent, and
    # a label that disagrees with the pixels is a slow, silent label-noise leak.
    lam = 1.0 - (box_h * box_w) / (height * width)
    geometry = lam * geometry + (1 - lam) * geometry[perm]
    return crops, geometry, _soft_targets(targets, perm, n_classes, lam)


class ModelEma:
    """Exponential moving average of weights, with a warmup on the decay.

    INCLUDE-50 has ~620 training clips, so an epoch is only ~9 steps and a whole
    run is ~1k updates. A fixed decay of 0.999 has a ~1000-step time constant, so
    the average never escapes its random initialisation -- it sat at chance for an
    entire run before this warmup was added. Ramping the decay in as
    ``(1+step)/(10+step)`` lets the EMA track closely early and smooth later.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.decay = decay
        self.steps = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.steps += 1
        decay = min(self.decay, (1 + self.steps) / (10 + self.steps))
        for ema_value, value in zip(self.module.state_dict().values(), model.state_dict().values()):
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(decay).add_(value.detach(), alpha=1 - decay)
            else:
                ema_value.copy_(value)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    correct = top5 = total = 0
    per_class_correct: dict[int, list[int]] = {}

    for batch in loader:
        crops = batch["crops"].to(device, non_blocking=True)
        detected = batch["detected"].to(device, non_blocking=True)
        geometry = batch["geometry"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(crops, detected, geometry, **model_inputs(batch, device))

        ranked = logits.float().topk(min(5, logits.size(1)), dim=1).indices
        hits = ranked[:, 0] == labels
        correct += hits.sum().item()
        top5 += (ranked == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.numel()

        for label, hit in zip(labels.tolist(), hits.tolist()):
            entry = per_class_correct.setdefault(label, [0, 0])
            entry[0] += int(hit)
            entry[1] += 1

    balanced = float(np.mean([c / n for c, n in per_class_correct.values()])) if per_class_correct else 0.0
    return {
        "top1": correct / max(1, total),
        "top5": top5 / max(1, total),
        "balanced": balanced,
        "n": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ISL-ViT-Tiny")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split-file", type=str, default=None, help="override config split_file")
    parser.add_argument("--tag", type=str, default=None, help="override run name")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--overfit-batch", action="store_true", help="sanity check: fit one batch")
    parser.add_argument("--no-pretrained", action="store_true", help="ablation: random init")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override config seed; repeat a run under several seeds to measure the "
        "noise floor rather than guessing it",
    )
    parser.add_argument(
        "--track-test",
        action="store_true",
        help="DIAGNOSTIC: log test accuracy each --track-every epochs so the shape of "
        "the generalisation curve is visible (e.g. to check for a delayed grokking "
        "transition). Never used for checkpoint selection; see the note at the call site",
    )
    parser.add_argument("--track-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true",
                        help="continue from runs/<tag>/resume.pt if it exists")
    parser.add_argument("--resume-every", type=int, default=25,
                        help="epochs between resume checkpoints; 0 disables them")
    # Augmentation and input-geometry overrides, so a sweep is one script rather
    # than a config file per cell. All default to None meaning "leave the config
    # alone", never to a value -- a flag that silently substitutes its own default
    # would change every run that does not mention it.
    parser.add_argument("--mixup", type=float, default=None, help="Beta alpha for whole-clip blending")
    parser.add_argument("--cutmix", type=float, default=None, help="Beta alpha for patch pasting; 0 disables")
    parser.add_argument("--speed-jitter", type=float, default=None,
                        help="max fraction of the clip's duration to drop when sampling frames")
    parser.add_argument("--random-erasing", type=float, default=None, help="per-stream probability of an erased box")
    parser.add_argument("--mixup-prob", type=float, default=None,
                        help="share of batches that get blended at all; DeiT uses 1.0, this project used 0.5")
    parser.add_argument("--n-frames", type=int, default=None, help="capacity ablation: frames sampled per clip")
    parser.add_argument("--landmarks", action="store_true",
                        help="add the hand/pose landmark stream (needs islvit.data.landmarks)")
    parser.add_argument("--lm-interp", action="store_true", help="fill short gaps in hand landmarks")
    parser.add_argument("--lm-velocity", action="store_true", help="add per-hand frame-to-frame motion")
    parser.add_argument("--lm-pair", action="store_true", help="add the wrist-to-wrist vector and distance")
    parser.add_argument("--lm-wrist-vel", action="store_true", help="add wrist motion in the body frame")
    parser.add_argument("--lm-aug", type=float, default=None, help="landmark augmentation strength")
    parser.add_argument("--val-every", type=int, default=1,
                        help="validate every N epochs (and the last); selection only sees those")
    parser.add_argument("--img-size", type=int, default=None, help="capacity ablation: input resolution per crop")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="override the config value. Added after a grokking probe silently ran both "
        "of its arms at the config default: the shell loop echoed the intended value but "
        "there was no flag to pass it to, so the two arms were byte-identical runs",
    )
    parser.add_argument(
        "--backbone-lr-scale",
        type=float,
        default=None,
        help="LR multiplier for the spatial encoder (config default 0.1). Set 0.0 to "
        "freeze it: tests whether finetuning on 2,845 INCLUDE clips is overwriting the "
        "corpus-general features that pretraining installed, which would explain why "
        "cross-corpus accuracy sits at chance regardless of pretraining corpus",
    )
    parser.add_argument(
        "--resolution-jitter",
        type=float,
        default=None,
        help="probability of degrading a clip to a random lower resolution and back; "
        "closes the 2.1x sharpness gap between INCLUDE and CISLR crops",
    )
    parser.add_argument(
        "--select",
        choices=("best-val", "last"),
        default="best-val",
        help="checkpoint selection; use 'last' when val has been folded into train "
        "and no longer provides an honest signal",
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="path to a pretrained.pt from islvit.pretrain; overrides ImageNet init",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.split_file:
        config["split_file"] = args.split_file
    if args.epochs:
        config["epochs"] = args.epochs
    if args.no_pretrained:
        config["pretrained"] = False
    if args.seed is not None:
        config["seed"] = args.seed
    if args.resolution_jitter is not None:
        config["resolution_jitter"] = args.resolution_jitter
    for name in ("mixup", "mixup_prob", "cutmix", "speed_jitter", "random_erasing", "n_frames", "img_size"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    if args.landmarks:
        config["landmarks"] = True
    if args.lm_interp:
        config["lm_interp"] = True
    if args.lm_velocity:
        config["lm_velocity"] = True
    if args.lm_pair:
        config["lm_pair"] = True
    if args.lm_wrist_vel:
        config["lm_wrist_vel"] = True
    if args.lm_aug is not None:
        config["lm_aug"] = args.lm_aug
    if args.backbone_lr_scale is not None:
        config["backbone_lr_scale"] = args.backbone_lr_scale
    if args.weight_decay is not None:
        config["weight_decay"] = args.weight_decay

    split_name = Path(config["split_file"]).stem
    tag = args.tag or f"{split_name}{'_scratch' if not config.get('pretrained', True) else ''}"
    output_dir = Path("runs") / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_set, val_set, test_set = build_datasets(
        config["split_file"],
        n_frames=config["n_frames"],
        img_size=config["img_size"],
        flip_prob=config.get("flip_prob", 0.5),
        color_jitter=config.get("color_jitter", 0.2),
        stream_dropout=config.get("stream_dropout", 0.1),
        grayscale_prob=config.get("grayscale_prob", 0.0),
        crop_scale=config.get("crop_scale", 0.8),
        resolution_jitter=config.get("resolution_jitter", 0.0),
        speed_jitter=config.get("speed_jitter", 0.0),
        random_erasing=config.get("random_erasing", 0.0),
        landmarks=config.get("landmarks", False),
        lm_interp=config.get("lm_interp", False),
        lm_aug=config.get("lm_aug", 0.0),
    )
    print(f"[{tag}] classes={train_set.n_classes} train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    loader_kwargs = dict(
        num_workers=config.get("num_workers", 4),
        pin_memory=device == "cuda",
        persistent_workers=config.get("num_workers", 4) > 0,
    )
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"] * 2, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"] * 2, shuffle=False, **loader_kwargs)

    model = ISLViT(
        n_classes=train_set.n_classes,
        n_frames=config["n_frames"],
        img_size=config["img_size"],
        patch_size=config.get("patch_size", 16),
        dim=config.get("dim", 192),
        spatial_depth=config.get("spatial_depth", 4),
        temporal_depth=config.get("temporal_depth", 4),
        heads=config.get("heads", 3),
        drop_path=config.get("drop_path", 0.1),
        landmarks=config.get("landmarks", False),
        lm_velocity=config.get("lm_velocity", False),
        lm_pair=config.get("lm_pair", False),
        lm_wrist_vel=config.get("lm_wrist_vel", False),
    )
    if args.init_from:
        # Self-supervised weights supersede ImageNet: the checkpoint was itself
        # initialised from DeiT-Tiny and then trained on signing crops, so loading
        # ImageNet over it would throw that away. The classifier head is not in the
        # checkpoint and stays freshly initialised.
        checkpoint = torch.load(args.init_from, map_location="cpu", weights_only=False)
        state = {key: value for key, value in checkpoint["backbone"].items() if not key.startswith("head.")}
        state = resize_position_embeddings(state, model)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  init from {args.init_from} (epoch {checkpoint.get('epoch', '?')}): "
              f"{len(state)} tensors, {len(missing)} missing, {len(unexpected)} unexpected")
        if unexpected:
            raise SystemExit(f"checkpoint has tensors this model lacks: {unexpected[:5]}")
        config["pretrained"] = f"ssl:{Path(args.init_from).parent.name}"
    elif config.get("pretrained", True):
        load_deit_tiny_weights(model)
    model = model.to(device)

    stats = count_parameters(model)
    print(f"  params total={stats['total_M']:.2f}M spatial={stats['spatial_M']:.2f}M temporal={stats['temporal_M']:.2f}M")

    optimizer = torch.optim.AdamW(
        build_param_groups(model, config["lr"], config["weight_decay"], config.get("backbone_lr_scale", 0.1)),
        betas=(0.9, 0.999),
    )
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.1))
    soft_criterion = nn.CrossEntropyLoss()
    ema = ModelEma(model, config.get("ema_decay", 0.999)) if config.get("ema", True) else None

    if args.overfit_batch:
        # If the model cannot drive one batch to ~100%, nothing downstream is worth
        # running. Fastest possible check that the wiring is sound.
        batch = next(iter(train_loader))
        crops = batch["crops"].to(device)
        detected = batch["detected"].to(device)
        geometry = batch["geometry"].to(device)
        labels = batch["label"].to(device)
        extras = model_inputs(batch, device)
        print(f"\nOverfitting one batch of {labels.numel()} for 200 steps...")
        model.train()
        for step in range(200):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(crops, detected, geometry, **extras), labels)
            loss.backward()
            optimizer.step()
            if step % 40 == 0 or step == 199:
                accuracy = (model(crops, detected, geometry, **extras).argmax(1) == labels).float().mean().item()
                print(f"  step {step:3d} loss {loss.item():.4f} acc {accuracy:.1%}")
        return

    total_steps = config["epochs"] * len(train_loader)
    warmup_steps = config.get("warmup_epochs", 5) * len(train_loader)

    best = {"val_top1": -1.0, "epoch": -1}
    step = 0
    start_epoch = 0

    # Resume state, distinct from best.pt. best.pt holds only the weights to
    # evaluate -- possibly the EMA copy -- which is all inference needs and not
    # nearly enough to continue training: the optimiser moments, the EMA shadow
    # and the LR schedule position are all absent. Two power cuts have now killed
    # multi-hour runs mid-flight, the second at epoch 1300 of 2000.
    resume_path = output_dir / "resume.pt"
    if args.resume and resume_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        # A resume across a changed recipe would silently blend two configurations
        # into one run and report it as the second.
        fingerprint = {k: config.get(k) for k in ("split_file", "n_frames", "img_size", "epochs", "seed")}
        if state["fingerprint"] != fingerprint:
            raise SystemExit(
                f"resume.pt was written for {state['fingerprint']}, this run is {fingerprint}; "
                f"delete runs/{tag}/resume.pt to start over"
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if ema is not None and state.get("ema") is not None:
            ema.module.load_state_dict(state["ema"])
            ema.steps = state["ema_steps"]
        step, best, start_epoch = state["step"], state["best"], state["epoch"] + 1
        print(f"  resumed from epoch {state['epoch']}, continuing at {start_epoch}/{config['epochs']}")

    history_path = output_dir / "history.csv"
    # Append on resume so the loss curve survives the interruption; the header is
    # written only when starting fresh.
    if start_epoch and history_path.exists():
        # Epochs logged after the last resume checkpoint were really trained, but
        # they are about to be trained again from the checkpoint's weights. Leaving
        # them in produces duplicate epoch numbers, and the curve analyses in this
        # project read history.csv positionally.
        with history_path.open(encoding="utf-8") as handle:
            kept = [row for row in csv.DictReader(handle) if int(row["epoch"]) < start_epoch]
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            out = csv.DictWriter(handle, fieldnames=list(kept[0].keys()) if kept else [])
            if kept:
                out.writeheader()
                out.writerows(kept)
    history = history_path.open("a" if start_epoch else "w", newline="", encoding="utf-8")
    writer = csv.writer(history)
    if not start_epoch:
        writer.writerow(["epoch", "lr", "train_loss", "val_top1", "val_top5", "val_balanced",
                         "ema_top1", "seconds", "tracked_test_top1"])
    mixup_alpha = config.get("mixup", 0.0)
    cutmix_alpha = config.get("cutmix", 0.0)

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        started = time.time()
        running_loss = 0.0
        seen = 0

        for batch in train_loader:
            scale = cosine_schedule(step, total_steps, warmup_steps)
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base_lr * scale

            crops = batch["crops"].to(device, non_blocking=True)
            detected = batch["detected"].to(device, non_blocking=True)
            geometry = batch["geometry"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            extras = model_inputs(batch, device)

            # One or the other per batch, never both: stacking them halves the
            # share of clips seen intact, and the DeiT recipe this follows picks
            # between them rather than composing them.
            use_mixup = mixup_alpha > 0 and np.random.rand() < config.get("mixup_prob", 0.5)
            use_cutmix = use_mixup and cutmix_alpha > 0 and np.random.rand() < config.get("cutmix_share", 0.5)
            if use_mixup:
                if use_cutmix and extras:
                    raise SystemExit("cutmix pastes pixel patches; it has no landmark equivalent")
                if use_cutmix:
                    crops, geometry, soft_targets = cutmix_batch(
                        crops, geometry, labels, train_set.n_classes, cutmix_alpha)
                else:
                    crops, geometry, soft_targets = mixup_batch(
                        crops, geometry, labels, train_set.n_classes, mixup_alpha, extras=extras)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(crops, detected, geometry, **extras)
                loss = soft_criterion(logits, soft_targets) if use_mixup else criterion(logits, labels)

            loss.backward()
            if config.get("grad_clip", 0):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            if ema is not None:
                ema.update(model)

            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
            step += 1

        train_loss = running_loss / max(1, seen)
        # Validation is the most expensive part of an epoch on a real val split;
        # --val-every thins it, and selection below only ever sees scored epochs.
        validated = epoch % args.val_every == 0 or epoch == config["epochs"] - 1
        if validated:
            val = evaluate(model, val_loader, device)
            ema_val = evaluate(ema.module, val_loader, device) if ema is not None else {"top1": 0.0}
        else:
            val = {"top1": float("nan"), "top5": float("nan"), "balanced": float("nan")}
            ema_val = {"top1": float("nan")}

        # DIAGNOSTIC ONLY. Logging test accuracy during training is legitimate for
        # observing the shape of the generalisation curve (e.g. checking for a
        # delayed grokking transition) but must never influence a decision --
        # checkpoint selection still uses --select, which never reads this value.
        # Quoting a number chosen by watching this curve would be test-set
        # peeking, and it is logged in its own column to keep that distinction
        # visible in the history file.
        tracked = (
            evaluate(model, test_loader, device)["top1"]
            if args.track_test and (epoch % args.track_every == 0 or epoch == config["epochs"] - 1)
            else float("nan")
        )
        elapsed = time.time() - started

        writer.writerow(
            [epoch, f"{optimizer.param_groups[0]['lr']:.2e}", f"{train_loss:.4f}",
             f"{val['top1']:.4f}", f"{val['top5']:.4f}", f"{val['balanced']:.4f}",
             f"{ema_val['top1']:.4f}", f"{elapsed:.1f}", f"{tracked:.4f}"]
        )
        history.flush()

        # With val folded into train there is no honest selection signal left, so
        # take the final epoch instead. The cosine schedule ends at zero learning
        # rate, which is what makes the last epoch a defensible choice rather than
        # an arbitrary one -- and EMA weights are preferred there because the raw
        # weights still carry the last batch's noise.
        #
        # Overwrite every epoch rather than only on the last one: saving solely at
        # the end means any interruption loses the whole run, which is exactly what
        # a machine restart did after 14 epochs. Rewriting a 15 MB checkpoint each
        # epoch is far cheaper than that.
        take_last = args.select == "last"
        score = max(val["top1"], ema_val["top1"])
        take = True if take_last else (validated and score > best["val_top1"])

        marker = ""
        if take:
            best = {
                "val_top1": score,
                "epoch": epoch,
                "use_ema": ema is not None if take_last else ema_val["top1"] > val["top1"],
            }
            torch.save(
                {
                    "model": (ema.module if best["use_ema"] and ema else model).state_dict(),
                    "config": config,
                    "classes": train_set.classes,
                    "epoch": epoch,
                },
                output_dir / "best.pt",
            )
            # Under "last" every epoch is saved, so a star would mark every line
            # and tell the reader nothing.
            marker = "" if take_last else " *"

        # Written periodically rather than every epoch: this carries the optimiser
        # moments and the EMA shadow as well as the weights, so it is several times
        # the size of best.pt. At the default interval the worst case is a couple of
        # minutes of lost training against a few seconds of I/O.
        if args.resume_every and (epoch % args.resume_every == 0 or epoch == config["epochs"] - 1):
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema.module.state_dict() if ema is not None else None,
                    "ema_steps": ema.steps if ema is not None else 0,
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "step": step,
                    "best": best,
                    "fingerprint": {
                        k: config.get(k) for k in ("split_file", "n_frames", "img_size", "epochs", "seed")
                    },
                },
                output_dir / "resume.pt",
            )

        if epoch % config.get("log_every", 1) == 0 or epoch == config["epochs"] - 1:
            print(
                f"  ep {epoch:3d}  loss {train_loss:.3f}  val {val['top1']:.1%}"
                f" (top5 {val['top5']:.1%}, bal {val['balanced']:.1%})  ema {ema_val['top1']:.1%}"
                f"  {elapsed:.0f}s{marker}",
                flush=True,
            )

    history.close()

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test = evaluate(model, test_loader, device)
    val_final = evaluate(model, val_loader, device)

    summary = {
        "tag": tag,
        "split_file": config["split_file"],
        # Which crop cache produced this number -- 80px and 128px sources give
        # measurably different results at the same img_size, so comparing runs
        # without knowing the cache is meaningless.
        "cache": os.environ.get("ISLVIT_CACHE", "cache"),
        "img_size": config["img_size"],
        # Both input axes are ablated, so a summary that records neither cannot be
        # placed in the ablation table after the fact.
        "n_frames": config["n_frames"],
        "pretrained": config.get("pretrained", True),
        # Recorded so the report can recognise seed replicates of one configuration
        # and average them, instead of ranking them against each other and quoting
        # whichever seed came out luckiest.
        "seed": config.get("seed", 0),
        # The recipe knobs that vary between runs. Without these a summary cannot
        # be attributed to a configuration after the fact: comparing the long
        # training runs meant reading the shell scripts that launched them,
        # because epochs and weight decay were nowhere in the artefact.
        "recipe": {
            key: config.get(key)
            for key in ("epochs", "weight_decay", "mixup", "mixup_prob", "cutmix", "resolution_jitter",
                        "speed_jitter", "random_erasing", "color_jitter", "grayscale_prob",
                        "stream_dropout", "backbone_lr_scale", "lr", "batch_size", "landmarks",
                        "lm_interp", "lm_velocity", "lm_aug", "lm_pair", "lm_wrist_vel")
        },
        "best_epoch": best["epoch"],
        "params_M": round(stats["total_M"], 3),
        "val": {key: round(value, 4) for key, value in val_final.items()},
        "test": {key: round(value, 4) for key, value in test.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # The run finished, so the resume state is dead weight -- several times the size
    # of best.pt, and a stale one would be silently picked up by a later --resume.
    (output_dir / "resume.pt").unlink(missing_ok=True)

    print(f"\n[{tag}] best epoch {best['epoch']}")
    print(f"  val  top1 {val_final['top1']:.1%}  top5 {val_final['top5']:.1%}  balanced {val_final['balanced']:.1%}")
    print(f"  TEST top1 {test['top1']:.1%}  top5 {test['top5']:.1%}  balanced {test['balanced']:.1%}  (n={test['n']})")


if __name__ == "__main__":
    main()
