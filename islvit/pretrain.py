"""Masked-crop pretraining for ISL-ViT-Tiny.

Every capacity lever tested in the supervised ablations was flat or negative,
which says the model is data-limited. The supervised corpus cannot grow -- INCLUDE
is what it is -- but unlabelled signing video can, so this pretrains both encoder
stages on crops without using any labels.

Objective is SimMIM-style rather than MAE-style: masked patches are replaced with a
learned mask token *inside* the encoder instead of being dropped from the sequence.
With only 16 patches per crop, dropping tokens leaves 4 visible patches and a
ragged batch; substituting keeps the tensor rectangular and the encoder identical
to the one that will later be finetuned, which matters more here than MAE's
compute saving.

Two things are specific to this data:

1. **The loss is gated on genuine detection.** A CISLR survey found a median of
   only 40.6% genuinely-detected hand frames per clip, against ~88% on INCLUDE, so
   38-52% of boxes are interpolated -- a stale box showing background the hand has
   already left. Reconstructing those teaches the model to predict background.
   Interpolated and missing crops are excluded from the loss, which turns a
   data-quality problem into simply less data.

2. **Masking is global across time, stream and space.** Masking within a crop only
   would let the spatial encoder solve the task alone and leave Stage B untrained.
   Sampling over all T*S*P tokens means whole crops are sometimes fully masked, and
   the only way to fill them is from neighbouring frames.

Usage::

    python -m islvit.pretrain --cache cache_cislr --epochs 100 --tag cislr_mim
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from islvit.data.crops import SOURCE_DIRECT, SOURCE_ROI
from islvit.data.dataset import prepare_clip
from islvit.models.isl_vit import Block, ISLViT, load_deit_tiny_weights

RUNS_DIR = Path("runs")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class PretrainCrops(Dataset):
    """Every cached clip in a crop cache, unlabelled.

    Deliberately not tied to a split file: pretraining data has no train/test
    structure to respect. When pretraining on INCLUDE itself, pass a split file
    via ``restrict_to`` -- pretraining on test-session clips would leak.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        n_frames: int = 8,
        img_size: int = 64,
        min_genuine: float = 0.25,
        restrict_to: str | Path | None = None,
        crop_scale: float = 0.8,
        color_jitter: float = 0.4,
        grayscale_prob: float = 0.15,
        flip_prob: float = 0.5,
        resolution_jitter: float = 0.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.n_frames = n_frames
        self.img_size = img_size
        self.crop_scale = crop_scale
        self.color_jitter = color_jitter
        self.grayscale_prob = grayscale_prob
        self.flip_prob = flip_prob
        self.resolution_jitter = resolution_jitter

        with (self.cache_dir / "index.csv").open(encoding="utf-8") as handle:
            index_rows = list(csv.DictReader(handle))
        cached = [row for row in index_rows if row["cached"] == "1"]

        allowed: set[str] | None = None
        if restrict_to is not None:
            with Path(restrict_to).open(encoding="utf-8") as handle:
                allowed = {
                    row["video_path"] for row in csv.DictReader(handle) if row["split"] == "train"
                }

        sources = np.load(self.cache_dir / "sources.npy", mmap_mode="r")
        keep: list[int] = []
        rates: list[float] = []
        for row in cached:
            if allowed is not None and row["video_path"] not in allowed:
                continue
            index = int(row["row"])
            hand_sources = np.asarray(sources[index][:, :2])
            genuine = float(((hand_sources == SOURCE_DIRECT) | (hand_sources == SOURCE_ROI)).mean())
            if genuine >= min_genuine:
                keep.append(index)
                rates.append(genuine)

        self.rows = np.array(keep, dtype=np.int64)
        self.genuine_rates = np.array(rates, dtype=np.float32)
        print(
            f"  pretrain set: {len(self.rows)} of {len(cached)} cached clips "
            f"(genuine hand detection >= {min_genuine:.0%}, "
            f"median kept {np.median(self.genuine_rates) if len(rates) else 0:.1%})"
        )

        self._crops: np.ndarray | None = None
        self._sources: np.ndarray | None = None
        self._geometry: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def _cache(self):
        if self._crops is None:
            self._crops = np.load(self.cache_dir / "crops.npy", mmap_mode="r")
            self._sources = np.load(self.cache_dir / "sources.npy", mmap_mode="r")
            self._geometry = np.load(self.cache_dir / "geometry.npy", mmap_mode="r")
        return self._crops, self._sources, self._geometry

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        crops_cache, sources_cache, geometry_cache = self._cache()
        row = self.rows[index]

        clip = np.asarray(crops_cache[row])
        sources = np.asarray(sources_cache[row])
        geometry = np.asarray(geometry_cache[row]).astype(np.float32)
        detected = (sources == SOURCE_DIRECT) | (sources == SOURCE_ROI)

        edges = np.linspace(0, clip.shape[0], self.n_frames + 1)
        picks = edges[:-1] + np.random.rand(self.n_frames) * (edges[1:] - edges[:-1])
        frame_indices = np.clip(picks.astype(int), 0, clip.shape[0] - 1)

        crops, detected, geometry = prepare_clip(
            clip[frame_indices],
            detected[frame_indices],
            geometry[frame_indices],
            img_size=self.img_size,
            train=True,
            crop_scale=self.crop_scale,
            color_jitter=self.color_jitter,
            grayscale_prob=self.grayscale_prob,
            flip_prob=self.flip_prob,
            # No stream dropout: blanking a stream would then be reconstructed as
            # black, which teaches nothing. Masking already provides that pressure.
            stream_dropout=0.0,
            resolution_jitter=self.resolution_jitter,
        )
        return {
            "crops": torch.from_numpy(crops),
            "detected": torch.from_numpy(detected),
            "geometry": torch.from_numpy(geometry),
        }


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def patchify(images: torch.Tensor, patch: int) -> torch.Tensor:
    """(N,3,H,W) -> (N, P, patch*patch*3), ordered row-major to match patch_embed."""
    n, channels, height, _ = images.shape
    grid = height // patch
    x = images.reshape(n, channels, grid, patch, grid, patch)
    x = x.permute(0, 2, 4, 3, 5, 1)
    return x.reshape(n, grid * grid, patch * patch * channels)


class MaskedPretrainer(nn.Module):
    """Wraps an ISLViT and adds a shallow decoder used only during pretraining."""

    def __init__(
        self,
        backbone: ISLViT,
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        decoder_dim: int = 128,
        decoder_depth: int = 2,
        decoder_heads: int = 4,
        norm_target: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.norm_target = norm_target

        dim = backbone.spatial.dim
        n_patches = backbone.spatial.n_patches
        self.n_patches = n_patches

        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.decoder_embed = nn.Linear(dim, decoder_dim)
        self.decoder_context = nn.Linear(dim, decoder_dim)
        self.decoder_pos = nn.Parameter(torch.zeros(1, n_patches, decoder_dim))
        nn.init.trunc_normal_(self.decoder_pos, std=0.02)
        self.decoder_blocks = nn.ModuleList(
            [Block(decoder_dim, decoder_heads) for _ in range(decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim, eps=1e-6)
        self.decoder_head = nn.Linear(decoder_dim, patch_size * patch_size * 3)

    def forward(
        self, crops: torch.Tensor, detected: torch.Tensor, geometry: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        batch, frames, streams = crops.shape[:3]
        spatial = self.backbone.spatial
        flat = crops.flatten(0, 2)                          # (N,3,H,W)
        n = flat.shape[0]

        embedded = spatial.embed(flat)                      # (N, 1+P, D), pos already added
        mask = torch.rand(n, self.n_patches, device=flat.device) < self.mask_ratio
        # Re-add the positional term to the mask token: embed() folded pos into the
        # patch embeddings, so a bare substitution would strip position from exactly
        # the tokens whose position the decoder needs most.
        masked_value = self.mask_token + spatial.pos_embed[:, 1:]
        patches = torch.where(mask.unsqueeze(-1), masked_value.expand(n, -1, -1), embedded[:, 1:])
        encoded = spatial.encode(torch.cat([embedded[:, :1], patches], dim=1))

        crop_embeddings = encoded[:, 0].reshape(batch, frames, streams, -1)
        context = self.backbone.temporal(crop_embeddings, detected, geometry)  # (B, 1+T*S, D)

        tokens = self.decoder_embed(encoded[:, 1:]) + self.decoder_pos
        tokens = tokens + self.decoder_context(context[:, 1:]).reshape(n, 1, -1)
        for block in self.decoder_blocks:
            tokens = block(tokens)
        prediction = self.decoder_head(self.decoder_norm(tokens))   # (N, P, patch^2*3)

        target = patchify(flat, self.patch_size)
        if self.norm_target:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6).sqrt()

        # Only score patches that are both masked and come from a genuinely
        # detected crop. Everything else is a stale or blank box.
        weight = mask & detected.reshape(n, 1).expand(-1, self.n_patches)
        loss_per_patch = (prediction - target).pow(2).mean(dim=-1)
        scored = weight.sum()
        if scored == 0:
            return prediction.sum() * 0.0, 0.0
        loss = (loss_per_patch * weight).sum() / scored
        return loss, float(scored) / weight.numel()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Masked-crop pretraining")
    parser.add_argument("--cache", type=str, default="cache_cislr")
    parser.add_argument("--restrict-to", type=str, default=None,
                        help="split file; keeps only its train rows (use when pretraining on INCLUDE)")
    parser.add_argument("--tag", type=str, default="mim")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--min-genuine", type=float, default=0.25)
    parser.add_argument(
        "--resolution-jitter", type=float, default=0.0,
        help="degrade a random fraction of clips to a lower resolution and back; "
        "applied on top of CISLR's native ~78px upscale so the encoder learns "
        "features robust to a *range* of sharpness rather than one fixed blur level",
    )
    parser.add_argument("--n-frames", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="start Stage A from ImageNet (default; --no-pretrained to disable)")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = PretrainCrops(
        args.cache,
        n_frames=args.n_frames,
        img_size=args.img_size,
        min_genuine=args.min_genuine,
        restrict_to=args.restrict_to,
        resolution_jitter=args.resolution_jitter,
    )
    if len(dataset) == 0:
        raise SystemExit("No clips passed the genuine-detection filter; lower --min-genuine.")
    if len(dataset) < args.batch_size:
        # drop_last=True on the DataLoader means a dataset smaller than one batch
        # yields zero batches per epoch, so `learning_rate` -- only ever assigned
        # inside the batch loop -- stays unbound and crashes at the first CSV
        # write with a confusing UnboundLocalError instead of the real problem:
        # not enough data. Surfaced directly instead of silently "training" for
        # zero steps -- this happened for real when a crop extraction crash cut
        # 18,000 clips down to 50 and pretraining ran anyway.
        raise SystemExit(
            f"only {len(dataset)} clips available, smaller than --batch-size {args.batch_size} "
            f"-- with drop_last=True this trains on nothing. Fix the upstream cache or lower --batch-size."
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=args.num_workers > 0,
    )

    # n_classes is irrelevant here -- the head is never used -- but the backbone
    # must be built exactly as it will be finetuned so the weights transfer.
    backbone = ISLViT(
        n_classes=1,
        n_frames=args.n_frames,
        img_size=args.img_size,
        drop_path=0.0,     # pretraining is not overfitting; keep the signal clean
    )
    if args.pretrained:
        load_deit_tiny_weights(backbone)
    model = MaskedPretrainer(backbone, mask_ratio=args.mask_ratio).to(device)

    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim <= 1 or name.endswith("token") else decay).append(parameter)
    optimiser = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    steps_per_epoch = max(1, len(loader))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return args.lr * step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    output_dir = RUNS_DIR / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    history = (output_dir / "history.csv").open("w", newline="", encoding="utf-8")
    writer = csv.writer(history)
    writer.writerow(["epoch", "lr", "loss", "scored_fraction", "seconds"])

    print(f"[{args.tag}] {len(dataset)} clips, {steps_per_epoch} steps/epoch, device={device}")
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        running, scored_running, seen = 0.0, 0.0, 0
        for batch in loader:
            learning_rate = lr_at(step)
            for group in optimiser.param_groups:
                group["lr"] = learning_rate

            crops = batch["crops"].to(device, non_blocking=True)
            detected = batch["detected"].to(device, non_blocking=True)
            geometry = batch["geometry"].to(device, non_blocking=True)

            with torch.amp.autocast(device, enabled=(device == "cuda")):
                loss, scored = model(crops, detected, geometry)

            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()

            running += float(loss) * crops.shape[0]
            scored_running += scored * crops.shape[0]
            seen += crops.shape[0]
            step += 1

        elapsed = time.time() - started
        mean_loss = running / max(1, seen)
        mean_scored = scored_running / max(1, seen)
        writer.writerow([epoch, f"{learning_rate:.2e}", f"{mean_loss:.4f}",
                         f"{mean_scored:.4f}", f"{elapsed:.1f}"])
        history.flush()
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{args.epochs}  loss {mean_loss:.4f}  "
                  f"scored {mean_scored:.1%}  {elapsed:.1f}s")

        torch.save(
            {"backbone": backbone.state_dict(), "epoch": epoch, "args": vars(args)},
            output_dir / "pretrained.pt",
        )

    history.close()
    (output_dir / "summary.json").write_text(
        json.dumps({"tag": args.tag, "cache": args.cache, "clips": len(dataset),
                    "epochs": args.epochs, "final_loss": mean_loss,
                    "mask_ratio": args.mask_ratio, "min_genuine": args.min_genuine}, indent=2),
        encoding="utf-8",
    )
    print(f"[{args.tag}] done. backbone -> {output_dir / 'pretrained.pt'}")


if __name__ == "__main__":
    main()
