"""Quantise a trained checkpoint to INT8 and measure what it actually costs.

The size figures quoted in the report so far are arithmetic -- parameter count
times one byte -- and that is not what a quantised model weighs. Two things the
arithmetic misses, both in the wrong direction:

* **Not everything quantises.** ``quantize_dynamic`` converts ``nn.Linear`` only.
  Here that is 95.7 % of parameters, but the remaining 4.3 % -- the patch-embedding
  Conv2d, the LayerNorms, and the position/time embeddings -- stay FP32 at four
  bytes each. The Conv2d is 148 k parameters and does *not* shrink when depth is
  reduced, so it becomes a larger share of a smaller model.
* **Quantised weights carry metadata.** Per-channel scales and zero-points are
  stored alongside, plus the packed-tensor container itself.

So a 2 M-parameter model is not a 2 MB file, and the gap matters when the target
is "sub-2 MB". This measures rather than assumes.

Usage::

    python -m islvit.export --run runs/f16_clean_s0          # size + accuracy
    python -m islvit.export --survey                          # size across configs
"""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from islvit.models.isl_vit import ISLViT, count_parameters


def file_size_mb(model: nn.Module) -> float:
    """Bytes the model's state_dict actually occupies on disk."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        path = Path(handle.name)
    try:
        torch.save(model.state_dict(), path)
        return path.stat().st_size / 2**20
    finally:
        path.unlink(missing_ok=True)


def quantise(model: nn.Module) -> nn.Module:
    """Dynamic INT8 over Linear layers, on a CPU copy."""
    model = copy.deepcopy(model).cpu().eval()
    return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def pack_int8(model: nn.Module) -> dict:
    """Weight-only INT8 storage format, covering what quantize_dynamic leaves behind.

    ``quantize_dynamic`` converts ``nn.Linear`` and nothing else. That is 95.7 % of
    parameters here, but the leftovers are stored FP32 at four bytes each and do not
    shrink with depth -- the patch-embedding Conv2d alone is 148 k parameters, so it
    becomes a *larger* share of a smaller model and is exactly what pushes a
    2 M-parameter configuration past a 2 MB budget.

    This packs the remainder too: Conv2d weights get per-output-channel INT8 scales,
    and every other tensor (LayerNorm, biases, position and time embeddings) drops to
    FP16. Those are storage decisions, not runtime ones -- the conv is dequantised at
    load. Per-channel rather than per-tensor for the conv because its output channels
    are independent filters whose weight ranges differ by more than an order of
    magnitude; one shared scale would crush the small ones to zero.
    """
    packed: dict[str, object] = {}
    for name, tensor in model.state_dict().items():
        # A quantised state_dict is not all tensors: quantize_dynamic stores each
        # Linear's weights inside a _packed_params object and records dtypes as
        # bare torch.dtype values. Both pass through untouched -- they are already
        # INT8, and unpacking them here would undo the quantisation.
        if not isinstance(tensor, torch.Tensor):
            packed[name] = tensor
            continue
        if not torch.is_floating_point(tensor):
            packed[name] = tensor
            continue
        # 4-D weights are the patch-embedding conv: quantise per output channel.
        if tensor.ndim == 4:
            flat = tensor.reshape(tensor.shape[0], -1)
            scale = flat.abs().amax(dim=1).clamp(min=1e-8) / 127.0
            packed[name] = {
                "int8": (flat / scale[:, None]).round().clamp(-127, 127).to(torch.int8),
                "scale": scale.to(torch.float16),
                "shape": tuple(tensor.shape),
            }
        else:
            packed[name] = tensor.to(torch.float16)
    return packed


def save_int4(packed: dict, path: Path, meta: dict | None = None) -> float:
    """Write a packed INT4 state compactly, and return the file size in MiB.

    ``torch.save`` on the packed dict costs roughly 240 bytes per tensor in pickle
    keys and headers. At ~200 tensors that is ~48 KB -- which is what put a model
    whose contents measure 2004 KB into a 2052 KB file, over a 2 MB budget by
    packaging alone.

    Everything is concatenated into three flat buffers (nibbles, scales, FP16
    leftovers) plus one small manifest describing where each tensor lives. The
    stored bits are identical, so this is lossless: it changes the container, not
    the model.
    """
    nibbles, scales, plain, manifest = [], [], [], []
    for name, value in packed.items():
        if isinstance(value, dict):
            manifest.append(("q", name, tuple(value["shape"]),
                             len(nibbles), value["nibbles"].shape,
                             len(scales), value["scale"].shape))
            nibbles.append(value["nibbles"].reshape(-1))
            scales.append(value["scale"].reshape(-1))
        else:
            manifest.append(("f", name, tuple(value.shape), len(plain), None, None, None))
            plain.append(value.reshape(-1))
    blob = {
        # Config and word list, so the file is a complete deliverable: without
        # them a reader cannot rebuild the architecture or name its outputs.
        "meta": meta or {},
        "manifest": manifest,
        "nibbles": torch.cat(nibbles) if nibbles else torch.empty(0, dtype=torch.uint8),
        "scales": torch.cat(scales) if scales else torch.empty(0, dtype=torch.float16),
        "plain": torch.cat([v.to(torch.float16).reshape(-1) for v in plain])
                 if plain else torch.empty(0, dtype=torch.float16),
        "splits": {
            "nibbles": [int(t.numel()) for t in nibbles],
            "scales": [int(t.numel()) for t in scales],
            "plain": [int(t.numel()) for t in plain],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path, _use_new_zipfile_serialization=True)
    return path.stat().st_size / 2**20


def load_int4(path: Path, group: int = 128) -> dict:
    """Read a file written by ``save_int4`` back into a usable state_dict.

    A write-only container is not a deliverable, so this is the other half of the
    format: it rebuilds full-precision tensors from the 4-bit codes and scales and
    returns something ``load_state_dict`` accepts directly.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    offsets = {kind: 0 for kind in ("nibbles", "scales", "plain")}
    cursors = {kind: [] for kind in ("nibbles", "scales", "plain")}
    for kind in cursors:
        start = 0
        for count in blob["splits"][kind]:
            cursors[kind].append((start, start + count))
            start += count

    state = {}
    for kind, name, shape, index, nib_shape, scale_index, scale_shape in blob["manifest"]:
        if kind == "f":
            lo, hi = cursors["plain"][index]
            state[name] = blob["plain"][lo:hi].reshape(shape).float()
            continue
        lo, hi = cursors["nibbles"][index]
        nibbles = blob["nibbles"][lo:hi].reshape(tuple(nib_shape))
        lo, hi = cursors["scales"][scale_index]
        scale = blob["scales"][lo:hi].reshape(tuple(scale_shape)).float()
        # Unpack two signed 4-bit codes per byte; values 8..15 are negative.
        low = (nibbles & 0x0F).to(torch.int16)
        high = ((nibbles >> 4) & 0x0F).to(torch.int16)
        low = torch.where(low > 7, low - 16, low)
        high = torch.where(high > 7, high - 16, high)
        codes = torch.empty(nibbles.shape[0], nibbles.shape[1] * 2, dtype=torch.float32)
        codes[:, 0::2], codes[:, 1::2] = low.float(), high.float()
        numel = int(torch.tensor(shape).prod())
        state[name] = int4_dequantise(codes, scale, tuple(shape), numel)
    return state



def load_release(path: Path, device: str = "cpu"):
    """(model, config, classes) from a file written by ``save_int4`` with meta.

    The weights are rebuilt from the stored 4-bit codes and FP16 scales, so the
    model evaluated is exactly the model in the file.
    """
    from islvit.models.isl_vit import ISLViT

    meta = torch.load(path, map_location="cpu", weights_only=False)["meta"]
    config, classes = meta["config"], meta["classes"]
    model = ISLViT(
        n_classes=len(classes), n_frames=config["n_frames"], img_size=config["img_size"],
        patch_size=config.get("patch_size", 16), dim=config.get("dim", 192),
        spatial_depth=config.get("spatial_depth", 4), temporal_depth=config.get("temporal_depth", 4),
        heads=config.get("heads", 3), drop_path=0.0, landmarks=config.get("landmarks", False),
        lm_velocity=config.get("lm_velocity", False), lm_pair=config.get("lm_pair", False),
        lm_wrist_vel=config.get("lm_wrist_vel", False),
    )
    model.load_state_dict(load_int4(path, group=meta.get("group", 128)))
    return model.to(device).eval(), config, classes

def int4_targets(model: nn.Module) -> set[str]:
    """state_dict keys the INT4 path quantises: module weight matrices only.

    Selecting on ``ndim >= 2`` alone is wrong and was a real bug: cls_token,
    stream_embed and time_embed are 3-D, so a dimension test quantised them while
    quantisation-aware training -- which walks modules looking for a parameter named
    "weight" -- left them alone. The two paths then rounded differently, which is the
    one failure that makes a QAT run silently worthless.

    Keeping them full precision is also what the size budget wants: together they are
    under 4 k parameters, so the saving is a few kilobytes, and embeddings are the
    most perturbation-sensitive tensors in the model.
    """
    targets = set()
    for module_name, module in model.named_modules():
        for name, parameter in module.named_parameters(recurse=False):
            if name == "weight" and parameter.ndim >= 2:
                targets.add(f"{module_name}.{name}" if module_name else name)
    return targets


def int4_codes(tensor: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Symmetric group-wise 4-bit codes and scales for one weight tensor.

    Factored out so quantisation-aware training and the exporter round identically.
    If QAT simulates a different scheme than the export applies -- a different group
    size, asymmetric zero points, per-tensor instead of per-group -- it optimises the
    model for rounding that never happens, and the gain evaporates at export time.
    """
    flat = tensor.reshape(-1)
    pad = (-flat.numel()) % group
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.reshape(-1, group)
    scale = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 7.0
    codes = (blocks / scale).round().clamp(-8, 7)
    return codes, scale, pad


def int4_dequantise(codes: torch.Tensor, scale: torch.Tensor, shape, numel: int) -> torch.Tensor:
    """Inverse of int4_codes, back to the original tensor shape."""
    return (codes * scale).reshape(-1)[:numel].reshape(shape)


def pack_int4(model: nn.Module, group: int = 128) -> tuple[dict, dict]:
    """Group-wise INT4 weight packing, plus the dequantised state for scoring.

    INT8 over Linear leaves a 3.76 M model at 3.70 MB, which is nearly twice a 2 MB
    budget. Halving the architecture to reach that budget costs accuracy nobody has
    measured yet; halving the *bit width* costs accuracy that can be measured in
    minutes, so it is worth trying first.

    Quantisation is per group of ``group`` consecutive weights rather than per
    tensor: a single scale across a whole 768x192 matrix is dominated by its largest
    outlier and crushes the rest to a handful of levels. At group 128 the scale
    overhead is 2 bytes per 128 weights -- about 1.6 % -- which is what keeps the
    packed total under 2 MB.

    Returns (packed, dequantised). The second is what the model is scored with, so
    the accuracy reported is the accuracy the packed file would actually deliver.
    """
    packed, dequant = {}, {}
    targets = int4_targets(model)
    for name, tensor in model.state_dict().items():
        # Only module weight matrices: norms, biases, tokens and position embeddings
        # stay FP16. They are a few kilobytes and the most sensitive tensors here.
        if not torch.is_floating_point(tensor) or name not in targets:
            if torch.is_floating_point(tensor):
                packed[name] = tensor.to(torch.float16)
                # Same rule as the scales: reconstruct from what is stored, so the
                # measured model and the saved model are the same model.
                dequant[name] = packed[name].to(tensor.dtype)
            else:
                packed[name] = tensor
                dequant[name] = tensor
            continue
        codes, scale, _ = int4_codes(tensor, group)
        # Round the scale to FP16 *before* reconstructing, because FP16 is what the
        # file stores. Dequantising from the FP32 scale reports an accuracy the
        # saved model cannot deliver -- the same class of mismatch as QAT and the
        # exporter rounding differently.
        scale = scale.to(torch.float16).float()
        byte_codes = codes.to(torch.int8)
        # Two 4-bit codes per byte.
        low, high = byte_codes[:, 0::2] & 0x0F, byte_codes[:, 1::2] & 0x0F
        packed[name] = {
            "nibbles": (low | (high << 4)).to(torch.uint8),
            "scale": scale.to(torch.float16),
            "shape": tuple(tensor.shape),
        }
        dequant[name] = int4_dequantise(
            codes, scale, tensor.shape, tensor.numel()).to(tensor.dtype)
    return packed, dequant


def build(dim: int, sp: int, tp: int, heads: int, n_classes: int, n_frames: int, img_size: int):
    return ISLViT(n_classes=n_classes, n_frames=n_frames, img_size=img_size, patch_size=16,
                  dim=dim, spatial_depth=sp, temporal_depth=tp, heads=heads, drop_path=0.0)


def survey(n_classes: int, n_frames: int, img_size: int) -> None:
    configs = [(192, 4, 4, 3), (192, 3, 3, 3), (192, 2, 3, 3), (192, 2, 2, 3),
               (192, 1, 2, 3), (160, 4, 4, 4), (128, 4, 4, 4), (128, 3, 3, 4), (96, 4, 4, 3)]
    print(f"  {'dim':>4s} {'sp':>3s} {'tp':>3s} {'params':>8s} {'naive':>7s} {'FP32':>8s} "
          f"{'INT8':>8s} {'packed':>8s}  keeps init?")
    for dim, sp, tp, heads in configs:
        model = build(dim, sp, tp, heads, n_classes, n_frames, img_size)
        params = count_parameters(model)["total_M"]
        fp32, int8 = file_size_mb(model), file_size_mb(quantise(model))
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
            ppath = Path(handle.name)
        try:
            torch.save(pack_int8(quantise(model)), ppath)
            packed = ppath.stat().st_size / 2**20
        finally:
            ppath.unlink(missing_ok=True)
        # Width 192 is what DeiT-Tiny and the iSign SSL checkpoint were trained at.
        # Any other width cannot load either one, which costs far more than it saves.
        keeps = "yes" if dim == 192 else "NO (no DeiT/SSL)"
        flag = "  <- sub-2MB packed" if packed < 2.0 else ""
        print(f"  {dim:>4d} {sp:>3d} {tp:>3d} {params:6.2f} M {params:6.2f} MB "
              f"{fp32:7.2f} MB {int8:7.2f} MB {packed:7.2f} MB  {keeps}{flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description="INT8 export and size measurement")
    parser.add_argument("--run", type=str, default=None, help="checkpoint to quantise and score")
    parser.add_argument("--survey", action="store_true", help="size across candidate configs")
    parser.add_argument("--n-classes", type=int, default=50)
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--out", type=str, default=None, help="write the quantised model here")
    args = parser.parse_args()

    if args.survey:
        survey(args.n_classes, args.n_frames, args.img_size)
        return
    if not args.run:
        raise SystemExit("pass --run or --survey")

    from torch.utils.data import DataLoader
    from islvit.data.dataset import build_datasets
    from islvit.eval import load_run

    run_dir = Path(args.run)
    model, config, classes = load_run(run_dir, "cpu")
    _, _, test_set = build_datasets(config["split_file"], n_frames=config["n_frames"],
                                    img_size=config["img_size"])
    loader = DataLoader(test_set, batch_size=32, num_workers=0, shuffle=False)

    def score(net) -> float:
        net.eval()
        hits = total = 0
        with torch.no_grad():
            for batch in loader:
                logits = net(batch["crops"], batch["detected"], batch["geometry"])
                hits += (logits.argmax(1) == batch["label"]).sum().item()
                total += batch["label"].numel()
        return hits / total

    model = model.cpu()
    fp32_mb, fp32_top1 = file_size_mb(model), score(model)
    quantised = quantise(model)
    int8_mb, int8_top1 = file_size_mb(quantised), score(quantised)

    print(f"\n[{run_dir.name}] {len(classes)} classes, {config['n_frames']}f/{config['img_size']}px")
    print(f"  FP32  {fp32_mb:6.2f} MB   top-1 {fp32_top1:.1%}")
    print(f"  INT8  {int8_mb:6.2f} MB   top-1 {int8_top1:.1%}   "
          f"({fp32_mb / int8_mb:.2f}x smaller, {100 * (int8_top1 - fp32_top1):+.1f} pts)")

    if args.out:
        torch.save({"model": quantised.state_dict(), "config": config, "classes": classes},
                   Path(args.out))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
