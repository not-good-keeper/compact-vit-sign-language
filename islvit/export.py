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
