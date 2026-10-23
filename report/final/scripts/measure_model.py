"""Measure parameter count, compute and model-only CPU latency of ISL-ViT-Tiny.

These are new measurements made for the final report; no earlier committed file
records them. Scope of each number:

* parameters: counted on the instantiated network (pixel-only and landmark).
* FLOPs: torch.utils.flop_counter on one clip, one view (matrix multiplies and
  convolutions; elementwise operations, softmax and LayerNorm are not counted).
  Reported as GMACs = FLOPs / 2.
* latency: forward pass of the released seed-0 model only (no video decoding,
  MediaPipe, cropping or TTA), batch 1, one view, CPU, float32 arithmetic, after
  warm-up; median and interquartile range over repeated runs. The release file
  stores 4-bit codes, but load_release rebuilds float32 weights, so the timed
  arithmetic is float32, not INT4.

Usage (repository root, islvit environment):
    python report/final/scripts/measure_model.py
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from islvit.export import load_release  # noqa: E402
from islvit.models.isl_vit import ISLViT  # noqa: E402

OUT = ROOT / "report" / "final" / "data" / "model_measurements.json"


def inputs(frames: int, landmarks: bool) -> dict:
    g = torch.Generator().manual_seed(0)
    x = {"crops": torch.rand(1, frames, 3, 3, 64, 64, generator=g),
         "detected": torch.ones(1, frames, 3, dtype=torch.bool),
         "geometry": torch.rand(1, frames, 3, 3, generator=g)}
    if landmarks:
        x["hands"] = torch.rand(1, frames, 2, 21, 3, generator=g)
        x["hand_present"] = torch.ones(1, frames, 2)
        pose = torch.rand(1, frames, 7, 3, generator=g)
        pose[..., 2] = 1.0
        x["pose"] = pose
    return x


def params(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    parts = {name: sum(p.numel() for p in child.parameters()) for name, child in model.named_children()}
    loose = total - sum(parts.values())
    return {"total": total, "by_module": parts, "free_parameters": loose}


def flops(model: torch.nn.Module, x: dict) -> int:
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(**x)
    return counter.get_total_flops()


def latency(model: torch.nn.Module, x: dict, warmup: int = 10, runs: int = 100) -> dict:
    with torch.no_grad():
        for _ in range(warmup):
            model(**x)
        times = []
        for _ in range(runs):
            t = time.perf_counter()
            model(**x)
            times.append(1000 * (time.perf_counter() - t))
    q = statistics.quantiles(times, n=4)
    return {"median_ms": statistics.median(times), "q1_ms": q[0], "q3_ms": q[2], "runs": runs}


def main() -> None:
    torch.manual_seed(0)
    res = {"environment": {"python": platform.python_version(), "torch": torch.__version__,
                           "cpu": platform.processor(), "threads": torch.get_num_threads(),
                           "os": platform.platform()}}
    for name, lm in (("pixel_only_262", False), ("landmark_262", True)):
        m = ISLViT(n_classes=262, n_frames=16, landmarks=lm).eval()
        res[name] = {"params": params(m)}
        for frames in (8, 16):
            mm = ISLViT(n_classes=262, n_frames=frames, landmarks=lm).eval()
            f = flops(mm, inputs(frames, lm))
            res[name][f"flops_{frames}f"] = f
            res[name][f"gmacs_{frames}f"] = f / 2e9
    # Crop-resolution ablation (8 frames, 50 words, pixel-only), as in runs/abl*__take-group.
    res["resolution_ablation_gmacs_8f"] = {}
    for px in (64, 112, 128):
        m = ISLViT(n_classes=50, n_frames=8, img_size=px).eval()
        x = {"crops": torch.rand(1, 8, 3, 3, px, px), "detected": torch.ones(1, 8, 3, dtype=torch.bool),
             "geometry": torch.rand(1, 8, 3, 3)}
        res["resolution_ablation_gmacs_8f"][px] = flops(m, x) / 2e9
    rel = ROOT / "release" / "isl_vit_tiny_lm_int4_s0.pt"
    model, config, classes = load_release(rel, device="cpu")
    res["release_s0"] = {"config": {k: config[k] for k in ("n_frames", "img_size") if k in config},
                         "n_classes": len(classes), "params": params(model)["total"],
                         "latency_one_view_cpu": latency(model, inputs(config["n_frames"], True))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    main()
