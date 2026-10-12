"""Image quality of each upscaler on held-out patches the GAN never trained on.

Each held-out 128 px patch (720p working resolution) is degraded the way the study
degrades whole frames -- area-downscaled by the level's factor, then one JPEG
encode -- and brought back to 128 px by bicubic, by SR-L1 and by SR-GAN.

Reported per level and upscaler:

* PSNR and SSIM against the original. These reward smooth averages, so the
  PSNR-trained SR-L1 is *expected* to beat the GAN on them; that is the
  perception-distortion trade-off, not a failure.
* Sharpness as variance of the Laplacian, the measure the report used to find
  that CISLR crops are 2.1x blurrier than INCLUDE's (section 11.5). The
  original patches give the target.

None of these is the study's verdict. A restored hand only matters if the
detector finds it and the classifier reads it, which the downstream study measures.

Usage::

    python -m islvit.sr.evaluate --bank C:/isl_gan_work/hr_heldout.npy \\
        --sr-l1 runs/sr_l1/generator.pt --sr-gan runs/sr_gan/generator.pt --out runs/gan_study/sr_quality.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from islvit.sr.degrade import jpeg
from islvit.sr.upscaler import super_resolve

LEVELS = {"L1 360p": (2, 60), "L2 240p": (3, 50), "L3 180p": (4, 40)}


def degrade_patch(hr: np.ndarray, factor: int, quality: int) -> np.ndarray:
    size = max(4, round(hr.shape[0] / factor))
    return jpeg(cv2.resize(hr, (size, size), interpolation=cv2.INTER_AREA), quality)


def gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    g = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def luma(x: torch.Tensor) -> torch.Tensor:
    return (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)


def psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mse = ((a - b) ** 2).mean(dim=(1, 2, 3)).clamp(min=1e-10)
    return 10 * torch.log10(1 / mse)


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SSIM on luma, 11x11 Gaussian window (Wang et al. 2004 constants)."""
    a, b = luma(a), luma(b)
    w = gaussian_window().to(a.device)
    mu_a, mu_b = F.conv2d(a, w), F.conv2d(b, w)
    var_a = F.conv2d(a * a, w) - mu_a ** 2
    var_b = F.conv2d(b * b, w) - mu_b ** 2
    cov = F.conv2d(a * b, w) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    value = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return value.mean(dim=(1, 2, 3))


def sharpness(image: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


def tensor(images: list[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().div(255)


def main() -> None:
    parser = argparse.ArgumentParser(description="Image quality of the SR upscalers")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--sr-l1", required=True)
    parser.add_argument("--sr-gan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=1200)
    args = parser.parse_args()

    bank = np.load(args.bank, mmap_mode="r")
    hr = [np.asarray(bank[i]) for i in range(min(args.limit, len(bank)))]
    size = hr[0].shape[0]
    results = {"bank": args.bank, "n": len(hr), "original_sharpness": float(np.median([sharpness(x) for x in hr])),
               "levels": {}}
    for name, (factor, quality) in LEVELS.items():
        low = [degrade_patch(x, factor, quality) for x in hr]
        # The bank is RGB; super_resolve speaks BGR like the video pipeline, so flip around it.
        outputs = {"bicubic": [cv2.resize(x, (size, size), interpolation=cv2.INTER_CUBIC) for x in low]}
        timings = {}
        for arm, path in (("sr_l1", args.sr_l1), ("sr_gan", args.sr_gan)):
            started = time.perf_counter()
            up = super_resolve([np.ascontiguousarray(x[..., ::-1]) for x in low], path)
            timings[arm] = (time.perf_counter() - started) / len(low) * 1000
            outputs[arm] = [cv2.resize(np.ascontiguousarray(x[..., ::-1]), (size, size), interpolation=cv2.INTER_AREA) for x in up]
        target = tensor(hr)
        level = {}
        for arm, images in outputs.items():
            pred = tensor(images)
            level[arm] = {
                "psnr": float(psnr(pred, target).mean()),
                "ssim": float(ssim(pred, target).mean()),
                "sharpness": float(np.median([sharpness(x) for x in images])),
                "ms_per_patch": round(timings.get(arm, 0.0), 3),
            }
        results["levels"][name] = level
        print(name, {arm: {k: round(v, 3) for k, v in m.items()} for arm, m in level.items()})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
