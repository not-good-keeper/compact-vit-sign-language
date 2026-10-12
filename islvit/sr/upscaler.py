"""Apply a trained generator in place of bicubic upscaling.

``crops.read_sampled_frames`` calls ``upscale_frames`` when a video is below the
720p working height and an upscaler is configured. The generator always runs at
x4 and the result is area-downscaled to the exact target, Real-ESRGAN's
"outscale" approach, so one model serves a 2x (360p), 2.4x (CISLR) and 4x
(180p) upscale alike.

Inference is deterministic: eval mode, float32, no randomness, so a clip's
detections depend only on the clip and the generator file.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from islvit.sr.models import SRGenerator

_LOADED: dict[tuple[str, str], SRGenerator] = {}
# Frames per forward pass. A 640x360 input at 64 features is ~60 MB of activations
# per layer per frame, so a handful at a time keeps peak memory well under 2 GB.
CHUNK = 4


def half_precision() -> bool:
    """ISLVIT_SR_FP16=1: run the generator in float16 on the GPU. Opt-in, so nothing that already
    exists changes. Measured on real 180p frames against float32: largest pixel difference 1/255,
    mean 0.03/255, no pixel off by more than 2, and 1.8x faster; it is recorded in each cache's
    FRONTEND.json so a half-precision cache is never paired with a float32 one."""
    return os.environ.get("ISLVIT_SR_FP16") == "1"


def device_for() -> str:
    wanted = os.environ.get("ISLVIT_SR_DEVICE")
    if wanted:
        return wanted
    return "cuda" if torch.cuda.is_available() else "cpu"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: str | Path, device: str | None = None) -> SRGenerator:
    device = device or device_for()
    key = (str(Path(path).resolve()), device)
    if key not in _LOADED:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        generator = SRGenerator(**checkpoint["config"])
        generator.load_state_dict(checkpoint["generator"])
        generator = generator.eval().to(device)
        if half_precision() and device.startswith("cuda"):
            generator = generator.half()
        _LOADED[key] = generator
    return _LOADED[key]


@torch.no_grad()
def super_resolve(images: list[np.ndarray], path: str | Path, device: str | None = None) -> list[np.ndarray]:
    """BGR uint8 images (all the same size) -> x4 BGR uint8 images."""
    device = device or device_for()
    generator = load(path, device)
    out: list[np.ndarray] = []
    for start in range(0, len(images), CHUNK):
        batch = np.stack([img[..., ::-1] for img in images[start:start + CHUNK]])  # BGR -> RGB
        x = torch.from_numpy(np.ascontiguousarray(batch)).to(device).permute(0, 3, 1, 2).float().div_(255)
        if next(generator.parameters()).dtype == torch.float16:
            x = x.half()
        y = generator(x).float().clamp_(0, 1).mul_(255).round_().byte().permute(0, 2, 3, 1).cpu().numpy()
        out.extend(np.ascontiguousarray(frame[..., ::-1]) for frame in y)  # RGB -> BGR
    return out


def upscale_frames(frames: list[np.ndarray], size: tuple[int, int], path: str | Path) -> list[np.ndarray]:
    """Replace ``cv2.resize(frame, size, INTER_CUBIC)`` for frames smaller than ``size``."""
    return [cv2.resize(frame, size, interpolation=cv2.INTER_AREA) for frame in super_resolve(frames, path)]
