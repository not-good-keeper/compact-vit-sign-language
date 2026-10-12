"""Hand-structure proxy: a differentiable stand-in for MediaPipe's hand landmarker.

The GAN study's diagnostics (docs/GAN_IMPROVEMENT_PLAN.md section 7) put about three
quarters of what super-resolution still loses at 180p on the *landmark* side: the
classifier is fine once the hand landmarks are right. MediaPipe is not
differentiable, so a generator cannot be trained against it directly. This network
is: it learns to read the 21 MediaPipe hand landmarks off a 128 px hand crop, from
the clean caches, and the generator (islvit.sr.train, ``--hand-proxy``) is then
trained so that the proxy reads the same landmarks from its output as from the
clean original.

Data: hand crops whose box is MediaPipe's own (Holistic, direct detection), so the
landmark positions inside the crop are exact, recomputed with crops.box_from_points.
Training clips are the *original* train sessions of full263__session-disjoint.csv;
the original val sessions are held out here and in generator training, and are what
development decisions are made on. The 472 test clips are never touched.

    python -m islvit.sr.handproxy build --split train --out C:/isl_gan_work/hp_train.npz
    python -m islvit.sr.handproxy build --split val   --out C:/isl_gan_work/hp_val.npz --per-clip 6
    python -m islvit.sr.handproxy train --train C:/isl_gan_work/hp_train.npz --val C:/isl_gan_work/hp_val.npz --out runs/hand_proxy
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from islvit.data import crops

CACHE, LM = Path("cache128_f32"), Path("cache_lm_f32")
SPLIT_FILE = Path("splits/full263__session-disjoint.csv")
TEST472 = Path("splits/_test472_paths.csv")
W, H = crops.WORK_WIDTH, crops.WORK_HEIGHT
N_POINTS = 21
# Mirror pairs are not needed: a horizontally flipped hand keeps its landmark order
# (index 0 is still the wrist), only u -> 1 - u.


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def split_rows(split: str) -> list[int]:
    with SPLIT_FILE.open(encoding="utf-8") as h:
        wanted = {r["video_path"] for r in csv.DictReader(h) if r["split"] == split}
    with TEST472.open(encoding="utf-8") as h:
        test = {r["video_path"] for r in csv.DictReader(h)}
    assert not wanted & test, "a test clip is in the requested split"
    with (CACHE / "index.csv").open(encoding="utf-8") as h:
        return [int(r["row"]) for r in csv.DictReader(h) if r["cached"] == "1" and r["video_path"] in wanted]


# The crop cache (built first) and the landmark cache (built later, with fresh detectors
# per video) disagree on about a quarter of frames: an older MediaPipe graph carried
# state between videos (see crops.reset_detectors). Only frames where the box rebuilt
# from the landmarks matches the stored crop geometry are used -- to within float16
# storage plus whole-pixel rounding -- so every label really belongs to its crop.
GEOMETRY_TOLERANCE = 0.0012


def crop_coords(points: np.ndarray, geometry: np.ndarray | None = None):
    """Frame-normalised (21, 3) landmarks -> (21, 2) in crop units, or None if the box
    was clamped at the frame edge or does not match the crop's stored geometry."""
    xs, ys = points[:, 0] * W, points[:, 1] * H
    box = crops.box_from_points(list(xs), list(ys), crops.HAND_MARGIN, W, H)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    if x1 - x0 != y1 - y0:          # clamped at the frame edge: the crop was stretched
        return None
    if geometry is not None:
        g = np.asarray(geometry, np.float32)
        if max(abs((x0 + x1) / 2 / W - g[0]), abs((y0 + y1) / 2 / H - g[1]), abs((x1 - x0) / W - g[2])) > GEOMETRY_TOLERANCE:
            return None
    side = x1 - x0
    return np.stack([(xs - x0) / side, (ys - y0) / side], axis=1).astype(np.float32)


def cmd_build(a):
    rows = split_rows(a.split)
    rng = np.random.default_rng(a.seed)
    sources = np.load(CACHE / "sources.npy", mmap_mode="r")
    hand_src = np.load(LM / "hand_src.npy", mmap_mode="r")
    hands = np.load(LM / "hands.npy", mmap_mode="r")
    images = np.load(CACHE / "crops.npy", mmap_mode="r")
    geometry = np.load(CACHE / "geometry.npy", mmap_mode="r")
    xs, ys, meta = [], [], []
    for row in rows:
        picks = []
        for t in range(sources.shape[1]):
            for s in range(2):
                # Both passes must say "Holistic found this hand directly", so crop and landmarks are one detection.
                if sources[row, t, s] == crops.SOURCE_DIRECT and hand_src[row, t, s] == 1:
                    picks.append((t, s))
        rng.shuffle(picks)
        for t, s in picks[: a.per_clip]:
            uv = crop_coords(np.asarray(hands[row, t, s], np.float32), geometry[row, t, s])
            if uv is None or uv.min() < -0.05 or uv.max() > 1.05:
                continue
            xs.append(np.asarray(images[row, t, s]))
            ys.append(uv)
            meta.append((row, t, s))
    x, y = np.stack(xs), np.stack(ys)
    np.savez(a.out, images=x, points=y, meta=np.array(meta, np.int32))
    print(f"{a.split}: {len(rows)} clips -> {len(x)} hand crops ({x.nbytes / 2**30:.2f} GiB) -> {a.out}")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class HandProxy(nn.Module):
    """ResNet-18 trunk -> 21 heatmaps -> soft-argmax (u, v) landmarks.

    SimpleBaseline-style: three deconvolutions bring the 4x4 trunk output back to
    32x32 heatmaps, and a soft-argmax turns each into a coordinate, so the output
    stays differentiable for the generator loss. (A flatten-and-regress head was
    tried first and underfit badly: 0.094 mean error against MediaPipe's own 0.019
    re-run consistency on the same crops.) ``features`` also returns the heatmaps
    and two trunk maps, which the generator loss can match as well."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        import torchvision
        net = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        layers, channels = [], 512
        for _ in range(3):
            layers += [nn.ConvTranspose2d(channels, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
            channels = 256
        self.deconv = nn.Sequential(*layers)
        self.heat = nn.Conv2d(256, N_POINTS, 1)
        self.register_buffer("mean", IMAGENET_MEAN.clone())
        self.register_buffer("std", IMAGENET_STD.clone())

    def features(self, x: torch.Tensor):
        """x: RGB in [0, 1], (B, 3, 128, 128) -> (points (B, 21, 2), [layer2, layer3, heatmap logits])."""
        x = (x - self.mean) / self.std
        x = self.layer1(self.stem(x))
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        logits = self.heat(self.deconv(self.layer4(f3)))          # (B, 21, 32, 32)
        b, k, h, w = logits.shape
        prob = logits.float().flatten(2).softmax(-1).view(b, k, h, w)
        grid_x = (torch.arange(w, device=x.device, dtype=prob.dtype) + 0.5) / w
        grid_y = (torch.arange(h, device=x.device, dtype=prob.dtype) + 0.5) / h
        points = torch.stack([(prob.sum(2) * grid_x).sum(-1), (prob.sum(3) * grid_y).sum(-1)], -1)
        return points, [f2, f3, logits]

    def forward(self, x):
        return self.features(x)[0]


def load(path: str | Path, device: str) -> HandProxy:
    model = HandProxy(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    return model.to(device).eval()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def to_tensor(images: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float().div_(255)


def augment(x: torch.Tensor, y: torch.Tensor):
    """Geometric augmentation only (labels move with the image). No blur or noise:
    the proxy should read hands the way MediaPipe does on clean video, so that a
    degraded-looking hand reads differently -- that difference is the training signal."""
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
    y = y.clone()
    y[flip, :, 0] = 1 - y[flip, :, 0]
    # Small rotation and scale about the crop centre.
    angle = (torch.rand(x.shape[0], device=x.device) - 0.5) * (math.pi / 6)
    scale = 1 + (torch.rand(x.shape[0], device=x.device) - 0.5) * 0.2
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos, -sin, torch.zeros_like(cos)], 1),
                         torch.stack([sin, cos, torch.zeros_like(cos)], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    # Output pixel p samples input at theta @ p (in [-1, 1] coords), so a label moves by the inverse.
    centred = (y - 0.5) * 2
    inv = torch.linalg.inv(torch.cat([theta, torch.tensor([0, 0, 1.0], device=x.device).expand(x.shape[0], 1, 3)], 1))[:, :2, :2]
    y = torch.einsum("bij,bkj->bki", inv, centred) / 2 + 0.5
    return x, y


def evaluate(model, images, points, device, batch=256):
    errs = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            p = model(to_tensor(images[i:i + batch], device))
            errs.append((p - torch.from_numpy(points[i:i + batch]).to(device)).norm(dim=-1).cpu())
    e = torch.cat(errs)
    return {"mean_err": float(e.mean()), "median_err": float(e.median()),
            "pck_0.05": float((e < 0.05).float().mean()), "pck_0.10": float((e < 0.10).float().mean())}


def cmd_train(a):
    device = "cuda"
    torch.manual_seed(a.seed)
    tr, va = np.load(a.train), np.load(a.val)
    xtr, ytr = tr["images"], tr["points"]
    model = HandProxy().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(xtr) // a.batch)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[hand proxy] {len(xtr)} train / {len(va['images'])} val crops, {a.epochs} epochs, {steps} steps")
    step, start, history = 0, time.time(), []
    for epoch in range(a.epochs):
        model.train()
        order = np.random.default_rng(a.seed + epoch).permutation(len(xtr))
        total = 0.0
        for i in range(0, len(order) - a.batch + 1, a.batch):
            idx = np.sort(order[i:i + a.batch])
            x, y = augment(to_tensor(xtr[idx], device), torch.from_numpy(ytr[idx]).to(device))
            lr = a.lr * 0.5 * (1 + math.cos(math.pi * step / steps))
            for g in opt.param_groups:
                g["lr"] = lr
            loss = F.smooth_l1_loss(model(x), y, beta=0.02)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
            step += 1
        model.eval()
        val = evaluate(model, va["images"], va["points"], device)
        history.append({"epoch": epoch + 1, "loss": total / max(1, len(order) // a.batch), **val})
        print(f"  ep {epoch + 1:2d}  loss {history[-1]['loss']:.4f}  val mean err {val['mean_err']:.4f}  "
              f"PCK@0.05 {val['pck_0.05']:.3f}  {time.time() - start:.0f}s", flush=True)
    torch.save({"model": model.state_dict(), "epochs": a.epochs}, out / "proxy.pt")
    summary = {"train": a.train, "val": a.val, "n_train": int(len(xtr)), "n_val": int(len(va["images"])),
               "epochs": a.epochs, "selection": "final weights, epoch count fixed in advance",
               "final_val": history[-1], "history": history}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out / 'proxy.pt'}")


def main():
    p = argparse.ArgumentParser(description="Hand-structure proxy for the SR GAN")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--split", choices=("train", "val"), required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--per-clip", type=int, default=24)
    b.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--train", required=True); t.add_argument("--val", required=True); t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=20); t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3); t.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    {"build": cmd_build, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
