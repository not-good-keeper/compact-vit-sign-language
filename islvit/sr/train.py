"""Train the super-resolution GAN in two stages.

``--stage l1``  The generator alone, pixel L1 loss. This is the PSNR-oriented
                model and it doubles as the study's control: learned
                super-resolution *without* an adversarial loss (arm "SR-L1").
``--stage gan`` Starts from the l1 generator and adds a U-Net discriminator, a
                VGG19 perceptual loss and an adversarial loss (arm "SR-GAN").
                Comparing the two separates what the GAN contributes from what
                any learned upscaler would.

Version 3 adds what the study's diagnostics asked for (docs/GAN_IMPROVEMENT_PLAN.md
section 7): about three quarters of what super-resolution still loses at 180p is on
the hand-landmark side, so the generator can now be trained against a
differentiable stand-in for MediaPipe's hand landmarker (islvit.sr.handproxy):

* ``--hands`` mixes MediaPipe-labelled hand crops into every batch;
* ``--hand-proxy`` with ``--w-hand`` / ``--w-hand-feat`` asks the proxy to read the
  same 21 landmarks (and the same intermediate features) from the generator's
  output as from the clean crop;
* ``--g-features`` / ``--g-convs`` size the generator; ``--gan-loss ragan`` uses the
  relativistic average loss; ``--codec-prob`` adds H.264 intra compression to the
  training degradations;
* ``--val-hands`` logs a development metric on held-out-session hand crops
  degraded like 180p video: how far the proxy's landmarks on the generator's
  output are from its landmarks on the clean crop, next to bicubic's.

Both stages run a fixed number of iterations and keep the final EMA weights:
nothing is selected by looking at a result, which is the rule the rest of this
project follows. A resume checkpoint is written every --resume-every iterations
because power cuts have killed long runs here before.

Usage::

    python -m islvit.sr.train --stage l1  --bank C:/isl_gan_work/hr_train.npy --iters 100000 --out runs/sr_l1
    python -m islvit.sr.train --stage gan --bank C:/isl_gan_work/hr_train.npy --iters 100000 \\
        --init runs/sr_l1/generator.pt --out runs/sr_gan
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from islvit.sr.degrade import degrade_frame, random_degrade
from islvit.sr.models import SRGenerator, UNetDiscriminatorSN, VGGPerceptualLoss, count


class PatchPairs(Dataset):
    """(LR, HR) pairs: a bank patch, randomly flipped/rotated, and its random degradation.

    ``rotate`` is off for hand crops: the hand proxy reads upright hands (it was
    trained with at most +-15 degrees), so a 90-degree turn would make its loss noise."""

    def __init__(self, bank: str, scale: int = 4, seed: int = 0, codec_prob: float = 0.0, rotate: bool = True) -> None:
        self.bank_path, self.scale, self.seed, self.codec_prob, self.rotate = bank, scale, seed, codec_prob, rotate
        self.length = len(np.load(bank, mmap_mode="r"))
        self.bank = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        if self.bank is None:  # opened lazily so each worker has its own handle
            self.bank = np.load(self.bank_path, mmap_mode="r")
        rng = np.random.default_rng([self.seed, index, torch.initial_seed() % (1 << 31)])
        hr = np.array(self.bank[index])  # a writable copy, not a view of the read-only memmap
        if rng.random() < 0.5:
            hr = hr[:, ::-1]
        if self.rotate:
            hr = np.rot90(hr, int(rng.integers(4)))
        hr = np.ascontiguousarray(hr)
        lr = random_degrade(hr, rng, self.scale, codec_prob=self.codec_prob)
        to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float().div_(255)
        return to(lr), to(hr)


def batches(loader):
    while True:
        yield from loader


@torch.no_grad()
def ema_update(ema: nn.Module, model: nn.Module, decay: float) -> None:
    for e, m in zip(ema.parameters(), model.parameters()):
        e.mul_(decay).add_(m.detach(), alpha=1 - decay)


def hand_terms(proxy, fake: torch.Tensor, real: torch.Tensor):
    """(landmark error in crop units, normalised feature distance) between the proxy's
    reading of the generator's output and of the clean crop."""
    points_fake, feats_fake = proxy.features(fake)
    with torch.no_grad():
        points_real, feats_real = proxy.features(real)
    point_err = (points_fake.float() - points_real.float()).norm(dim=-1).mean()
    feat = sum(F.l1_loss(a.float(), b.float()) / (b.float().abs().mean() + 1e-6)
               for a, b in zip(feats_fake[:2], feats_real[:2])) / 2
    return point_err, feat


class ValHands:
    """Development metric on held-out-session hand crops (handproxy --split val):
    HR crop -> 32 px with the study's 180p degradation (area x4, JPEG 40) -> upscaler.
    Reports the proxy's landmark error against its own reading of the clean crop."""

    def __init__(self, path: str, device: str, n: int = 512) -> None:
        images = np.load(path)["images"][:n]                      # RGB uint8 (n, 128, 128, 3)
        lows = np.stack([degrade_frame(im[..., ::-1], 32, 40)[..., ::-1] for im in images])
        to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device).permute(0, 3, 1, 2).float().div_(255)
        self.high, self.low = to(images), to(lows)
        self.bicubic = None

    @torch.no_grad()
    def __call__(self, proxy, generator) -> dict:
        def err(upscale):
            out = []
            for i in range(0, len(self.low), 64):
                fake = upscale(self.low[i:i + 64]).clamp(0, 1)
                out.append(hand_terms(proxy, fake, self.high[i:i + 64])[0].item())
            return float(np.mean(out))
        if self.bicubic is None:
            self.bicubic = err(lambda x: F.interpolate(x, scale_factor=4, mode="bicubic", align_corners=False))
        return {"val_hand_err": err(generator), "val_hand_err_bicubic": self.bicubic}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SR GAN")
    parser.add_argument("--stage", choices=("l1", "gan"), required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--init", default=None, help="generator.pt to start from (required for --stage gan)")
    parser.add_argument("--iters", type=int, default=100000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None, help="default 2e-4 (l1) / 1e-4 (gan)")
    parser.add_argument("--w-pixel", type=float, default=1.0)
    parser.add_argument("--w-percep", type=float, default=1.0)
    parser.add_argument("--w-adv", type=float, default=0.1)
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resume-every", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--channels-last", action="store_true",
                        help="channels_last memory format for the networks and batches (faster bf16 convolutions)")
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                        help="run the discriminator and perceptual network in fp32 (default: bf16 autocast). "
                             "Measured on the RTX 3060 they are ~95 %% of a GAN step in fp32; the generator "
                             "always stays fp32")
    # v3
    parser.add_argument("--g-features", type=int, default=64)
    parser.add_argument("--g-convs", type=int, default=16)
    parser.add_argument("--gan-loss", choices=("bce", "ragan"), default="bce")
    parser.add_argument("--codec-prob", type=float, default=0.0, help="share of samples compressed with H.264")
    parser.add_argument("--hands", default=None, help="hand-crop bank (.npy, uint8 RGB 128 px) mixed into every batch")
    parser.add_argument("--hand-frac", type=float, default=0.5, help="share of each batch drawn from --hands")
    parser.add_argument("--hand-proxy", default=None, help="islvit.sr.handproxy checkpoint")
    parser.add_argument("--w-hand", type=float, default=0.0, help="weight of the proxy landmark error")
    parser.add_argument("--w-hand-feat", type=float, default=0.0, help="weight of the proxy feature distance")
    parser.add_argument("--val-hands", default=None, help="handproxy --split val .npz for the development metric")
    parser.add_argument("--val-every", type=int, default=5000)
    args = parser.parse_args()

    if args.stage == "gan" and not args.init:
        raise SystemExit("--stage gan needs --init runs/sr_l1/generator.pt")
    if (args.w_hand or args.w_hand_feat) and not (args.hands and args.hand_proxy):
        raise SystemExit("--w-hand / --w-hand-feat need --hands and --hand-proxy")
    lr = args.lr or (2e-4 if args.stage == "l1" else 1e-4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True  # fixed 32/128 px shapes
    cl = torch.channels_last if args.channels_last else torch.contiguous_format
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n_hand = int(round(args.batch * args.hand_frac)) if args.hands else 0
    n_bank = args.batch - n_hand

    def loader(dataset, size):
        return DataLoader(dataset, batch_size=size, shuffle=True, drop_last=True,
                          num_workers=max(1, args.workers // (2 if n_hand else 1)),
                          pin_memory=device == "cuda", persistent_workers=args.workers > 0)

    data = PatchPairs(args.bank, seed=args.seed, codec_prob=args.codec_prob)
    stream = batches(loader(data, n_bank))
    hand_stream = None
    if n_hand:
        hand_data = PatchPairs(args.hands, seed=args.seed + 1, codec_prob=args.codec_prob, rotate=False)
        hand_stream = batches(loader(hand_data, n_hand))

    if args.init:
        config = torch.load(args.init, map_location="cpu", weights_only=False)["config"]
        generator = SRGenerator(**config).to(device, memory_format=cl)
        generator.load_state_dict(torch.load(args.init, map_location=device, weights_only=False)["generator"])
    else:
        generator = SRGenerator(features=args.g_features, convs=args.g_convs).to(device, memory_format=cl)
    ema = copy.deepcopy(generator).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.9, 0.99))

    discriminator = perceptual = opt_d = None
    if args.stage == "gan":
        discriminator = UNetDiscriminatorSN().to(device, memory_format=cl)
        perceptual = VGGPerceptualLoss().to(device, memory_format=cl)
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.9, 0.99))
    bce = nn.BCEWithLogitsLoss()

    proxy = None
    if args.hand_proxy:
        from islvit.sr import handproxy
        proxy = handproxy.load(args.hand_proxy, device).to(memory_format=cl)
        for p in proxy.parameters():
            p.requires_grad_(False)
    val_hands = ValHands(args.val_hands, device) if (args.val_hands and proxy is not None) else None

    fingerprint = {k: getattr(args, k) for k in ("stage", "bank", "iters", "batch", "seed", "init", "g_features",
                                                 "g_convs", "gan_loss", "codec_prob", "hands", "hand_frac",
                                                 "hand_proxy", "w_hand", "w_hand_feat")}
    start = 0
    resume_path = out / "resume.pt"
    if args.resume and resume_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state["fingerprint"] != fingerprint:
            raise SystemExit(f"resume.pt was written for {state['fingerprint']}, this run is {fingerprint}")
        generator.load_state_dict(state["generator"])
        ema.load_state_dict(state["ema"])
        opt_g.load_state_dict(state["opt_g"])
        if discriminator is not None:
            discriminator.load_state_dict(state["discriminator"])
            opt_d.load_state_dict(state["opt_d"])
        start = state["iter"] + 1
        print(f"  resumed at iteration {start}/{args.iters}")

    history_path = out / "history.csv"
    if start and history_path.exists():
        with history_path.open(encoding="utf-8") as handle:
            kept = [row for row in csv.DictReader(handle) if int(row["iter"]) < start]
    else:
        kept = []
    fields = ["iter", "lr", "pixel", "percep", "adv", "d_loss", "d_real", "d_fake", "hand", "hand_feat", "it_per_s",
              "val_hand_err", "val_hand_err_bicubic"]
    history = history_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(history, fieldnames=fields)
    writer.writeheader()
    writer.writerows(kept)

    print(f"[sr {args.stage}] G {count(generator) / 1e6:.2f} M"
          + (f", D {count(discriminator) / 1e6:.2f} M, perceptual {perceptual.backbone}, {args.gan_loss}" if discriminator else "")
          + f", {len(data)} patches" + (f" + hand crops ({n_hand}/{args.batch} per batch)" if n_hand else "")
          + (f", hand loss {args.w_hand}/{args.w_hand_feat}" if (args.w_hand or args.w_hand_feat) else "")
          + f", codec {args.codec_prob}, batch {args.batch}, {args.iters} iterations on {device}")

    autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda" and args.amp)
    tally = {k: 0.0 for k in fields[2:10]}
    tick, started = time.time(), time.time()
    last_val: dict = {}
    for it in range(start, args.iters):
        # Cosine decay to 1 % over the fixed schedule.
        scale = 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * it / args.iters))
        for group in opt_g.param_groups:
            group["lr"] = lr * scale
        if opt_d:
            for group in opt_d.param_groups:
                group["lr"] = lr * scale

        low, high = (t.to(device, non_blocking=True, memory_format=cl) for t in next(stream))
        if hand_stream is not None:
            hand_low, hand_high = (t.to(device, non_blocking=True, memory_format=cl) for t in next(hand_stream))
            low, high = torch.cat([low, hand_low]), torch.cat([high, hand_high])
        fake = generator(low)
        pixel = F.l1_loss(fake, high)
        loss_g = args.w_pixel * pixel
        if proxy is not None and (args.w_hand or args.w_hand_feat):
            with autocast():
                hand, hand_feat = hand_terms(proxy, fake[n_bank:].clamp(0, 1), high[n_bank:])
            loss_g = loss_g + args.w_hand * hand + args.w_hand_feat * hand_feat
            tally["hand"] += hand.item()
            tally["hand_feat"] += hand_feat.item()
        if discriminator is not None:
            for p in discriminator.parameters():
                p.requires_grad_(False)
            with autocast():
                percep = perceptual(fake, high).float()
                logits = discriminator(fake).float()
                if args.gan_loss == "ragan":
                    with torch.no_grad():
                        real_ref = discriminator(high).float()
            if args.gan_loss == "ragan":
                adv = (bce(logits - real_ref.mean(), torch.ones_like(logits))
                       + bce(real_ref - logits.mean(), torch.zeros_like(real_ref))) / 2
            else:
                adv = bce(logits, torch.ones_like(logits))
            loss_g = loss_g + args.w_percep * percep + args.w_adv * adv
            tally["percep"] += percep.item()
            tally["adv"] += adv.item()
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()
        tally["pixel"] += pixel.item()

        if discriminator is not None:
            for p in discriminator.parameters():
                p.requires_grad_(True)
            with autocast():
                real_logits = discriminator(high).float()
                fake_logits = discriminator(fake.detach()).float()
            if args.gan_loss == "ragan":
                loss_d = (bce(real_logits - fake_logits.mean(), torch.ones_like(real_logits))
                          + bce(fake_logits - real_logits.mean(), torch.zeros_like(fake_logits))) / 2
            else:
                loss_d = bce(real_logits, torch.ones_like(real_logits)) + bce(fake_logits, torch.zeros_like(fake_logits))
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()
            tally["d_loss"] += loss_d.item()
            tally["d_real"] += torch.sigmoid(real_logits).mean().item()
            tally["d_fake"] += torch.sigmoid(fake_logits).mean().item()

        ema_update(ema, generator, args.ema if it > 1000 else min(args.ema, (1 + it) / (10 + it)))

        if val_hands is not None and ((it + 1) % args.val_every == 0 or it == args.iters - 1):
            with autocast():
                last_val = val_hands(proxy, ema)

        if (it + 1) % args.log_every == 0 or it == args.iters - 1:
            n = (it % args.log_every) + 1
            rate = n / max(1e-6, time.time() - tick)
            row = {"iter": it, "lr": f"{lr * scale:.2e}", "it_per_s": f"{rate:.1f}",
                   **{k: f"{v / n:.4f}" for k, v in tally.items()},
                   **{k: f"{v:.4f}" for k, v in last_val.items()}}
            writer.writerow(row)
            history.flush()
            if (it + 1) % (args.log_every * 10) == 0 or it == args.iters - 1:
                left = (args.iters - it - 1) / max(rate, 1e-6) / 60
                extra = f"  hand {row['hand']}/{row['hand_feat']}" if n_hand else ""
                extra += (f"  val-hand {last_val['val_hand_err']:.4f} (bicubic {last_val['val_hand_err_bicubic']:.4f})"
                          if last_val else "")
                print(f"  it {it + 1:6d}  pixel {row['pixel']}  percep {row['percep']}  adv {row['adv']}  "
                      f"D {row['d_loss']} ({row['d_real']}/{row['d_fake']}){extra}  {rate:.1f} it/s  {left:.0f} min left",
                      flush=True)
            tally = {k: 0.0 for k in tally}
            tick = time.time()

        if args.resume_every and ((it + 1) % args.resume_every == 0):
            state = {"generator": generator.state_dict(), "ema": ema.state_dict(), "opt_g": opt_g.state_dict(),
                     "iter": it, "fingerprint": fingerprint}
            if discriminator is not None:
                state.update(discriminator=discriminator.state_dict(), opt_d=opt_d.state_dict())
            torch.save(state, resume_path)

    history.close()
    torch.save({"generator": ema.state_dict(), "config": generator.config, "stage": args.stage,
                "iters": args.iters}, out / "generator.pt")
    summary = {
        "stage": args.stage, "iters": args.iters, "batch": args.batch, "lr": lr, "seed": args.seed,
        "bank": args.bank, "init": args.init, "generator_params": count(generator), "generator_config": generator.config,
        "discriminator_params": count(discriminator) if discriminator else None,
        "perceptual": perceptual.backbone if perceptual else None, "gan_loss": args.gan_loss if discriminator else None,
        "weights": ({"pixel": args.w_pixel, "percep": args.w_percep, "adv": args.w_adv} if discriminator else {"pixel": 1.0})
                   | {"hand": args.w_hand, "hand_feat": args.w_hand_feat},
        "hands": args.hands, "hand_frac": args.hand_frac if args.hands else None, "hand_proxy": args.hand_proxy,
        "codec_prob": args.codec_prob, "final_val": last_val or None,
        "ema": args.ema, "selection": "final EMA weights, iteration count fixed in advance",
        "amp": bool(args.amp and discriminator is not None),
        "minutes": round((time.time() - started) / 60, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    resume_path.unlink(missing_ok=True)
    print(f"[sr {args.stage}] wrote {out / 'generator.pt'}")


if __name__ == "__main__":
    main()
