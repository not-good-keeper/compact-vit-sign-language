"""Trace one real clip through every stage of the model, with real numbers.

Written for the review: a reader should be able to follow a single video from
pixels to a ranked word without taking anything on trust. Everything printed here
is read off the actual tensors as they pass through, not reconstructed afterwards.

Two clips are traced by default -- one the model answers confidently and one it
declines -- because a system whose defining behaviour is "say nothing when unsure"
is only half explained by a successful example.

Usage::

    python -m islvit.walkthrough --run runs/f16_262w_s0
    python -m islvit.walkthrough --run runs/f16_262w_s0 --figure docs/figures/fig19_walkthrough.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from islvit.data.dataset import prepare_clip
from islvit.eval import load_run
from islvit.predict import OFFSETS, FLIPS, extract, views

STREAMS = ("left hand", "right hand", "face")
SOURCE_NAMES = {0: "missing", 1: "direct", 2: "ROI-rescued", 3: "interpolated"}


def resolve(rel: str) -> Path | None:
    for base in ("INCLUDE_raw", "INCLUDE_480p"):
        for suffix in (None, ".MOV", ".mp4", ".MP4"):
            path = Path(base) / rel
            path = path if suffix is None else path.with_suffix(suffix)
            if path.exists():
                return path
    return None


def trace(video: Path, truth: str, model, config, classes, keep, device, crop_size):
    """Run one clip and report every intermediate the model actually used."""
    print(f"\n{'=' * 78}\n  CLIP  {video.name}     ground truth: {truth}\n{'=' * 78}")

    import cv2
    capture = cv2.VideoCapture(str(video))
    raw_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    print(f"\n  STAGE 1 · INPUT")
    print(f"    raw video            {width}x{height}, {raw_frames} frames @ {fps:.0f} fps")
    print(f"    tensor              ({raw_frames}, {height}, {width}, 3) uint8")

    clip, detected, geometry, sources = extract(video, crop_size)
    n_cached = clip.shape[0]
    print(f"\n  STAGE 2 · SAMPLE + DETECT")
    print(f"    {n_cached} frames sampled uniformly across the clip, letterboxed to 720x1280")
    print(f"    MediaPipe Holistic per frame; on a miss, HandLandmarker on an upscaled ROI")
    print(f"    crops tensor        {tuple(clip.shape)} uint8   <- 3 anatomical streams")
    print(f"\n    where each crop's box came from:")
    for s, name in enumerate(STREAMS):
        counts = np.bincount(sources[:, s].astype(int), minlength=4)
        parts = ", ".join(f"{SOURCE_NAMES[i]} {c}" for i, c in enumerate(counts) if c)
        genuine = (counts[1] + counts[2]) / max(1, counts.sum())
        print(f"      {name:<12s} {parts:<46s} genuine {genuine:.0%}")

    print(f"\n  STAGE 3 · GEOMETRY  (what cropping deletes, fed back as a feature)")
    print(f"    geometry tensor     {tuple(geometry.shape)} float32  = (centre x, centre y, size)")
    print(f"    frame 8, normalised to the source frame:")
    for s, name in enumerate(STREAMS):
        cx, cy, size = geometry[8, s]
        print(f"      {name:<12s} cx {cx:.3f}  cy {cy:.3f}  size {size:.3f}"
              f"   ({'detected' if detected[8, s] else 'interpolated'})")
    hands = geometry[:, :2, 1]
    print(f"    right-hand height over the clip: {hands[:, 1].min():.2f} -> {hands[:, 1].max():.2f}"
          f"  (vertical travel {hands[:, 1].max() - hands[:, 1].min():.2f})")
    print(f"    this is the signal that separates the same handshape at forehead vs chest")

    n_frames, img_size = config["n_frames"], config["img_size"]
    edges = np.linspace(0, n_cached, n_frames + 1)
    picks = np.clip((edges[:-1] + 0.5 * (edges[1:] - edges[:-1])).astype(int), 0, n_cached - 1)
    prepared = prepare_clip(clip[picks], detected[picks], geometry[picks],
                            img_size=img_size, train=False)
    print(f"\n  STAGE 4 · SAMPLE {n_frames} OF {n_cached} + NORMALISE")
    print(f"    frames chosen       {list(picks)}")
    print(f"    model input         {tuple(prepared[0].shape)} float32"
          f"   = ({n_frames} frames, 3 streams, 3 channels, {img_size}, {img_size})")
    print(f"    value range         [{prepared[0].min():+.2f}, {prepared[0].max():+.2f}]"
          f"   (ImageNet mean/std normalised)")

    patches = (img_size // config.get("patch_size", 16)) ** 2
    dim = config.get("dim", 192)
    print(f"\n  STAGE 5 · TOKENS")
    print(f"    per crop            {patches} patches of 16x16 + 1 [cls] = {patches + 1} tokens")
    print(f"    total crops         {n_frames} x 3 = {n_frames * 3}")
    print(f"    spatial stage in    ({n_frames * 3}, {patches + 1}, {dim})   weights shared over all"
          f" {n_frames * 3} crops")
    print(f"    spatial stage out   ({n_frames}, 3, {dim})        [cls] token per crop, + geometry")
    print(f"    temporal stage in   ({n_frames}, {dim})           3 streams fused, + time embedding")
    print(f"    pooled              ({dim},)              mean over {n_frames} timesteps")
    print(f"    logits              ({len(classes)},)")

    total = None
    with torch.no_grad():
        for crops, det, geo in views(clip, detected, geometry, n_frames, img_size, True):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(crops.to(device), det.to(device), geo.to(device))
            p = logits.float().softmax(1)
            total = p if total is None else total + p
    probs = (total / (len(OFFSETS) * len(FLIPS))).squeeze(0).cpu().numpy()

    masked = np.full_like(probs, -np.inf)
    masked[keep] = probs[keep]
    order = np.argsort(-masked)[:5]
    confidence = float(masked[order[0]])

    print(f"\n  STAGE 6 · OUTPUT  ({len(OFFSETS) * len(FLIPS)} TTA views averaged,"
          f" head restricted to the {len(keep)} deployed words)")
    for rank, index in enumerate(order, 1):
        bar = "#" * int(round(masked[index] * 40))
        mark = "  <- correct" if classes[index] == truth else ""
        print(f"    {rank}. {classes[index]:<24s} {masked[index]:6.1%}  {bar}{mark}")

    tau = 0.40
    print(f"\n  STAGE 7 · GATE   tau = {tau:.2f}")
    if confidence >= tau:
        print(f"    confidence {confidence:.1%} >= {tau:.2f}  ->  ANSWER \"{classes[order[0]]}\"")
    else:
        print(f"    confidence {confidence:.1%} <  {tau:.2f}  ->  ABSTAIN, \"didn't catch that\"")
        print(f"    (the top guess {classes[order[0]]!r} is "
              f"{'correct' if classes[order[0]] == truth else 'wrong'} — "
              f"declining here {'costs a repeat' if classes[order[0]] == truth else 'avoids a wrong answer'})")

    return dict(clip=clip, sources=sources, picks=picks, probs=masked, order=order,
                truth=truth, confidence=confidence, name=video.name)


def figure(traces, classes, path: Path):
    """Real crops beside the ranked output, one block per traced clip."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from islvit.figures import INK, INK2

    steps = [0, 6, 12, 18, 24, 31]
    provenance_colour = {1: "#1d6b3f", 2: "#5aa27a", 3: "#8a5a0c", 0: "#9a352f"}

    fig = plt.figure(figsize=(15, 4.6 * len(traces)))
    # One outer row per clip, split into an image block and a chart. The chart gets
    # its own column with real padding: an earlier version nested both in one grid
    # and the bar labels landed on top of the crops.
    outer = fig.add_gridspec(len(traces), 2, width_ratios=[len(steps), 4.4],
                             hspace=0.34, wspace=0.30, left=0.05, right=0.97,
                             top=0.90, bottom=0.07)

    for t, tr in enumerate(traces):
        inner = outer[t, 0].subgridspec(3, len(steps), wspace=0.06, hspace=0.06)
        for s in range(3):
            for k, frame in enumerate(steps):
                ax = fig.add_subplot(inner[s, k])
                index = min(frame, tr["clip"].shape[0] - 1)
                ax.imshow(tr["clip"][index, s])
                ax.set_xticks([]); ax.set_yticks([])
                for side in ax.spines.values():
                    side.set_color(provenance_colour[int(tr["sources"][index, s])])
                    side.set_linewidth(2.2)
                if k == 0:
                    ax.set_ylabel(STREAMS[s], fontsize=8.5, color=INK2)
                if s == 0:
                    ax.set_title(f"t={frame}", fontsize=9, color=INK2, pad=4)

        ax = fig.add_subplot(outer[t, 1])
        names = [classes[i].split(". ", 1)[-1] for i in tr["order"]][::-1]
        values = [tr["probs"][i] * 100 for i in tr["order"]][::-1]
        colours = ["#0d5c5b" if classes[i] == tr["truth"] else "#c8ccd8"
                   for i in tr["order"]][::-1]
        ax.barh(range(5), values, color=colours, height=0.6)
        ax.set_yticks(range(5)); ax.set_yticklabels(names, fontsize=10)
        ax.set_xlim(0, 112)
        for i, value in enumerate(values):
            ax.text(value + 2, i, f"{value:.1f}%", va="center", fontsize=9, color=INK2)
        ax.axvline(40, color="#9a352f", ls="--", lw=1.5)
        ax.text(41, 4.45, "tau = 0.40", fontsize=8.5, color="#9a352f")
        answered = tr["confidence"] >= 0.40
        ax.set_title(
            f'truth "{tr["truth"].split(". ", 1)[-1]}"   '
            + (f'ANSWERED at {tr["confidence"]:.0%}' if answered
               else f'ABSTAINED at {tr["confidence"]:.0%}'),
            fontsize=11.5, fontweight="600", loc="left", pad=10,
            color="#1d6b3f" if answered else "#8a5a0c")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=9, colors=INK2)
        ax.set_xlabel("probability after 6-view TTA (%)", fontsize=9, color=INK2)

    fig.suptitle("One clip end to end: three crop streams per timestep, "
                 "then ranked words, then the gate",
                 fontsize=13.5, fontweight="600", color=INK, y=0.975)
    handles = [plt.Line2D([], [], marker="s", ls="", ms=9, color=c)
               for c in ("#1d6b3f", "#5aa27a", "#8a5a0c")]
    fig.legend(handles, ["box from direct detection", "box rescued by ROI pass",
                         "box interpolated from a neighbour"],
               loc="lower left", bbox_to_anchor=(0.05, 0.005), ncol=3,
               frameon=False, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"\n  wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace real clips through the model")
    parser.add_argument("--run", type=str, default="runs/f16_262w_s0")
    parser.add_argument("--split", type=str, default="splits/vocab50clean__session-disjoint.csv")
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--figure", type=str, default="docs/figures/fig19_walkthrough.png")
    parser.add_argument("--clips", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config, classes = load_run(Path(args.run), device)
    rows = [r for r in csv.DictReader(open(args.split, encoding="utf-8")) if r["split"] == "test"]
    keep_words = sorted({r["label"] for r in csv.DictReader(open(args.split, encoding="utf-8"))})
    mapping = {c: i for i, c in enumerate(classes)}
    keep = np.array([mapping[w] for w in keep_words])

    print(f"\n  MODEL  {Path(args.run).name}")
    print(f"    trained on {len(classes)} words, head restricted to {len(keep)} at inference")
    print(f"    {config['n_frames']} frames, {config['img_size']} px crops, device {device}")

    # One confident clip and one the gate declines, chosen by running candidates.
    traces, seen = [], set()
    for row in rows:
        if row["label"] in seen:
            continue
        path = resolve(row["video_path"])
        if path is None:
            continue
        seen.add(row["label"])
        tr = trace(path, row["label"], model, config, classes, keep, device, args.crop_size)
        want_abstain = any(t["confidence"] < 0.40 for t in traces)
        if (tr["confidence"] >= 0.40 and not any(t["confidence"] >= 0.40 for t in traces)) or \
           (tr["confidence"] < 0.40 and not want_abstain):
            traces.append(tr)
        if len(traces) >= args.clips:
            break

    if args.figure and traces:
        figure(traces, classes, Path(args.figure))


if __name__ == "__main__":
    main()
