"""Point the model at a video file and get ranked sign predictions.

Deliberately reuses ``islvit.data.crops.extract_clip`` and
``islvit.data.dataset.prepare_clip`` -- the exact functions that built the
training cache. Writing a separate inference preprocessor is the classic way a
model that scores well offline fails on a real video: a different resize, colour
order, or crop margin silently shifts the input distribution. Sharing the code
makes that impossible rather than merely unlikely.

Usage::

    python -m islvit.predict --run runs/vocab50_s0 --video clip.mp4
    python -m islvit.predict --run runs/vocab50_s0 --video clip.mp4 --no-tta --topk 10

Scope, stated plainly: this classifies ONE isolated sign per video. It does not
segment continuous signing and does not translate to English -- see
``islvit.slt`` for the translation scaffold. Accuracy is only established for
INCLUDE-like recording conditions (§11.6 of the report measures ~1 % on a
different corpus), so treat predictions on unfamiliar footage as unvalidated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from islvit.data import crops as crops_module
from islvit.data.dataset import SOURCE_DIRECT, SOURCE_ROI, prepare_clip
from islvit.eval import load_run

# Views for test-time augmentation: three temporal phases x horizontal flip.
# Measured at +1.6 to +5.1 points on every checkpoint tried, and free at
# inference beyond the extra forward passes.
OFFSETS = (0.25, 0.5, 0.75)
FLIPS = (False, True)


def extract(video: Path, crop_size: int, frames: int = 32):
    """Run the cache-building detection path on an arbitrary file.

    ``frames`` must match the depth of the cache the model was trained from, not
    the number of frames the model consumes. Training samples 16 of 32, which is
    what gives test-time augmentation distinct temporal phases to average -- worth
    a measured +3.2 points. Extracting only 16 here (the module default, which this
    function used to inherit) made every TTA view land on the same frames, so the
    live path quietly lost a gain the offline benchmark still reported.
    """
    crops_module.init_worker(crop_size, "include", frames)
    clip, sources, geometry, error = crops_module.extract_clip(video)
    if clip is None:
        raise SystemExit(f"could not process {video}: {error}")
    detected = (sources == SOURCE_DIRECT) | (sources == SOURCE_ROI)
    return clip, detected, geometry.astype(np.float32), sources


def extract_landmarks(video: Path, frames: int = 32) -> dict:
    """The landmark cache's own extraction path, on an arbitrary file.

    This runs MediaPipe a second time over the same frames that ``extract``
    already processed. Detection is deterministic in IMAGE mode, so the result is
    identical to a single combined pass; it costs latency, not correctness, and
    folding the two passes together is a later optimisation.
    """
    from islvit.data.landmarks import extract_landmarks as run

    crops_module.init_worker(crops_module.CROP_SIZE, "include", frames)
    hands, source, pose, error = run(video)
    if hands is None:
        raise SystemExit(f"could not extract landmarks from {video}: {error}")
    return {"hands": hands.astype(np.float32), "hand_present": source > 0,
            "pose": pose.astype(np.float32)}


def views(clip, detected, geometry, n_frames, img_size, use_tta, landmarks=None):
    """Yield (crops, detected, geometry, extras) for each augmentation view.

    ``extras`` holds the landmark tensors for models trained with them, sampled at
    the same frames and mirrored by the same flip as the pixels; it is {} for a
    pixel-only model, so ``model(crops, det, geo, **extras)`` works for both.
    """
    combos = [(o, f) for f in FLIPS for o in OFFSETS] if use_tta else [(0.5, False)]
    edges = np.linspace(0, clip.shape[0], n_frames + 1)
    for offset, flip in combos:
        picks = np.clip(
            (edges[:-1] + offset * (edges[1:] - edges[:-1])).astype(int), 0, clip.shape[0] - 1
        )
        extras = {key: value[picks].copy() for key, value in landmarks.items()} if landmarks else None
        prepared = prepare_clip(
            clip[picks],
            detected[picks],
            geometry[picks],
            img_size=img_size,
            train=False,
            flip_prob=1.0 if flip else 0.0,
            extras=extras,
        )
        tensors = tuple(torch.from_numpy(a).unsqueeze(0) for a in prepared)
        batch = {key: torch.from_numpy(np.ascontiguousarray(value)).unsqueeze(0)
                 for key, value in (extras or {}).items()}
        yield (*tensors, batch)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the sign in a video")
    parser.add_argument("--run", type=str, required=True, help="a runs/<tag> directory")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--no-tta", dest="tta", action="store_false", help="single centre view only")
    parser.add_argument("--crop-size", type=int, default=128, help="must match the training cache")
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"{video} not found")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config, classes = load_run(Path(args.run), device)

    clip, detected, geometry, sources = extract(video, args.crop_size)
    landmarks = extract_landmarks(video) if config.get("landmarks") else None
    # Detection quality is the single best predictor of whether the prediction
    # means anything: with no hands found, the model is classifying background.
    hands = sources[:, :2]
    genuine = float(((hands == SOURCE_DIRECT) | (hands == SOURCE_ROI)).mean())

    total = None
    with torch.no_grad():
        for crops, det, geo, extras in views(
            clip, detected, geometry, config["n_frames"], config["img_size"], args.tta, landmarks
        ):
            extras = {key: value.to(device) for key, value in extras.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(crops.to(device), det.to(device), geo.to(device), **extras)
            probability = logits.float().softmax(1)
            total = probability if total is None else total + probability
    probability = (total / (len(OFFSETS) * len(FLIPS) if args.tta else 1)).squeeze(0).cpu()

    top = probability.topk(min(args.topk, len(classes)))
    print(f"\n{video.name}")
    print(f"  model {Path(args.run).name}  |  {len(classes)} signs  |  "
          f"{'6-view TTA' if args.tta else 'single view'}")
    print(f"  hand detection: {genuine:.0%} of frames", end="")
    if genuine < 0.25:
        print("  <- LOW: hands were rarely found, so this prediction is unreliable")
    else:
        print()
    print()
    for rank, (score, index) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
        bar = "#" * int(round(score * 40))
        print(f"  {rank}. {classes[index]:<28s} {score:6.1%}  {bar}")

    if probability.max() < 0.2:
        print("\n  (no confident prediction -- the top score is low, which usually means")
        print("   the sign is outside this model's vocabulary or the clip is unclear)")


if __name__ == "__main__":
    main()
