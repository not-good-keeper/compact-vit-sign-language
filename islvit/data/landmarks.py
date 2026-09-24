"""Extract hand and upper-body landmarks aligned to an existing crop cache.

``crops.py`` already runs MediaPipe Holistic on every sampled frame, gets 21
landmarks per hand plus a body pose, uses them to draw a crop box, and throws
them away. The model then has to re-derive handshape from a 64-pixel crop.
Landmarks are the handshape, measured directly: joint angles and fingertip
positions are exactly what distinguishes most signs, and at 42 points per frame
they cost a rounding error in parameters next to the pixel stream.

**Alignment is the whole contract.** The landmark for frame *t* of row *r* must
come from the same video frame as crop *t* of row *r*, or the model is shown a
hand pose from one moment next to pixels from another. So this reuses
``crops.resolve_video`` and ``crops.read_sampled_frames`` verbatim, walks the
crop cache's own ``index.csv``, and writes arrays indexed by the same row
numbers. It adds no frame logic of its own.

Detection follows the crop pipeline as well: Holistic first, then for a missing
hand the dedicated hand model on an upscaled pose ROI. The ROI stage is what
lifted crop detection rates, and dropping it here would give the landmark stream
worse coverage than the pixel stream it is meant to complement.

Layout (N rows of the crop cache, T frames)::

    hands.npy     float16 (N, T, 2, 21, 3)  x, y in frame-normalised coords; z relative depth
    hand_src.npy  uint8   (N, T, 2)         0 missing, 1 holistic, 2 ROI
    pose.npy      float16 (N, T, 7, 3)      nose, shoulders, elbows, wrists: x, y, visibility
    done.npy      bool    (N,)              resume marker

Usage::

    python -m islvit.data.landmarks --cache cache128_f32 --out cache_lm_f32 --frames 32
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp_proc
import os
import time
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import numpy as np

from islvit.data import crops

# Anatomical left/right, matching the crop streams' order.
POSE_POINTS = (0, 11, 12, 13, 14, 15, 16)  # nose, shoulders, elbows, wrists
N_HAND = 21
SRC_MISSING, SRC_DIRECT, SRC_ROI = 0, 1, 2


def hand_points_in_roi(frame: np.ndarray, roi, width: int, height: int):
    """Second-stage hand landmarks inside an upscaled ROI, mapped back to the frame."""
    import mediapipe as mp

    x0, y0, x1, y1 = roi
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    scale = crops.ROI_RESIZE / max(patch.shape[:2])
    enlarged = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(enlarged, cv2.COLOR_BGR2RGB))
    result = crops.get_hand_landmarker().detect(image)
    if not result.hand_landmarks:
        return None
    points = np.array([[p.x * enlarged.shape[1] / scale + x0, p.y * enlarged.shape[0] / scale + y0, p.z]
                       for p in result.hand_landmarks[0]], dtype=np.float32)
    # Same minimum-size rejection the crop pipeline applies to this stage.
    extent = max(np.ptp(points[:, 0]), np.ptp(points[:, 1])) * crops.HAND_MARGIN
    if extent < 8:
        return None
    points[:, 0] /= width
    points[:, 1] /= height
    return points


def extract(job: tuple[int, str]):
    """Landmarks for one cache row. Pool worker entry point."""
    row, video_path = job
    path = crops.resolve_video(video_path)
    if path is None:
        return row, None, None, None, "file not found"
    return (row, *extract_landmarks(path))


def extract_landmarks(path: Path):
    """(hands, source, pose, error) for one video file.

    Split out so inference (``islvit.predict``) runs exactly the code that built
    the training cache, on an arbitrary path -- the same reason
    ``crops.extract_clip`` exists.
    """
    frames = crops.read_sampled_frames(path, crops.FRAMES_PER_CLIP)
    if frames is None:
        return None, None, None, "cannot decode"
    # Fresh detectors per video: see crops.reset_detectors for why this matters.
    crops.reset_detectors()

    import mediapipe as mp

    count = len(frames)
    height, width = frames[0].shape[:2]
    hands = np.zeros((count, 2, N_HAND, 3), dtype=np.float32)
    source = np.zeros((count, 2), dtype=np.uint8)
    pose = np.zeros((count, len(POSE_POINTS), 3), dtype=np.float32)
    holistic = crops.get_holistic()
    try:
        for t, frame in enumerate(frames):
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = holistic.detect(image)
            if result.pose_landmarks:
                for k, index in enumerate(POSE_POINTS):
                    p = result.pose_landmarks[index]
                    pose[t, k] = (p.x, p.y, getattr(p, "visibility", 1.0))
            for s, side in enumerate(("left", "right")):
                landmarks = result.left_hand_landmarks if side == "left" else result.right_hand_landmarks
                if landmarks:
                    hands[t, s] = [(p.x, p.y, p.z) for p in landmarks]
                    source[t, s] = SRC_DIRECT
                    continue
                roi = crops.pose_roi(result.pose_landmarks, side, width, height)
                points = hand_points_in_roi(frame, roi, width, height) if roi else None
                if points is not None:
                    hands[t, s] = points
                    source[t, s] = SRC_ROI
    except RuntimeError as error:
        return None, None, None, f"mediapipe: {error}"
    return hands, source, pose, ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Landmarks aligned to a crop cache")
    parser.add_argument("--cache", default="cache128_f32", help="crop cache whose index.csv defines the rows")
    parser.add_argument("--out", default="cache_lm_f32")
    parser.add_argument("--frames", type=int, default=32, help="must equal the crop cache's frame count")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split-file", default=None, help="only extract clips in this split file...")
    parser.add_argument("--split", default=None, help="...and in this split (e.g. test)")
    args = parser.parse_args()

    cache, out = Path(args.cache), Path(args.out)
    crop_frames = np.load(cache / "crops.npy", mmap_mode="r").shape[1]
    assert crop_frames == args.frames, f"crop cache has {crop_frames} frames, asked for {args.frames}"
    with (cache / "index.csv").open(encoding="utf-8") as handle:
        rows = [(int(r["row"]), r["video_path"]) for r in csv.DictReader(handle) if r["cached"] == "1"]
    n = max(r for r, _ in rows) + 1
    if args.split_file:
        with Path(args.split_file).open(encoding="utf-8") as handle:
            wanted = {r["video_path"] for r in csv.DictReader(handle)
                      if args.split is None or r["split"] == args.split}
        rows = [job for job in rows if job[1] in wanted]
        print(f"restricted to {len(rows)} clips from {args.split_file} ({args.split or 'all splits'})")

    out.mkdir(exist_ok=True)
    fmt = np.lib.format

    def array(name, shape, dtype):
        path = out / name
        if path.exists():
            existing = fmt.open_memmap(path, mode="r+")
            assert existing.shape == shape, f"{name}: {existing.shape} on disk, expected {shape}"
            return existing
        return fmt.open_memmap(path, mode="w+", dtype=dtype, shape=shape)

    hands = array("hands.npy", (n, args.frames, 2, N_HAND, 3), np.float16)
    source = array("hand_src.npy", (n, args.frames, 2), np.uint8)
    pose = array("pose.npy", (n, args.frames, len(POSE_POINTS), 3), np.float16)
    done = array("done.npy", (n,), np.bool_)
    (out / "SOURCE_CACHE").write_text(f"{cache.resolve()}\n{args.frames}\n", encoding="utf-8")

    remaining = [job for job in rows if not done[job[0]]]
    pending = remaining[: args.limit]
    print(f"{len(rows)} cached clips, {len(rows) - len(remaining)} already done, {len(pending)} to extract")
    start, failures = time.time(), []
    with mp_proc.Pool(processes=args.workers, initializer=crops.init_worker,
                      initargs=(crops.CROP_SIZE, "include", args.frames)) as pool:
        for i, (row, h, s, p, error) in enumerate(pool.imap_unordered(extract, pending, chunksize=2), 1):
            if error:
                failures.append((row, error))
            else:
                hands[row], source[row], pose[row] = h, s, p
                done[row] = True
            if i % 100 == 0 or i == len(pending):
                for a in (hands, source, pose, done):
                    a.flush()
                rate = i / (time.time() - start)
                print(f"  {i}/{len(pending)}  {rate:.2f} clips/s  eta {(len(pending) - i) / rate / 60:.0f} min  "
                      f"failures {len(failures)}", flush=True)
    for a in (hands, source, pose, done):
        a.flush()

    finished = np.flatnonzero(done[:])
    cover = (source[finished] > 0).mean(axis=(0, 1))
    print(f"done {len(finished)}/{len(rows)}; hand coverage left {cover[0]:.1%} right {cover[1]:.1%}")
    for row, error in failures[:10]:
        print(f"  failed row {row}: {error}")


if __name__ == "__main__":
    main()
