"""Build the high-resolution patch banks the super-resolution GAN learns from.

Two kinds of 128x128 RGB patch, at the 720p working resolution the detector sees:

* **Hand and face crops** from ``cache128_f32``, genuine detections only (direct
  or ROI-rescued boxes). This is the content the detector and the classifier
  depend on, so it is what the generator must get right.
* **Random patches from whole frames**, decoded through ``crops.read_sampled_frames``
  exactly as extraction decodes them. The generator runs on whole frames, and the
  body, face and background it sees there matter to pose detection too.

**Leak guard.** The training bank draws only from clips marked ``train`` in
``splits/full263__session-disjoint-foldval.csv`` and asserts that none of them is
a test, unused or 472-deployed-test clip. The held-out bank draws only from the
session-disjoint test sessions and is used to measure image quality, never to train.

Usage::

    python -m islvit.sr.patches --split train --out C:/isl_gan_work/hr_train.npy
    python -m islvit.sr.patches --split heldout --crops 600 --frames-videos 75 --out C:/isl_gan_work/hr_heldout.npy
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp_proc
from pathlib import Path

import numpy as np

from islvit.data import crops
from islvit.data.dataset import load_cache_index

SPLIT = Path("splits/full263__session-disjoint-foldval.csv")
TEST472 = Path("splits/_test472_paths.csv")
PATCH = 128


def split_paths(kind: str) -> list[str]:
    with SPLIT.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with TEST472.open(encoding="utf-8") as handle:
        deployed_test = {row["video_path"] for row in csv.DictReader(handle)}
    train = {row["video_path"] for row in rows if row["split"] == "train"}
    held = {row["video_path"] for row in rows if row["split"] in ("test", "unused")}
    # The guarantee this module exists to keep: no clip the classifier is tested
    # on, and no clip from a held-out session, ever reaches the training bank.
    assert not train & held, "train and held-out rows overlap in the split file"
    assert not train & deployed_test, "a deployed test clip is marked train"
    return sorted(train if kind == "train" else held)


def crop_patches(paths: list[str], count: int, rng: np.random.Generator, cache: Path) -> np.ndarray:
    index = load_cache_index(cache)
    rows = np.array([index[p] for p in paths if p in index])
    sources = np.load(cache / "sources.npy", mmap_mode="r")
    crops_cache = np.load(cache / "crops.npy", mmap_mode="r")
    frames, streams = sources.shape[1], sources.shape[2]
    picked: list[tuple[int, int, int]] = []
    tries = 0
    while len(picked) < count and tries < count * 20:
        tries += 1
        row, t, s = int(rng.choice(rows)), int(rng.integers(frames)), int(rng.integers(streams))
        if sources[row, t, s] in (crops.SOURCE_DIRECT, crops.SOURCE_ROI):
            picked.append((row, t, s))
    picked.sort()  # sequential memmap reads
    out = np.empty((len(picked), PATCH, PATCH, 3), np.uint8)
    for i, (row, t, s) in enumerate(picked):
        patch = np.asarray(crops_cache[row, t, s])
        if patch.shape[0] != PATCH:
            import cv2
            patch = cv2.resize(patch, (PATCH, PATCH), interpolation=cv2.INTER_AREA)
        out[i] = patch
    return out


def _frame_job(job):
    video_path, per_frame, frames, seed = job
    path = crops.resolve_video(video_path)
    if path is None:
        return []
    sampled = crops.read_sampled_frames(path, frames)
    if not sampled:
        return []
    rng = np.random.default_rng(seed)
    patches = []
    for frame in sampled:
        # Only sample inside the picture, never the letterbox bars.
        columns = np.flatnonzero(frame.max(axis=(0, 2)) > 8)
        if columns.size < PATCH:
            continue
        left, right = int(columns[0]), int(columns[-1]) + 1
        for _ in range(per_frame):
            x = int(rng.integers(left, max(left + 1, right - PATCH)))
            y = int(rng.integers(0, frame.shape[0] - PATCH + 1))
            patches.append(np.ascontiguousarray(frame[y:y + PATCH, x:x + PATCH, ::-1]))  # BGR -> RGB
    return patches


def frame_patches(paths: list[str], videos: int, rng: np.random.Generator, workers: int,
                  frames: int = 4, per_frame: int = 4) -> np.ndarray:
    chosen = rng.choice(paths, size=min(videos, len(paths)), replace=False)
    jobs = [(p, per_frame, frames, int(rng.integers(1 << 31))) for p in chosen]
    out: list[np.ndarray] = []
    # A fresh worker sees the module defaults: no degradation, no upscaler.
    with mp_proc.Pool(workers, initializer=crops.init_worker, initargs=(crops.CROP_SIZE, "include", None)) as pool:
        for i, patches in enumerate(pool.imap_unordered(_frame_job, jobs, chunksize=4), 1):
            out.extend(patches)
            if i % 200 == 0:
                print(f"  frames: {i}/{len(jobs)} videos, {len(out)} patches", flush=True)
    return np.stack(out) if out else np.empty((0, PATCH, PATCH, 3), np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="High-resolution patch bank for the SR GAN")
    parser.add_argument("--split", choices=("train", "heldout"), default="train")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache", default="cache128_f32")
    parser.add_argument("--crops", type=int, default=16000, help="hand/face crop patches")
    parser.add_argument("--frames-videos", type=int, default=1000, help="videos to draw whole-frame patches from")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-file", default=None,
                        help="split file whose 'train' rows are drawn from (default: the folded-val file). "
                             "splits/full263__session-disjoint.csv keeps its val sessions out of the bank, "
                             "so they can serve as the development set")
    args = parser.parse_args()
    global SPLIT
    if args.split_file:
        SPLIT = Path(args.split_file)

    rng = np.random.default_rng(args.seed + (0 if args.split == "train" else 1))
    paths = split_paths(args.split)
    print(f"{args.split}: {len(paths)} clips")
    a = crop_patches(paths, args.crops, rng, Path(args.cache))
    print(f"  {len(a)} hand/face crop patches")
    b = frame_patches(paths, args.frames_videos, rng, args.workers)
    print(f"  {len(b)} whole-frame patches")

    bank = np.concatenate([a, b])
    order = rng.permutation(len(bank))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    memmap = np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8, shape=bank.shape)
    memmap[:] = bank[order]
    memmap.flush()
    meta = {"split": args.split, "split_file": SPLIT.as_posix(), "clips": len(paths), "crop_patches": len(a),
            "frame_patches": len(b), "patch": PATCH, "seed": args.seed, "cache": args.cache}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out} {bank.shape} ({bank.nbytes / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
