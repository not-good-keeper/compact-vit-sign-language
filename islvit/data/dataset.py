"""Torch Dataset over the cached INCLUDE crops.

The cache holds 16 frames per clip at 80x80; training samples ``n_frames`` of
those with temporal jitter and random-crops 80 -> 64. Keeping the cache slightly
larger than the model input is what makes spatial and temporal augmentation free
at train time -- no decoding, no MediaPipe, just indexing into a memmap.

The whole cache is ~3.9 GB, so it sits in the OS page cache after the first epoch
and the GPU stops waiting on data entirely.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


import os

# Overridable so a re-extraction at a different crop size can be built and
# evaluated without disturbing runs reading the current cache.
CACHE_DIR = Path(os.environ.get("ISLVIT_CACHE", "cache"))
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Mirrors islvit.data.crops box-source codes.
SOURCE_MISSING, SOURCE_DIRECT, SOURCE_ROI, SOURCE_INTERP = 0, 1, 2, 3

# Stream order is fixed by crops.py: left hand, right hand, face. A horizontal
# flip turns a right-dominant signer into a left-dominant one, which only stays
# physically coherent if the two hand streams swap with it.
FLIP_PERMUTATION = (1, 0, 2)


# Where each corpus's crops live. A split file may carry a `corpus` column when
# clips come from more than one source (see islvit.data.merge_cislr); rows without
# one are INCLUDE, which keeps every pre-existing split file working unchanged.
# Per-corpus caches, each overridable by environment variable. The CISLR one has
# to be switchable because frame depth must match across corpora in a single
# split: a 16-frame CISLR clip mixed into a 32-frame INCLUDE run gets sampled with
# no temporal jitter while every INCLUDE clip gets 2x headroom, which silently
# makes the two corpora different augmentation regimes rather than different data.
CORPUS_CACHES = {
    "cislr": Path(os.environ.get("ISLVIT_CACHE_CISLR", "cache_cislr")),
    "custom": Path(os.environ.get("ISLVIT_CACHE_CUSTOM", "cache_custom")),
    "isign": Path(os.environ.get("ISLVIT_CACHE_ISIGN", "cache_isign")),
}


def cache_dir_for(corpus: str) -> Path:
    return CORPUS_CACHES.get(corpus, CACHE_DIR)


# Hand and pose landmarks, row-aligned to one specific crop cache (see
# islvit.data.landmarks). INCLUDE only: no other corpus has them yet.
LANDMARK_CACHE = Path(os.environ.get("ISLVIT_CACHE_LM", "cache_lm_f32"))
# Pose order is nose, L/R shoulder, L/R elbow, L/R wrist; a mirror swaps each pair.
POSE_FLIP = (0, 2, 1, 4, 3, 6, 5)
HAND_FLIP = (1, 0)


def interpolate_hands(hands: np.ndarray, present: np.ndarray, max_gap: int = 4):
    """Fill short interior gaps in each hand's joints by linear interpolation.

    hands (T,2,21,3), present (T,2) bool, over the *cached* frame axis -- this runs
    before frame sampling, so a gap is measured in cache frames. The detector
    misses a hand on 9-15 % of frames, usually for one or two frames mid-motion;
    leaving those as zeros tells the model the hand vanished. Only gaps bounded by
    detections on both sides and at most ``max_gap`` long are filled -- a hand that
    is genuinely off-screen at the start or end stays missing. This mirrors what
    crops.fill_gaps does for crop boxes.

    Shared by training and inference (islvit.predict), so both fill identically.
    """
    hands, present = hands.copy(), present.copy()
    for side in range(hands.shape[1]):
        seen = np.flatnonzero(present[:, side])
        for a, b in zip(seen[:-1], seen[1:]):
            gap = b - a - 1
            if 0 < gap <= max_gap:
                for k in range(1, gap + 1):
                    w = k / (gap + 1)
                    hands[a + k, side] = (1 - w) * hands[a, side] + w * hands[b, side]
                present[a + 1:b, side] = True
    return hands, present


def augment_landmarks(extras: dict, strength: float, rng=np.random) -> None:
    """One random similarity-plus-shear per clip, applied in place, train only.

    The pixel stream gets crop jitter, colour, blur and erasing; the landmark
    stream got nothing, so every training epoch showed it the same joints. This
    rotates (+-10 deg), scales (+-10 %) and shears (+-0.1) all joints and the pose
    together about the shoulder midpoint -- a stand-in for camera angle, distance
    and signer build -- and adds small per-joint noise for detector jitter.
    Magnitudes scale with ``strength``. Work is done in isotropic units because
    frames are 16:9; rotating normalised coordinates directly would also shear.
    """
    aspect = 1280 / 720
    hands, pose, present = extras["hands"], extras["pose"], extras["hand_present"]
    seen = pose[..., 1:3, 2].min(axis=-1) > 0.3
    if seen.any():
        t = int(np.flatnonzero(seen)[0])
        centre = pose[t, 1:3, :2].mean(axis=0)
    else:
        centre = np.array([0.5, 0.5], dtype=np.float32)
    theta = np.deg2rad(rng.uniform(-10, 10) * strength)
    scale = np.exp(rng.uniform(-0.1, 0.1) * strength)
    shear = rng.uniform(-0.1, 0.1) * strength
    rotate = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    matrix = (scale * rotate @ np.array([[1.0, shear], [0.0, 1.0]])).astype(np.float32)

    def transform(xy):
        iso = (xy - centre) * np.array([aspect, 1.0], dtype=np.float32)
        return (iso @ matrix.T) / np.array([aspect, 1.0], dtype=np.float32) + centre

    moved = transform(hands[..., :2])
    moved = moved + rng.normal(0.0, 0.002 * strength, size=moved.shape).astype(np.float32)
    # Missing hands are zeros and must stay zeros, not become a transformed origin.
    hands[..., :2] = np.where(present[..., None, None], moved, hands[..., :2])
    pose[..., :2] = np.where(pose[..., 2:3] > 0, transform(pose[..., :2]), pose[..., :2])


def check_landmark_alignment(crop_cache: Path, landmark_cache: Path = LANDMARK_CACHE) -> None:
    """Refuse landmarks extracted against a different crop cache.

    Row r of the landmark arrays is only meaningful next to row r of the crop
    cache it was built from, sampled at the same frame count. A 16-frame crop
    cache paired with 32-frame landmarks -- or a re-extracted cache whose row
    order moved -- would train without error on landmarks from the wrong frames,
    or the wrong video.
    """
    marker = landmark_cache / "SOURCE_CACHE"
    if not marker.exists():
        raise FileNotFoundError(f"{marker} missing -- run `python -m islvit.data.landmarks` first")
    source, frames = marker.read_text(encoding="utf-8").split()[:2]
    if Path(source).resolve() != crop_cache.resolve():
        raise ValueError(f"landmarks in {landmark_cache} were built from {source}, not {crop_cache}")
    crop_frames = np.load(crop_cache / "crops.npy", mmap_mode="r").shape[1]
    if int(frames) != crop_frames:
        raise ValueError(f"landmarks have {frames} frames per clip, crop cache has {crop_frames}")


def load_cache_index(cache_dir: Path | None = None) -> dict[str, int]:
    """video_path -> row in the crop cache, for rows that actually got cached."""
    index_path = (cache_dir or CACHE_DIR) / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} missing -- run `python -m islvit.data.crops` first")

    mapping: dict[str, int] = {}
    with index_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["cached"]):
                mapping[row["video_path"]] = int(row["row"])
    return mapping


class IncludeCrops(Dataset):
    def __init__(
        self,
        split_file: str | Path,
        split: str,
        n_frames: int = 8,
        img_size: int = 64,
        train: bool = False,
        flip_prob: float = 0.5,
        color_jitter: float = 0.2,
        stream_dropout: float = 0.1,
        grayscale_prob: float = 0.0,
        crop_scale: float = 0.8,
        resolution_jitter: float = 0.0,
        speed_jitter: float = 0.0,
        random_erasing: float = 0.0,
        label_to_index: dict[str, int] | None = None,
        landmarks: bool = False,
        lm_interp: bool = False,
        lm_aug: float = 0.0,
    ) -> None:
        self.crop_scale = crop_scale
        self.n_frames = n_frames
        self.img_size = img_size
        self.train = train
        self.flip_prob = flip_prob if train else 0.0
        self.color_jitter = color_jitter if train else 0.0
        self.stream_dropout = stream_dropout if train else 0.0
        self.grayscale_prob = grayscale_prob if train else 0.0
        self.resolution_jitter = resolution_jitter if train else 0.0
        self.speed_jitter = speed_jitter if train else 0.0
        self.random_erasing = random_erasing if train else 0.0
        # Eval-time knobs, left at the plain-evaluation defaults. islvit.tta varies
        # them to build several deterministic views of the same clip.
        self.eval_offset = 0.5
        self.eval_flip = False

        with Path(split_file).open(encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["split"] == split]

        corpora = sorted({row.get("corpus") or "include" for row in rows})
        indices = {name: load_cache_index(cache_dir_for(name)) for name in corpora}

        def cached(row: dict) -> bool:
            return row["video_path"] in indices[row.get("corpus") or "include"]

        missing = [row["video_path"] for row in rows if not cached(row)]
        rows = [row for row in rows if cached(row)]
        if missing:
            print(f"  [{split}] {len(missing)} videos absent from the crop cache, skipped")
        if len(corpora) > 1:
            counts = {name: sum(1 for row in rows if (row.get("corpus") or "include") == name) for name in corpora}
            print(f"  [{split}] corpora: {counts}")

        if label_to_index is None:
            with Path(split_file).open(encoding="utf-8") as handle:
                labels = sorted({row["label"] for row in csv.DictReader(handle)})
            label_to_index = {label: index for index, label in enumerate(labels)}
        self.label_to_index = label_to_index
        self.classes = sorted(label_to_index, key=label_to_index.get)

        self.corpora = corpora
        self.row_corpus = [row.get("corpus") or "include" for row in rows]
        self.rows = np.array(
            [indices[corpus][row["video_path"]] for corpus, row in zip(self.row_corpus, rows)], dtype=np.int64
        )
        self.labels = np.array([label_to_index[row["label"]] for row in rows], dtype=np.int64)
        self.video_paths = [row["video_path"] for row in rows]
        self.take_groups = [row["take_group"] for row in rows]

        self._memmaps: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        self.landmarks = landmarks
        self.lm_interp = lm_interp
        self.lm_aug = lm_aug if train else 0.0
        self._landmarks = None
        if landmarks:
            # An empty split (val, when it has been folded into train) has no corpus.
            if corpora and corpora != ["include"]:
                raise ValueError(f"landmarks exist for INCLUDE only; this split mixes {corpora}")
            check_landmark_alignment(cache_dir_for("include"))
            done = np.load(LANDMARK_CACHE / "done.npy")
            if not done[self.rows].all():
                raise ValueError(f"{int((~done[self.rows]).sum())} clips in this split have no landmarks yet")

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def n_classes(self) -> int:
        return len(self.label_to_index)

    def _cache(self, corpus: str = "include") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Opened lazily so each DataLoader worker gets its own memmap handle
        # rather than inheriting one through pickling.
        if corpus not in self._memmaps:
            directory = cache_dir_for(corpus)
            self._memmaps[corpus] = tuple(
                np.load(directory / name, mmap_mode="r")
                for name in ("crops.npy", "sources.npy", "geometry.npy")
            )
        return self._memmaps[corpus]

    def _sample_frame_indices(self, available: int) -> np.ndarray:
        """Uniform segments, with a random offset inside each at train time.

        At eval time the offset is fixed, normally to the segment centre. It is
        settable so test-time augmentation can take several deterministic passes
        at different phases of the clip (see islvit.tta) without reintroducing the
        randomness that would make a single evaluation irreproducible.
        """
        start, stop = 0.0, float(available)
        if self.speed_jitter > 0:
            # Sample a sub-span of the clip and spread the frames across it,
            # which reads as the same sign performed faster (short span) or as a
            # partial view of a slower one. Signing tempo varies a lot between
            # people and is exactly the nuisance factor a session-disjoint test
            # set punishes, but nothing in the pipeline varied it before: every
            # clip was always sampled edge to edge.
            span = available * (1.0 - np.random.uniform(0.0, self.speed_jitter))
            start = np.random.uniform(0.0, available - span)
            stop = start + span
        edges = np.linspace(start, stop, self.n_frames + 1)
        if self.train:
            offsets = np.random.rand(self.n_frames)
        else:
            offsets = np.full(self.n_frames, self.eval_offset)
        picks = edges[:-1] + offsets * (edges[1:] - edges[:-1])
        return np.clip(picks.astype(int), 0, available - 1)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        crops_cache, sources_cache, geometry_cache = self._cache(self.row_corpus[index])
        row = self.rows[index]

        clip = np.asarray(crops_cache[row])                      # (T_cache, S, 80, 80, 3) uint8
        sources = np.asarray(sources_cache[row])                 # (T_cache, S) uint8
        geometry = np.asarray(geometry_cache[row]).astype(np.float32)  # (T_cache, S, 3)

        # Only a real detection counts as reliable. An interpolated box is a stale
        # box carried from a neighbouring frame -- it usually shows background the
        # hand has already left, so the model is told not to trust it.
        detected = (sources == SOURCE_DIRECT) | (sources == SOURCE_ROI)

        frame_indices = self._sample_frame_indices(clip.shape[0])
        clip = clip[frame_indices]
        detected = detected[frame_indices]
        geometry = geometry[frame_indices]

        extras = None
        if self.landmarks:
            if self._landmarks is None:
                self._landmarks = tuple(np.load(LANDMARK_CACHE / name, mmap_mode="r")
                                        for name in ("hands.npy", "hand_src.npy", "pose.npy"))
            hands, hand_src, pose = self._landmarks
            clip_hands = np.asarray(hands[row]).astype(np.float32)
            clip_present = np.asarray(hand_src[row]) > 0
            if self.lm_interp:
                # On the full cached sequence, before sampling, as at inference.
                clip_hands, clip_present = interpolate_hands(clip_hands, clip_present)
            # Same frame indices as the crops: this is the alignment that matters.
            extras = {
                "hands": clip_hands[frame_indices],
                "hand_present": clip_present[frame_indices],
                "pose": np.asarray(pose[row])[frame_indices].astype(np.float32),
            }
            if self.lm_aug > 0:
                augment_landmarks(extras, self.lm_aug)

        crops, detected, geometry = prepare_clip(
            clip,
            detected,
            geometry,
            img_size=self.img_size,
            train=self.train,
            crop_scale=self.crop_scale,
            color_jitter=self.color_jitter,
            grayscale_prob=self.grayscale_prob,
            flip_prob=1.0 if self.eval_flip else self.flip_prob,
            stream_dropout=self.stream_dropout,
            resolution_jitter=self.resolution_jitter,
            random_erasing=self.random_erasing,
            extras=extras,
        )

        item = {
            "crops": torch.from_numpy(crops),
            "detected": torch.from_numpy(detected),
            "geometry": torch.from_numpy(geometry),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
        }
        if extras is not None:
            item.update({key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in extras.items()})
        return item


def prepare_clip(
    clip: np.ndarray,
    detected: np.ndarray,
    geometry: np.ndarray,
    *,
    img_size: int,
    train: bool,
    crop_scale: float = 0.8,
    color_jitter: float = 0.0,
    grayscale_prob: float = 0.0,
    flip_prob: float = 0.0,
    stream_dropout: float = 0.0,
    resolution_jitter: float = 0.0,
    random_erasing: float = 0.0,
    extras: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop, augment and normalise one already frame-sampled clip.

    Shared by supervised training and masked pretraining so the two cannot drift
    apart -- a pretraining input distribution that differs from the finetuning one
    silently wastes the pretraining.

    ``clip`` is (T, S, cache_px, cache_px, 3) uint8; returns (T, S, 3, H, W) float32
    alongside the matching detected and geometry arrays.

    ``extras`` holds landmark arrays, updated in place with the *same* flip and
    stream-dropout draws as the pixels. Drawing them separately would mirror the
    crops while leaving the hand joints unmirrored on half the batches.
    """
    n_frames, n_streams, source_size = clip.shape[0], clip.shape[1], clip.shape[2]
    output = np.empty((n_frames, n_streams, img_size, img_size, 3), dtype=np.uint8)

    for stream in range(n_streams):
        # One window per stream per clip, held across time: this models the crop
        # box sitting slightly off, not the hand teleporting each frame. Window
        # size is sampled independently of img_size so the model input resolution
        # (and hence its token count) can be changed without re-extracting the
        # cache.
        if train:
            window = int(round(source_size * np.random.uniform(crop_scale, 1.0)))
            span = source_size - window
            top, left = (np.random.randint(0, span + 1, size=2) if span > 0 else (0, 0))
        else:
            window = int(round(source_size * (1 + crop_scale) / 2))
            top = left = (source_size - window) // 2

        patch = clip[:, stream, top : top + window, left : left + window]
        if window == img_size:
            output[:, stream] = patch
        else:
            interpolation = cv2.INTER_AREA if window > img_size else cv2.INTER_LINEAR
            for frame in range(patch.shape[0]):
                output[frame, stream] = cv2.resize(
                    patch[frame], (img_size, img_size), interpolation=interpolation
                )

    if resolution_jitter > 0 and np.random.rand() < resolution_jitter:
        # Degrade to a random lower resolution and back, which is exactly how a
        # CISLR crop is produced: its hands are ~78 source px upscaled to 128,
        # against INCLUDE's ~173 downscaled, leaving CISLR 2.1x blurrier by
        # variance-of-Laplacian. Training only on sharp crops makes that blur an
        # out-of-distribution shift, so pretrained CISLR features have to be
        # unlearned during finetuning instead of reused.
        #
        # One factor per clip, not per frame: a clip whose sharpness flickered
        # frame to frame would be an artefact no camera produces, and the
        # temporal stage would be free to key on the flicker itself.
        factor = np.random.uniform(0.35, 0.9)
        small = max(8, int(round(img_size * factor)))
        for frame in range(n_frames):
            for stream in range(n_streams):
                shrunk = cv2.resize(output[frame, stream], (small, small), interpolation=cv2.INTER_AREA)
                output[frame, stream] = cv2.resize(
                    shrunk, (img_size, img_size), interpolation=cv2.INTER_LINEAR
                )

    crops = output.astype(np.float32) / 255.0

    if color_jitter > 0:
        jitter = color_jitter
        # Brightness and contrast about the clip's own mean.
        scale = 1.0 + np.random.uniform(-jitter, jitter)
        shift = np.random.uniform(-jitter, jitter) * 0.5
        crops = crops * scale + shift
        # Per-channel gain stands in for white balance and skin tone. The
        # take-group split holds out whole recording sessions, so lighting and
        # signer appearance are exactly what the model must not key on.
        crops = crops * (1.0 + np.random.uniform(-jitter, jitter, size=3).astype(np.float32))
        crops = np.clip(crops, 0.0, 1.0)

    if grayscale_prob > 0 and np.random.rand() < grayscale_prob:
        luma = (crops * np.array([0.299, 0.587, 0.114], dtype=np.float32)).sum(-1, keepdims=True)
        crops = np.repeat(luma, 3, axis=-1)

    if flip_prob > 0 and np.random.rand() < flip_prob:
        crops = crops[:, list(FLIP_PERMUTATION)][:, :, :, ::-1]
        detected = detected[:, list(FLIP_PERMUTATION)]
        # Mirroring the image has to mirror the geometry with it, or the model is
        # told a left-side hand is on the right.
        geometry = geometry[:, list(FLIP_PERMUTATION)]
        geometry[:, :, 0] = 1.0 - geometry[:, :, 0]
        if extras is not None:
            extras["hands"] = extras["hands"][:, list(HAND_FLIP)]
            extras["hands"][..., 0] = 1.0 - extras["hands"][..., 0]
            extras["hand_present"] = extras["hand_present"][:, list(HAND_FLIP)]
            extras["pose"] = extras["pose"][:, list(POSE_FLIP)]
            extras["pose"][..., 0] = 1.0 - extras["pose"][..., 0]

    crops = (crops - IMAGENET_MEAN) / IMAGENET_STD

    if random_erasing > 0:
        # Applied after normalisation so filling with 0 means "the mean pixel"
        # rather than "black", which would itself be a strong out-of-distribution
        # signal the temporal stage could key on.
        #
        # One box per stream, held across time, for the same reason the crop
        # window is: an occlusion that jumped around every frame is an artefact
        # no camera produces. Occluding a whole stream for the whole clip is
        # already covered by stream_dropout; this is the partial case, which is
        # what a hand leaving frame or crossing the body actually looks like.
        for stream in range(n_streams):
            if np.random.rand() >= random_erasing:
                continue
            area = img_size * img_size * np.random.uniform(0.02, 0.25)
            ratio = np.exp(np.random.uniform(np.log(0.3), np.log(3.3)))
            height = min(img_size, int(round(np.sqrt(area * ratio))))
            width = min(img_size, int(round(np.sqrt(area / ratio))))
            if height < 1 or width < 1:
                continue
            top = np.random.randint(0, img_size - height + 1)
            left = np.random.randint(0, img_size - width + 1)
            crops[:, stream, top : top + height, left : left + width] = 0.0

    crops = np.ascontiguousarray(crops.transpose(0, 1, 4, 2, 3))  # (T, S, 3, H, W)
    detected = detected.copy()

    if stream_dropout > 0:
        # Occasionally blank a whole stream so the model cannot become dependent
        # on, say, the face always being present.
        drop = np.random.rand(n_streams) < stream_dropout
        if drop.any() and not drop.all():
            crops[:, drop] = 0.0
            detected[:, drop] = False
            geometry[:, drop] = 0.0
            if extras is not None:
                # A dropped hand stream loses its joints too, or the landmark
                # branch would hand back exactly what the dropout took away. The
                # body pose is not the face stream, so a dropped face keeps it.
                extras["hand_present"] = extras["hand_present"].copy()
                extras["hand_present"][:, drop[:2]] = False

    # Centre the geometry so an undetected stream's zeros are not confused with a
    # real box at the top-left corner.
    geometry = geometry - np.array([0.5, 0.5, 0.0], dtype=np.float32)
    return crops, detected, np.ascontiguousarray(geometry)


def model_inputs(batch: dict, device: str) -> dict:
    """Landmark keyword arguments for the model, or {} for a pixel-only batch."""
    return {key: batch[key].to(device, non_blocking=True)
            for key in ("hands", "hand_present", "pose") if key in batch}


def build_datasets(
    split_file: str | Path, n_frames: int = 8, img_size: int = 64, landmarks: bool = False,
    lm_interp: bool = False, **train_kwargs,
) -> tuple[IncludeCrops, IncludeCrops, IncludeCrops]:
    """Train/val/test over one split file, sharing a single label mapping."""
    with Path(split_file).open(encoding="utf-8") as handle:
        labels = sorted({row["label"] for row in csv.DictReader(handle)})
    label_to_index = {label: index for index, label in enumerate(labels)}

    common = dict(n_frames=n_frames, img_size=img_size, label_to_index=label_to_index,
                  landmarks=landmarks, lm_interp=lm_interp)
    train = IncludeCrops(split_file, "train", train=True, **common, **train_kwargs)
    val = IncludeCrops(split_file, "val", train=False, **common)
    test = IncludeCrops(split_file, "test", train=False, **common)
    return train, val, test


if __name__ == "__main__":
    train, val, test = build_datasets("splits/include50__take-group.csv")
    print(f"classes={train.n_classes} train={len(train)} val={len(val)} test={len(test)}")
    sample = train[0]
    for key, value in sample.items():
        print(f"  {key}: {tuple(value.shape)} {value.dtype}")
    print(f"  crops range [{sample['crops'].min():.2f}, {sample['crops'].max():.2f}]")
