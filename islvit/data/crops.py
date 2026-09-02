"""Extract hand and face crops from INCLUDE videos into a memmapped cache.

The model never sees a full frame. It sees three 80x80 streams per timestep --
left hand, right hand, face -- which strips the background and the signer's body
out of the input entirely. With ~16 clips per class that matters: a full-frame
model has more than enough capacity to memorise the studio and the signer instead
of learning the sign.

Crops come from the raw 1080p ``.MOV`` files rather than the 480p copies. Hands
occupy roughly a tenth of the frame width, so the source has ~190px of hand at
1080p versus ~85px at 480p, and we are resizing into an 80px box.

**Hand boxes come from a two-stage detector.** Full-frame Holistic finds a hand on
only ~79% of frames: at 1080p a hand is ~190px in a 1920px frame, which is small
for the detector, and signing hands blur badly at 25fps. So whenever the direct
pass misses, we take a generous ROI around the pose-predicted hand, upscale it to
256px, and run a dedicated HandLandmarker on that. Measured over 25 clips this
rescues a further 10% of frames, for ~89% genuine coverage.

An earlier version instead *synthesised* a box from pose geometry (wrist, thumb,
index, pinky). It reported 100% coverage and was badly wrong -- pose reports
optimistic visibility for occluded hands, so the boxes landed on shirt fabric and
background. Coverage is not correctness; the crops have to be looked at. The
remaining ~11% is filled by carrying the nearest real detection forward, which at
least stays anchored to a hand that was actually seen.

**Box geometry is saved alongside the pixels.** Cropping deletes *where* the hands
are, and location is one of the defining parameters of a sign -- the same handshape
at the forehead and at the chest are different words. So each crop's normalised
centre and scale go into ``geometry.npy`` for the model to consume as a per-token
feature, which puts back the spatial configuration that cropping removes.

Output (all indexed by the same row order, sorted by ``video_path``):

    cache/crops.npy      uint8   (N, T, 3, H, W, 3)  the crops themselves
    cache/sources.npy    uint8   (N, T, 3)           0 missing 1 direct 2 roi 3 interp
    cache/geometry.npy   float16 (N, T, 3, 3)        box centre x, centre y, size
    cache/index.csv      row -> video_path, label, per-stream coverage
    cache/done.npy       bool    (N,)                resume marker

Run as a module::

    python -m islvit.data.crops [--limit N] [--workers K]
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp_proc
import os
import time
from pathlib import Path

# MediaPipe is extremely chatty on every graph construction; with several workers
# it drowns the progress output. Must be set before the mediapipe import.
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import numpy as np


RAW_DIR = Path("INCLUDE_raw")
PROCESSED_DIR = Path("INCLUDE_480p")
SPLIT_FILE = Path("splits/full263__take-group.csv")
CACHE_DIR = Path("cache")

# Which corpus is being extracted. Only clip discovery and path resolution differ
# between them -- detection, cropping and the cache layout are identical, which is
# the point: a pretraining cache the finetuning loader cannot read is useless.
CORPUS = "include"
CISLR_DIR = Path("CISLR/CISLR_v1.5-a_videos")
CISLR_INDEX = Path("CISLR/dataset.csv")

# Newly recorded clips, laid out as custom/<signer>/<label>/<take>.mp4. The
# signer directory is the point: INCLUDE ships no signer IDs at all, so
# "signer-independent" can only ever be approximated on it (see the report's
# limitations). Recording our own is the only way to measure it directly.
CUSTOM_DIR = Path("custom")
ISIGN_DIR = Path("iSign_videos")
HOLISTIC_MODEL = Path("models/holistic_landmarker.task")
HAND_MODEL = Path("models/hand_landmarker.task")

FRAMES_PER_CLIP = 16
CROP_SIZE = 80
STREAMS = ("left_hand", "right_hand", "face")
N_STREAMS = len(STREAMS)

SOURCE_MISSING, SOURCE_DIRECT, SOURCE_ROI, SOURCE_INTERP = 0, 1, 2, 3

# Enough air around the landmark hull to keep the wrist and finger tips inside a
# moving hand, but not so much that handshape stops filling the crop. Sign
# *location* is carried separately by geometry.npy, so the box does not need to
# include body context to be informative.
HAND_MARGIN = 1.5
FACE_MARGIN = 1.3

# MediaPipe pose landmark indices, anatomical (the subject's own left/right).
POSE_ELBOW = {"left": 13, "right": 14}
POSE_HAND_POINTS = {"left": (15, 17, 19, 21), "right": (16, 18, 20, 22)}
# Pose only has to get us in the neighbourhood -- the hand model does the real
# localisation inside the ROI -- so the visibility bar is low and the ROI is
# deliberately generous. Anchoring the ROI to forearm length keeps it scale-stable
# as the signer moves toward or away from the camera.
POSE_VISIBILITY_THRESHOLD = 0.3
ROI_SCALE = 2.8
ROI_MIN_PIXELS = 60
ROI_RESIZE = 256

VIDEO_EXTENSIONS = (".MOV", ".mov", ".mp4", ".avi", ".mkv")

# Every frame is resized to this height before detection. Two reasons: MediaPipe's
# holistic graph carries state between calls and hard-fails when consecutive
# images differ in size ("current_mat->rows == previous_mat->rows (480 vs 1080)"),
# and a fixed working resolution keeps hand scale consistent across sources.
WORK_HEIGHT = 720
# Wide enough to letterbox rather than crop for every aspect ratio this project
# has seen, including iSign's varied YouTube sources (see pad_to_work_width).
WORK_WIDTH = 1280


def resolve_video(video_path: str) -> Path | None:
    """Find a clip on disk, preferring the raw 1080p source over the 480p copy."""
    if CORPUS == "cislr":
        candidate = CISLR_DIR / video_path
        return candidate if candidate.exists() else None

    if CORPUS == "custom":
        candidate = CUSTOM_DIR / video_path
        return candidate if candidate.exists() else None

    if CORPUS == "isign":
        candidate = ISIGN_DIR / video_path
        return candidate if candidate.exists() else None

    raw = RAW_DIR / video_path
    if raw.exists():
        return raw

    stem = Path(video_path).with_suffix("")
    for extension in VIDEO_EXTENSIONS:
        candidate = RAW_DIR / stem.with_suffix(extension)
        if candidate.exists():
            return candidate
    for extension in (".mp4", ".MOV"):
        candidate = PROCESSED_DIR / stem.with_suffix(extension)
        if candidate.exists():
            return candidate
    return None


def read_sampled_frames(path: Path, count: int) -> list[np.ndarray] | None:
    """Read a clip and return ``count`` frames sampled uniformly across it.

    INCLUDE clips run 50-120 frames, so a sequential read of the whole file is
    cheaper and far more reliable than seeking.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None

    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()

    if not frames:
        return None

    indices = np.linspace(0, len(frames) - 1, count).round().astype(int)
    sampled = [frames[index] for index in indices]

    height, width = sampled[0].shape[:2]
    if height != WORK_HEIGHT:
        scale = WORK_HEIGHT / height
        size = (int(round(width * scale)), WORK_HEIGHT)
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        sampled = [cv2.resize(frame, size, interpolation=interpolation) for frame in sampled]

    return [pad_to_work_width(frame) for frame in sampled]


def pad_to_work_width(frame: np.ndarray) -> np.ndarray:
    """Letterbox to a fixed (WORK_HEIGHT, WORK_WIDTH), never crop.

    MediaPipe's HolisticLandmarker keeps one graph per worker, reused across
    every video that worker handles (``get_holistic``), and its segmentation-
    smoothing calculator retains internal state across calls *regardless* of
    ``output_segmentation_mask=False`` -- confirmed by reproducing it directly:
    two real frames of different widths fed to the same instance crash with
    ``current_mat->cols == previous_mat->cols``, and worse, the graph stays
    broken afterward -- even feeding the *original* width back crashes again.
    Resizing to a fixed height alone (the previous behaviour) still leaves width
    tied to each source video's aspect ratio, so any two videos of different
    aspect ratio processed back-to-back in one worker poisoned it. INCLUDE and
    CISLR share one aspect ratio and never exposed this; iSign's YouTube sources
    do not.

    The fix is a fixed **width** too, so no two calls, from any two videos ever
    disagree. Letterboxing (black-bar padding) rather than cropping, because a
    crop risks cutting off a hand or face that a signer placed near the frame
    edge -- exactly the content this pipeline exists to keep.
    """
    height, width = frame.shape[:2]
    if width == WORK_WIDTH:
        return frame
    if width > WORK_WIDTH:
        # Wider than the padding budget (rare: an ultra-wide source). Center-crop
        # as the last resort rather than silently growing the target size, which
        # would just move the mismatch onto the next unusually wide video.
        start = (width - WORK_WIDTH) // 2
        return frame[:, start : start + WORK_WIDTH]
    pad_total = WORK_WIDTH - width
    left = pad_total // 2
    right = pad_total - left
    return cv2.copyMakeBorder(frame, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def box_from_points(
    xs: list[float], ys: list[float], margin: float, width: int, height: int, min_side: float = 8.0
) -> tuple[int, int, int, int] | None:
    """Square box centred on the points, expanded by ``margin``, clamped to frame."""
    if not xs:
        return None

    centre_x, centre_y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    side = max(max(xs) - min(xs), max(ys) - min(ys)) * margin
    if side < min_side:
        return None
    return finalise_box(centre_x, centre_y, side, width, height)


def finalise_box(
    centre_x: float, centre_y: float, side: float, width: int, height: int
) -> tuple[int, int, int, int] | None:
    half = side / 2
    x0, y0 = max(0, int(round(centre_x - half))), max(0, int(round(centre_y - half)))
    x1, y1 = min(width, int(round(centre_x + half))), min(height, int(round(centre_y + half)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def hand_box_from_landmarks(landmarks, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not landmarks:
        return None
    xs = [point.x * width for point in landmarks]
    ys = [point.y * height for point in landmarks]
    return box_from_points(xs, ys, HAND_MARGIN, width, height)


def pose_roi(pose, side: str, width: int, height: int) -> tuple[int, int, int, int] | None:
    """A generous search region around where pose thinks the hand is.

    This is only a search hint -- the hand model localises inside it -- so being
    loose here is safe and being tight is not.
    """
    indices = POSE_HAND_POINTS[side]
    elbow_index = POSE_ELBOW[side]
    if not pose or max(indices) >= len(pose) or elbow_index >= len(pose):
        return None

    points = [pose[index] for index in indices]
    if min(getattr(point, "visibility", 1.0) for point in points) < POSE_VISIBILITY_THRESHOLD:
        return None

    xs = [point.x * width for point in points]
    ys = [point.y * height for point in points]
    wrist, elbow = pose[indices[0]], pose[elbow_index]
    forearm = np.hypot((wrist.x - elbow.x) * width, (wrist.y - elbow.y) * height)
    return finalise_box(
        sum(xs) / len(xs), sum(ys) / len(ys), max(forearm * ROI_SCALE, ROI_MIN_PIXELS), width, height
    )


def hand_box_in_roi(frame: np.ndarray, roi: tuple[int, int, int, int], width: int, height: int):
    """Run the dedicated hand model on an upscaled ROI; map its box back to frame."""
    import mediapipe as mp

    x0, y0, x1, y1 = roi
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return None

    scale = ROI_RESIZE / max(patch.shape[:2])
    enlarged = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(enlarged, cv2.COLOR_BGR2RGB))
    result = get_hand_landmarker().detect(image)
    if not result.hand_landmarks:
        return None

    landmarks = result.hand_landmarks[0]
    xs = [point.x * enlarged.shape[1] / scale + x0 for point in landmarks]
    ys = [point.y * enlarged.shape[0] / scale + y0 for point in landmarks]
    side_length = max(max(xs) - min(xs), max(ys) - min(ys)) * HAND_MARGIN
    if side_length < 8:
        return None
    return finalise_box((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, side_length, width, height)


def face_box_from_landmarks(landmarks, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not landmarks:
        return None
    xs = [point.x * width for point in landmarks]
    ys = [point.y * height for point in landmarks]
    return box_from_points(xs, ys, FACE_MARGIN, width, height)


def fill_gaps(boxes: list, sources: list[int]) -> tuple[list, list[int]]:
    """Carry the nearest detected box into frames that have none."""
    known = [index for index, box in enumerate(boxes) if box is not None]
    if not known:
        return boxes, sources

    filled, filled_sources = list(boxes), list(sources)
    for index, box in enumerate(boxes):
        if box is None:
            nearest = min(known, key=lambda candidate: abs(candidate - index))
            filled[index] = boxes[nearest]
            filled_sources[index] = SOURCE_INTERP
    return filled, filled_sources


def crop_to_square(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    resized = cv2.resize(patch, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def init_worker(crop_size: int, corpus: str = "include", frames: int | None = None) -> None:
    """Push config into a spawned worker.

    Windows spawns rather than forks, so each worker re-imports this module and
    would otherwise see the module-level defaults instead of any CLI override.

    Every CLI override has to be listed here. ``--frames`` was added without it and
    the workers kept returning 16-frame clips into a cache allocated for 32, which
    at least failed loudly on the shape mismatch -- a knob that merely changed
    behaviour rather than shape would have passed silently.
    """
    global CROP_SIZE, CORPUS, FRAMES_PER_CLIP
    CROP_SIZE = crop_size
    CORPUS = corpus
    if frames is not None:
        FRAMES_PER_CLIP = frames


_holistic = None
_hand = None


def get_holistic():
    """One MediaPipe graph per worker process, built lazily and reused."""
    global _holistic
    if _holistic is None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions

        vision = mp.tasks.vision
        _holistic = vision.HolisticLandmarker.create_from_options(
            vision.HolisticLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(HOLISTIC_MODEL)),
                # IMAGE mode, not VIDEO: our 16 frames are sampled across the whole
                # clip, so consecutive inputs are ~0.3s apart and the tracker's
                # temporal assumption does not hold. Costs nothing -- measured
                # 34ms vs 32ms per frame.
                running_mode=vision.RunningMode.IMAGE,
                min_hand_landmarks_confidence=0.3,
                # We never use the mask, and its smoothing calculator is the part
                # that retains cross-call state and crashes on size changes.
                output_segmentation_mask=False,
            )
        )
    return _holistic


def get_hand_landmarker():
    """Second-stage hand model, run only on upscaled ROIs."""
    global _hand
    if _hand is None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions

        vision = mp.tasks.vision
        _hand = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=1,
                # The ROI already says a hand is here; the thresholds only need to
                # decide where, so they are deliberately permissive.
                min_hand_detection_confidence=0.2,
                min_hand_presence_confidence=0.2,
            )
        )
    return _hand


def process_video(job: tuple[int, str]):
    """Resolve a corpus-relative path and extract it. Pool worker entry point."""
    row, video_path = job

    path = resolve_video(video_path)
    if path is None:
        return row, None, None, None, "file not found"

    crops, source_array, geometry, error = extract_clip(path)
    return row, crops, source_array, geometry, error


def extract_clip(path: Path):
    """Detect and crop one video file. Returns (crops, sources, geometry, error).

    Split out of ``process_video`` so inference (``islvit.predict``) runs the
    *identical* detection and cropping path as cache building. A separate
    inference preprocessor is the classic way a model that scores well offline
    fails on a real video, and the only reliable defence is to not have one.
    """
    import mediapipe as mp

    frames = read_sampled_frames(path, FRAMES_PER_CLIP)
    if frames is None:
        return None, None, None, "cannot decode"

    holistic = get_holistic()
    height, width = frames[0].shape[:2]

    boxes: list[list] = [[] for _ in STREAMS]
    sources: list[list[int]] = [[] for _ in STREAMS]

    # MediaPipe's CalculatorGraph can raise RuntimeError on individual frames --
    # unusual codecs, corrupt frames, resolutions it dislikes. INCLUDE and CISLR
    # are clean studio/near-studio footage and never triggered this; iSign is
    # unconstrained YouTube video, where it happened on the very first pooled
    # batch and killed all 18,000 pending clips before the first checkpoint,
    # because the exception was never caught here -- it propagated out of the
    # worker and through imap_unordered, aborting the whole run. One bad video
    # must not cost the other 17,999.
    try:
        for frame in frames:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = holistic.detect(image)

            for stream_index, side in enumerate(("left", "right")):
                landmarks = result.left_hand_landmarks if side == "left" else result.right_hand_landmarks
                box = hand_box_from_landmarks(landmarks, width, height)
                source = SOURCE_DIRECT

                if box is None:
                    # Second stage: the hand is small or blurred at full resolution,
                    # so look again inside an upscaled ROI around the pose estimate.
                    roi = pose_roi(result.pose_landmarks, side, width, height)
                    box = hand_box_in_roi(frame, roi, width, height) if roi else None
                    source = SOURCE_ROI if box is not None else SOURCE_MISSING

                boxes[stream_index].append(box)
                sources[stream_index].append(source)

            face_box = face_box_from_landmarks(result.face_landmarks, width, height)
            boxes[2].append(face_box)
            sources[2].append(SOURCE_DIRECT if face_box is not None else SOURCE_MISSING)
    except RuntimeError as error:
        return None, None, None, f"mediapipe: {error}"

    crops = np.zeros((FRAMES_PER_CLIP, N_STREAMS, CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    source_array = np.zeros((FRAMES_PER_CLIP, N_STREAMS), dtype=np.uint8)
    geometry = np.zeros((FRAMES_PER_CLIP, N_STREAMS, 3), dtype=np.float16)

    for stream_index in range(N_STREAMS):
        stream_boxes, stream_sources = fill_gaps(boxes[stream_index], sources[stream_index])
        for time_index, box in enumerate(stream_boxes):
            if box is None:
                continue
            crops[time_index, stream_index] = crop_to_square(frames[time_index], box)
            source_array[time_index, stream_index] = stream_sources[time_index]

            x0, y0, x1, y1 = box
            geometry[time_index, stream_index] = (
                (x0 + x1) / 2 / width,
                (y0 + y1) / 2 / height,
                (x1 - x0) / width,
            )

    return crops, source_array, geometry, ""


def write_index(videos: list[tuple[str, str]], sources_cache: np.ndarray, done: np.ndarray) -> None:
    """Rewrite index.csv. Called at every checkpoint so a partially built cache is
    already usable for smoke-testing the training path."""
    with (CACHE_DIR / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "video_path", "label", "cached", *(f"{name}_cov" for name in STREAMS)])
        for row, (video_path, label) in enumerate(videos):
            coverage = (sources_cache[row] > 0).mean(axis=0) if done[row] else np.zeros(N_STREAMS)
            writer.writerow([row, video_path, label, int(done[row]), *(f"{value:.3f}" for value in coverage)])


def load_video_list() -> list[tuple[str, str]]:
    """All videos in canonical (sorted) row order, as (video_path, label)."""
    if CORPUS == "cislr":
        # 7,050 clips over 4,765 glosses -- roughly 1.5 clips per word, which is
        # useless for supervised classification and is exactly why this corpus is
        # only ever used for self-supervised pretraining. The gloss is carried
        # anyway so the cache stays inspectable.
        with CISLR_INDEX.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return sorted({(f"{row['uid']}.mp4", row["gloss"]) for row in rows})

    if CORPUS == "isign":
        # iSign is pretraining data only (see islvit.data.isign_extract) -- no
        # gloss to carry, so the "label" is just the filename for inspectability.
        clips = sorted((path.name, path.stem) for path in ISIGN_DIR.glob("*.mp4"))
        if not clips:
            raise SystemExit(f"no videos under {ISIGN_DIR}/ -- run islvit.data.isign_extract first")
        return clips

    if CORPUS == "custom":
        # custom/<signer>/<label>/<take>.mp4 -- the label is the parent
        # directory, so no index file has to be kept in sync by hand.
        clips = [
            (path.relative_to(CUSTOM_DIR).as_posix(), path.parent.name)
            for extension in VIDEO_EXTENSIONS
            for path in CUSTOM_DIR.rglob(f"*{extension}")
        ]
        if not clips:
            raise SystemExit(
                f"no videos under {CUSTOM_DIR}/ -- expected "
                f"{CUSTOM_DIR}/<signer>/<label>/<take>.mp4"
            )
        return sorted(set(clips))

    with SPLIT_FILE.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return sorted({(row["video_path"], row["label"]) for row in rows})


def main() -> None:
    global CACHE_DIR, CROP_SIZE, CORPUS, FRAMES_PER_CLIP

    parser = argparse.ArgumentParser(description="Extract INCLUDE hand/face crops")
    parser.add_argument("--limit", type=int, default=0, help="only process the first N videos")
    parser.add_argument("--workers", type=int, default=0, help="worker processes (default: cpu-2, max 6)")
    parser.add_argument(
        "--require-raw",
        action="store_true",
        help="skip clips with no 1080p source, leaving them for a later pass "
        "(use while refetch_raw is still downloading)",
    )
    parser.add_argument("--cache-dir", type=str, default=None, help="output dir (default: cache/)")
    parser.add_argument("--crop-size", type=int, default=None, help="crop pixels (default: 80)")
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="frames cached per clip (default: 16). Training samples n_frames from these, "
        "so this is the ceiling on the temporal axis -- and §11.10 measured that axis as "
        "the one that pays, worth +6.4 pts going from 8 to 16.",
    )
    parser.add_argument(
        "--corpus",
        choices=("include", "cislr", "custom", "isign"),
        default="include",
        help="which corpus to extract; cislr is pretraining data only, "
        "custom reads newly recorded clips from custom/<signer>/<label>/, "
        "isign reads the extracted subset from iSign_videos/",
    )
    args = parser.parse_args()

    if args.cache_dir:
        CACHE_DIR = Path(args.cache_dir)
    if args.crop_size:
        CROP_SIZE = args.crop_size
    if args.frames:
        FRAMES_PER_CLIP = args.frames
    CORPUS = args.corpus

    missing_models = [path for path in (HOLISTIC_MODEL, HAND_MODEL) if not path.exists()]
    if missing_models:
        base = "https://storage.googleapis.com/mediapipe-models"
        urls = {
            HOLISTIC_MODEL: f"{base}/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task",
            HAND_MODEL: f"{base}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        }
        lines = "\n".join(f"  curl -sL -o {path} {urls[path]}" for path in missing_models)
        raise SystemExit(f"Missing model assets. Download with:\n{lines}")

    videos = load_video_list()
    if args.limit:
        videos = videos[: args.limit]
    total = len(videos)

    workers = args.workers or min(6, max(1, (os.cpu_count() or 4) - 2))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    crops_path = CACHE_DIR / "crops.npy"
    sources_path = CACHE_DIR / "sources.npy"
    geometry_path = CACHE_DIR / "geometry.npy"
    done_path = CACHE_DIR / "done.npy"

    shape = (total, FRAMES_PER_CLIP, N_STREAMS, CROP_SIZE, CROP_SIZE, 3)
    mode = "r+" if crops_path.exists() else "w+"
    print(f"{total} videos -> {crops_path} {shape} ({np.prod(shape) / 1e9:.2f} GB), {workers} workers, mode={mode}")

    # open_memmap ignores `shape` when reopening, so a cache built under different
    # settings -- a --limit smoke test, another --crop-size, a changed split --
    # silently returns the OLD array and the write loop then runs off the end.
    # Checking here turns that into one clear message instead of an IndexError
    # thousands of clips into a run.
    if mode == "r+":
        existing = np.lib.format.open_memmap(crops_path, mode="r")
        if existing.shape != shape:
            raise SystemExit(
                f"{crops_path} holds {existing.shape}, this run needs {shape}.\n"
                f"The cache was built with different settings (clip count or --crop-size).\n"
                f"Delete {CACHE_DIR} to rebuild, or point --cache-dir somewhere else."
            )
        del existing

    crops_cache = np.lib.format.open_memmap(crops_path, mode=mode, dtype=np.uint8, shape=shape)
    sources_cache = np.lib.format.open_memmap(
        sources_path, mode=mode, dtype=np.uint8, shape=(total, FRAMES_PER_CLIP, N_STREAMS)
    )
    geometry_cache = np.lib.format.open_memmap(
        geometry_path, mode=mode, dtype=np.float16, shape=(total, FRAMES_PER_CLIP, N_STREAMS, 3)
    )
    done = np.zeros(total, dtype=bool)
    if done_path.exists():
        previous = np.load(done_path)
        if previous.shape == (total,):
            done = previous

    pending = [(row, path) for row, (path, _) in enumerate(videos) if not done[row]]
    if args.require_raw:
        deferred = [job for job in pending if not str(resolve_video(job[1]) or "").startswith(str(RAW_DIR))]
        pending = [job for job in pending if job not in deferred]
        print(f"--require-raw: deferring {len(deferred)} clips that have no 1080p source yet")
    if not pending:
        print("Nothing to do; cache is complete.")
        return
    print(f"{len(pending)} videos still to process ({total - int(done.sum())} not yet cached)")

    failures: list[tuple[str, str]] = []
    started = time.time()
    completed = 0

    with mp_proc.Pool(processes=workers, initializer=init_worker, initargs=(CROP_SIZE, CORPUS, FRAMES_PER_CLIP)) as pool:
        for row, crops, source_array, geometry, error in pool.imap_unordered(process_video, pending, chunksize=2):
            completed += 1
            if crops is None:
                failures.append((videos[row][0], error))
            else:
                crops_cache[row] = crops
                sources_cache[row] = source_array
                geometry_cache[row] = geometry
                done[row] = True

            if completed % 100 == 0 or completed == len(pending):
                elapsed = time.time() - started
                rate = completed / elapsed
                remaining = (len(pending) - completed) / rate if rate else 0
                coverage = (sources_cache[done] > 0).mean() if done.any() else 0.0
                print(
                    f"  [{completed}/{len(pending)}] {rate:.1f} clips/s  eta {remaining / 60:.1f} min  "
                    f"coverage {coverage:.1%}  failures {len(failures)}",
                    flush=True,
                )
                np.save(done_path, done)
                write_index(videos, sources_cache, done)

    crops_cache.flush()
    sources_cache.flush()
    geometry_cache.flush()
    np.save(done_path, done)

    write_index(videos, sources_cache, done)

    print(f"\nCached {int(done.sum())}/{total} videos in {(time.time() - started) / 60:.1f} min")
    cached = sources_cache[done]
    for stream_index, name in enumerate(STREAMS):
        column = cached[:, :, stream_index]
        parts = " ".join(
            f"{label}={(column == code).mean():.1%}"
            for code, label in (
                (SOURCE_DIRECT, "direct"),
                (SOURCE_ROI, "roi"),
                (SOURCE_INTERP, "interp"),
                (SOURCE_MISSING, "missing"),
            )
        )
        print(f"  {name:11s} {parts}")
    if failures:
        print(f"\n{len(failures)} failures:")
        for video_path, error in failures[:20]:
            print(f"  {video_path}: {error}")


if __name__ == "__main__":
    main()
