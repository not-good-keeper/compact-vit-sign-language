"""Cut real images from one held-out clip for the plain-language diagram.

Clip: row 0 of the clean test cache, the sign "loud" (MVI_5177). Frames come
from the same sampler that built the caches, so the stored crops and landmarks
line up with them exactly.

    python report/case_study/make_assets.py
"""
from pathlib import Path

import cv2
import numpy as np

from islvit.data import crops

ROW = 0
VIDEO = Path("INCLUDE_raw/Adjectives/1. loud/MVI_5177.MOV")
OUT = Path(__file__).parent / "assets"
SHOW_T = 12  # both hands up near the face

# MediaPipe's 21-point hand topology: wrist, then four joints per finger.
FINGERS = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16),
           (0, 17, 18, 19, 20), (5, 9, 13, 17)]


def save(name, image, width):
    height = round(image.shape[0] * width / image.shape[1])
    cv2.imwrite(str(OUT / f"{name}.jpg"), cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 88])


def trim_letterbox(frame):
    """Drop the black bars the sampler adds, so thumbnails show only video."""
    rows = np.where(frame.max(axis=(1, 2)) > 16)[0]
    cols = np.where(frame.max(axis=(0, 2)) > 16)[0]
    return frame[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1], rows[0], cols[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = crops.read_sampled_frames(VIDEO, 32)
    if frames is None:
        raise SystemExit(f"cannot decode {VIDEO}")

    frame, top, left = trim_letterbox(frames[SHOW_T])
    save("frame", frame, 420)

    # The signer stands centre-frame; a portrait slice keeps the strip readable small.
    strip = []
    for t in (0, 8, 16, 24):
        whole = trim_letterbox(frames[t])[0]
        half = int(whole.shape[1] * 0.2)
        middle = whole.shape[1] // 2
        strip.append(whole[:, middle - half:middle + half])
    height = min(f.shape[0] for f in strip)
    gap = np.full((height, 16, 3), 255, np.uint8)
    film = np.hstack([part for f in strip for part in (f[:height], gap)][:-1])
    save("filmstrip", film, 640)

    cached = np.load("cache128_f32_test/crops.npy", mmap_mode="r")[ROW, SHOW_T]
    for stream, name in enumerate(("crop_left", "crop_right", "crop_face")):
        # The cache stores RGB; OpenCV writes BGR.
        save(name, cv2.cvtColor(np.ascontiguousarray(cached[stream]), cv2.COLOR_RGB2BGR), 160)

    hands = np.load("cache_lm_f32_test/hands.npy", mmap_mode="r")[ROW, SHOW_T].astype(np.float32)
    source = np.load("cache_lm_f32_test/hand_src.npy", mmap_mode="r")[ROW, SHOW_T]
    full_h, full_w = frames[SHOW_T].shape[:2]
    canvas = frame.copy()
    drawn = []
    for side in range(2):
        if not source[side]:
            continue
        points = [(int(x * full_w) - left, int(y * full_h) - top) for x, y, _ in hands[side]]
        for chain in FINGERS:
            for a, b in zip(chain, chain[1:]):
                cv2.line(canvas, points[a], points[b], (255, 255, 255), 3, cv2.LINE_AA)
                cv2.line(canvas, points[a], points[b], (0, 140, 255), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(canvas, point, 3, (180, 60, 20), -1, cv2.LINE_AA)
        drawn.extend(points)
    if not drawn:
        raise SystemExit(f"no hand detected at t={SHOW_T}; pick another timestep")

    xs, ys = zip(*drawn)
    margin = 60
    x0, x1 = max(min(xs) - margin, 0), min(max(xs) + margin, canvas.shape[1])
    y0, y1 = max(min(ys) - margin, 0), min(max(ys) + margin, canvas.shape[0])
    save("joints", canvas[y0:y1, x0:x1], 320)
    print(f"hands detected: {int((source > 0).sum())} of 2; assets in {OUT}")


if __name__ == "__main__":
    main()
