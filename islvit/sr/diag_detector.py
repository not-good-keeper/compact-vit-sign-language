"""Detector-side levers on degraded video: can the landmark extractor be made to find more hands?

The clean-pixels diagnostic (runs/gan_study/diagnostics) showed that at 180p most of
what the SR front end still loses is on the landmark side. This re-runs the landmark
extractor on the *same* degraded + super-resolved frames with different detector
settings, and scores each variant with **clean crops** beside the variant's
landmarks, so the number isolates the landmark side and is comparable with the
diagnostic's 78.2 % (SR-GAN, 180p) and 73.7 % (bicubic).

Nothing here changes the shipped pipeline: the overrides live in this module and are
applied inside the workers only.

    python -m islvit.sr.diag_detector extract  --variant lowconf --out DIR --sr G.pt --degrade 180:40
    python -m islvit.sr.diag_detector interp   --src DIR --out DIR2 [--gap 4]
    python -m islvit.sr.diag_detector score    DIR [DIR ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp_proc
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import numpy as np

from islvit.data import crops, landmarks

CLEAN_CACHE = Path("cache128_f32_test")
CLEAN_LM = Path("cache_lm_f32_test")
# holistic_conf / hand_conf / pose_vis: detection thresholds; roi_resize: size the ROI is
# upscaled to before the hand model; roi_scales: ROI sizes tried in order until one succeeds.
VARIANTS = {
    "base":     dict(holistic_conf=0.3, hand_conf=0.2, pose_vis=0.3, roi_resize=256, roi_scales=(2.8,)),
    "lowconf":  dict(holistic_conf=0.1, hand_conf=0.05, pose_vis=0.1, roi_resize=256, roi_scales=(2.8,)),
    "bigroi":   dict(holistic_conf=0.3, hand_conf=0.2, pose_vis=0.3, roi_resize=512, roi_scales=(2.8,)),
    "multiroi": dict(holistic_conf=0.3, hand_conf=0.2, pose_vis=0.3, roi_resize=256, roi_scales=(2.8, 2.0, 4.5)),
    "all":      dict(holistic_conf=0.1, hand_conf=0.05, pose_vis=0.1, roi_resize=512, roi_scales=(2.8, 2.0, 4.5)),
}
_V: dict = {}


def _holistic():
    if crops._holistic is None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        vision = mp.tasks.vision
        crops._holistic = vision.HolisticLandmarker.create_from_options(vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(crops.HOLISTIC_MODEL)),
            running_mode=vision.RunningMode.IMAGE, min_hand_landmarks_confidence=_V["holistic_conf"],
            output_segmentation_mask=False))
    return crops._holistic


def _hand():
    if crops._hand is None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        vision = mp.tasks.vision
        crops._hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(crops.HAND_MODEL)), running_mode=vision.RunningMode.IMAGE,
            num_hands=1, min_hand_detection_confidence=_V["hand_conf"], min_hand_presence_confidence=_V["hand_conf"]))
    return crops._hand


def init(crop_size, corpus, frames, upscaler, degrade, variant):
    global _V
    crops.init_worker(crop_size, corpus, frames, upscaler, degrade)
    _V = VARIANTS[variant]
    crops.get_holistic, crops.get_hand_landmarker = _holistic, _hand
    crops.ROI_RESIZE, crops.POSE_VISIBILITY_THRESHOLD = _V["roi_resize"], _V["pose_vis"]


def extract_clip(path: Path):
    """landmarks.extract_landmarks with a multi-scale ROI rescue."""
    import mediapipe as mp
    frames = crops.read_sampled_frames(path, crops.FRAMES_PER_CLIP)
    if frames is None:
        return None, None, None, "cannot decode"
    crops.reset_detectors()
    count, (height, width) = len(frames), frames[0].shape[:2]
    hands = np.zeros((count, 2, landmarks.N_HAND, 3), np.float32)
    source = np.zeros((count, 2), np.uint8)
    pose = np.zeros((count, len(landmarks.POSE_POINTS), 3), np.float32)
    holistic = crops.get_holistic()
    try:
        for t, frame in enumerate(frames):
            result = holistic.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if result.pose_landmarks:
                for k, index in enumerate(landmarks.POSE_POINTS):
                    p = result.pose_landmarks[index]
                    pose[t, k] = (p.x, p.y, getattr(p, "visibility", 1.0))
            for s, side in enumerate(("left", "right")):
                found = result.left_hand_landmarks if side == "left" else result.right_hand_landmarks
                if found:
                    hands[t, s] = [(p.x, p.y, p.z) for p in found]
                    source[t, s] = landmarks.SRC_DIRECT
                    continue
                for scale in _V["roi_scales"]:
                    crops.ROI_SCALE = scale
                    roi = crops.pose_roi(result.pose_landmarks, side, width, height)
                    points = landmarks.hand_points_in_roi(frame, roi, width, height) if roi else None
                    if points is not None:
                        hands[t, s], source[t, s] = points, landmarks.SRC_ROI
                        break
    except RuntimeError as error:
        return None, None, None, f"mediapipe: {error}"
    return hands, source, pose, ""


def job(item):
    row, video_path = item
    path = crops.resolve_video(video_path)
    return (row, *extract_clip(path)) if path else (row, None, None, None, "file not found")


def read_rows(split: str | None = None, n: int | None = None):
    """Rows of the reference cache; ``split`` restricts to one split of the original
    session-disjoint file (the dev set is its 'val' sessions), ``n`` to an even subsample."""
    with (CLEAN_CACHE / "index.csv").open(encoding="utf-8") as h:
        rows = [(int(r["row"]), r["video_path"]) for r in csv.DictReader(h) if r["cached"] == "1"]
    if split:
        with Path("splits/full263__session-disjoint.csv").open(encoding="utf-8") as h:
            wanted = {r["video_path"] for r in csv.DictReader(h) if r["split"] == split}
        with Path("splits/_test472_paths.csv").open(encoding="utf-8") as h:
            assert not wanted & {r["video_path"] for r in csv.DictReader(h)}, "test clip in the dev split"
        rows = [r for r in rows if r[1] in wanted]
    if n and n < len(rows):
        rows = [rows[i] for i in np.linspace(0, len(rows) - 1, n).astype(int)]
    return rows


def use_reference(cache: str, lm: str) -> None:
    global CLEAN_CACHE, CLEAN_LM
    CLEAN_CACHE, CLEAN_LM = Path(cache), Path(lm)


def write_meta(out: Path):
    (out / "SOURCE_CACHE").write_text(f"{CLEAN_CACHE.resolve().as_posix()}\n32\n", encoding="utf-8")
    # Scored beside the clean crops, so the pair must present as one clean front end.
    (out / "FRONTEND.json").write_text(json.dumps(crops.read_frontend(CLEAN_CACHE), indent=2), encoding="utf-8")


def cmd_extract(a):
    from islvit.sr.degrade import parse_spec
    degrade = parse_spec(a.degrade)
    use_reference(a.ref_cache, a.ref_lm)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(a.split, a.n)[: a.limit]
    n = max(r for r, _ in read_rows()) + 1
    fmt = np.lib.format
    hands = fmt.open_memmap(out / "hands.npy", "w+", np.float16, (n, 32, 2, 21, 3))
    source = fmt.open_memmap(out / "hand_src.npy", "w+", np.uint8, (n, 32, 2))
    pose = fmt.open_memmap(out / "pose.npy", "w+", np.float16, (n, 32, len(landmarks.POSE_POINTS), 3))
    done = np.zeros(n, bool)
    start = time.time()
    with mp_proc.Pool(a.workers, initializer=init, initargs=(128, "include", 32, a.sr, degrade, a.variant)) as pool:
        for i, (row, h, s, p, err) in enumerate(pool.imap_unordered(job, rows, chunksize=2), 1):
            if err:
                print("failed", row, err)
                continue
            hands[row], source[row], pose[row], done[row] = h, s, p, True
            if i % 100 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)} {i / (time.time() - start):.2f} clips/s", flush=True)
    for x in (hands, source, pose):
        x.flush()
    np.save(out / "done.npy", done)
    write_meta(out)
    print(f"wrote {out}  coverage {(source[done] > 0).mean(axis=(0, 1))}")


def cmd_interp(a):
    from islvit.data.dataset import interpolate_hands
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hands, source, pose = (np.load(src / f"{k}.npy") for k in ("hands", "hand_src", "pose"))
    done = np.load(src / "done.npy")
    new_h, new_s = hands.copy(), source.copy()
    for r in np.flatnonzero(done):
        h, present = interpolate_hands(hands[r].astype(np.float32), source[r] > 0, a.gap)
        filled = present & ~(source[r] > 0)
        new_h[r], new_s[r][filled] = h, 2   # counted as present (ROI-like), as the model sees interpolated hands
    np.save(out / "hands.npy", new_h); np.save(out / "hand_src.npy", new_s)
    np.save(out / "pose.npy", pose); np.save(out / "done.npy", done)
    write_meta(out)
    print(f"interpolated {int(((new_s > 0) & ~(source > 0)).sum())} hand-frames (gap <= {a.gap})")


def stats(lm: Path):
    present, ref = np.load(lm / "hand_src.npy") > 0, np.load(CLEAN_LM / "hand_src.npy") > 0
    done = np.load(lm / "done.npy")
    present, ref = present[done], ref[done]
    hands, rhands = np.load(lm / "hands.npy")[done].astype(np.float32), np.load(CLEAN_LM / "hands.npy")[done].astype(np.float32)
    both = present & ref
    d = hands[..., 0, :2] - rhands[..., 0, :2]
    err = np.hypot(d[..., 0] * 16 / 9, d[..., 1])
    # All 21 points, relative to the wrist so a whole-hand shift and a wrong handshape are told apart.
    rel, rrel = hands[..., :2] - hands[..., :1, :2], rhands[..., :2] - rhands[..., :1, :2]
    shape_err = np.hypot((rel - rrel)[..., 0] * 16 / 9, (rel - rrel)[..., 1]).mean(-1)
    return {"present": float(present.mean()), "clean_present": float(ref.mean()),
            "missed": float((~present & ref).mean()), "extra": float((present & ~ref).mean()),
            "wrist_err_med": float(np.median(err[both])), "misplaced": float(((err > 0.05) & both).mean()),
            "points_err_med": float(np.median(shape_err[both])), "points_err_mean": float(np.mean(shape_err[both])),
            "jitter_corr": jitter(err, both)}


def jitter(err, both):
    """Lag-1 autocorrelation of the wrist error over time: ~1 = a steady bias, ~0 = flicker."""
    a, b = [], []
    for c in range(err.shape[0]):
        for s in range(2):
            ok = both[c, :, s]
            idx = np.flatnonzero(ok[1:] & ok[:-1])
            a += list(err[c, idx, s]); b += list(err[c, idx + 1, s])
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 10 else None


def cmd_stats(a):
    use_reference(a.ref_cache, a.ref_lm)
    results = {}
    for d in a.dirs:
        r = results[Path(d).name] = stats(Path(d))
        print(f"{Path(d).name:28s} present {100 * r['present']:.1f} (clean {100 * r['clean_present']:.1f})  "
              f"missed {100 * r['missed']:.1f}  extra {100 * r['extra']:.1f}  misplaced {100 * r['misplaced']:.2f}  "
              f"wrist {r['wrist_err_med']:.4f}  points {r['points_err_med']:.4f}", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2))


def cmd_score(a):
    results = {}
    for d in a.dirs:
        lm = Path(d)
        accs = []
        for seed in range(3):
            out = lm / f"score_s{seed}.json"
            if not out.exists():
                env = dict(os.environ, ISLVIT_CACHE=str(CLEAN_CACHE), ISLVIT_CACHE_LM=str(lm))
                subprocess.run([sys.executable, "-m", "islvit.mask50", "--run", f"release/isl_vit_tiny_lm_int4_s{seed}.pt",
                                "--out", str(out)], env=env, check=True, capture_output=True)
            accs.append(json.loads(out.read_text())["masked_tta"]["top1"])
        results[lm.name] = {**stats(lm), "top1_mean": float(np.mean(accs)), "top1_seeds": accs}
        r = results[lm.name]
        print(f"{lm.name:28s} top1 {100 * r['top1_mean']:.1f}  present {100 * r['present']:.1f} (clean {100 * r['clean_present']:.1f})  "
              f"missed {100 * r['missed']:.1f}  extra {100 * r['extra']:.1f}  misplaced {100 * r['misplaced']:.1f}  "
              f"wrist {r['wrist_err_med']:.2f}  lag1 {r['jitter_corr']}", flush=True)
    Path(a.json).write_text(json.dumps(results, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--variant", required=True, choices=VARIANTS)
    e.add_argument("--out", required=True); e.add_argument("--sr", default=None); e.add_argument("--degrade", default=None)
    e.add_argument("--workers", type=int, default=6); e.add_argument("--limit", type=int, default=None)
    e.add_argument("--split", default=None, help="'val' = the development sessions")
    e.add_argument("--n", type=int, default=None, help="even subsample of the rows")
    e.add_argument("--ref-cache", default=str(CLEAN_CACHE)); e.add_argument("--ref-lm", default=str(CLEAN_LM))
    st = sub.add_parser("stats")
    st.add_argument("dirs", nargs="+"); st.add_argument("--json", default=None)
    st.add_argument("--ref-cache", default=str(CLEAN_CACHE)); st.add_argument("--ref-lm", default=str(CLEAN_LM))
    i = sub.add_parser("interp")
    i.add_argument("--src", required=True); i.add_argument("--out", required=True); i.add_argument("--gap", type=int, default=4)
    s = sub.add_parser("score")
    s.add_argument("dirs", nargs="+"); s.add_argument("--json", default="runs/gan_study/diagnostics/detector_levers.json")
    a = p.parse_args()
    {"extract": cmd_extract, "interp": cmd_interp, "score": cmd_score, "stats": cmd_stats}[a.cmd](a)


if __name__ == "__main__":
    main()
