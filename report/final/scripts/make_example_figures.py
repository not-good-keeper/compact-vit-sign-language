"""Example-image figures for the final report: dataset frames, crops, and the super-resolution front ends.

Every image is produced by the project's own preprocessing code (islvit.data.crops, islvit.sr), run on
real clips, so the figures show what the model actually receives. Nothing here changes the pipeline.

    python report/final/scripts/make_example_figures.py [--out DIR] [--only dataset|crops|gan|detect]

Outputs fig_ex_dataset.png, fig_ex_crops.png, fig_ex_gan.png and fig_ex_detect.png in DIR.
"""
import argparse
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from islvit.data import crops  # noqa: E402

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

INK, MUTE = "#1F2933", "#5B6670"
BLUE, ORNG, GRN, RED = "#34659A", "#B8742B", "#3E7D55", "#A94442"
BGR = {"left": (154, 101, 52), "right": (43, 116, 184), "face": (85, 125, 62)}   # BGR of BLUE, ORNG, GRN
GENERATORS = {"SR-L1": "runs/sr_l1/generator.pt", "SR-GAN v2": "runs/sr_gan/generator.pt",
              "SR-GAN v3": "runs/sr3_gan/generator.pt"}
DEGRADE_180 = (180, 40)
SPLIT = "splits/full263__session-disjoint-foldval.csv"
TEST = "splits/_test472_paths.csv"
CACHE = Path("cache128_f32")
W_IN = 7.3


# ----------------------------------------------------------------------------- layout helpers
class Sheet:
    """A figure laid out in inches from the top-left corner."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w, h), dpi=220)
        self.fig.patch.set_facecolor("white")

    def image(self, x, y, w, h, img, edge=None):
        ax = self.fig.add_axes([x / self.w, 1 - (y + h) / self.h, w / self.w, h / self.h])
        ax.imshow(img[..., ::-1] if img.ndim == 3 else img, aspect="auto", interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.0 if edge else 0.4)
            spine.set_edgecolor(edge or "#9AA3AB")
        return ax

    def text(self, x, y, s, size=6.5, ha="left", va="top", color=INK, weight="normal", rot=0, style="normal"):
        self.fig.text(x / self.w, 1 - y / self.h, s, fontsize=size, ha=ha, va=va, color=color, fontweight=weight,
                      rotation=rot, fontstyle=style, linespacing=1.2)

    def save(self, path):
        self.fig.savefig(path, dpi=220)
        plt.close(self.fig)
        print("wrote", path)


def plain(label):
    return re.sub(r"^\d+\.\s*", "", label)


def read_rows(path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def frames_at(path, fractions):
    """Original-resolution frames at the given fractions of the clip (sequential read)."""
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = {int(round(f * (count - 1))): i for i, f in enumerate(fractions)}
    found = {}
    for index in range(count):
        ok, frame = cap.read()
        if not ok:
            break
        if index in wanted:
            found[wanted[index]] = frame
    cap.release()
    return [found[i] for i in range(len(fractions)) if i in found]


def fit(img, width):
    height = int(round(img.shape[0] * width / img.shape[1]))
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------- detection on a clip
def detect_clip(frames):
    """The per-frame logic of crops.extract_clip, but returning boxes, sources and joints."""
    import mediapipe as mp

    crops.reset_detectors()
    holistic = crops.get_holistic()
    height, width = frames[0].shape[:2]
    records = []
    for frame in frames:
        result = holistic.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        record = {}
        for side in ("left", "right"):
            landmarks = result.left_hand_landmarks if side == "left" else result.right_hand_landmarks
            box = crops.hand_box_from_landmarks(landmarks, width, height)
            source = crops.SOURCE_DIRECT
            points = [(p.x * width, p.y * height) for p in landmarks] if landmarks else None
            if box is None:
                roi = crops.pose_roi(result.pose_landmarks, side, width, height)
                box = crops.hand_box_in_roi(frame, roi, width, height) if roi else None
                source = crops.SOURCE_ROI if box is not None else crops.SOURCE_MISSING
                points = None
            record[side] = (box, source, points)
        face = crops.face_box_from_landmarks(result.face_landmarks, width, height)
        record["face"] = (face, crops.SOURCE_DIRECT if face is not None else crops.SOURCE_MISSING, None)
        records.append(record)
    return records


def frontend_frames(path, kind, count=32):
    """720p letterboxed frames after a front end: clean, bicubic (180p input) or a generator (180p input)."""
    crops.UPSCALER, crops.DEGRADE = None, None
    if kind != "clean":
        crops.DEGRADE = DEGRADE_180
        if kind != "Bicubic":
            crops.UPSCALER = GENERATORS[kind]
    try:
        return crops.read_sampled_frames(path, count)
    finally:
        crops.UPSCALER, crops.DEGRADE = None, None


def hands_found(record):
    return sum(record[s][1] in (crops.SOURCE_DIRECT, crops.SOURCE_ROI) for s in ("left", "right"))


# ----------------------------------------------------------------------------- Figure 1: dataset
def figure_dataset(out):
    rng = random.Random(11)
    train = [r for r in read_rows(SPLIT) if r["split"] == "train"]
    by_cat = {}
    for row in train:
        by_cat.setdefault(row["parent_label"], []).append(row)
    cats = ["Adjectives", "Animals", "Clothes", "Colours", "Days_and_Time", "Greetings", "Home", "Jobs", "People", "Places"]
    picks = [rng.choice(by_cat[c]) for c in cats]

    sheet = Sheet(W_IN, 5.0)
    sheet.text(0.05, 0.03, "(a) INCLUDE", 7.5, weight="bold")
    tw, th, gap = 1.38, 0.78, 0.075
    for i, row in enumerate(picks):
        frame = frames_at(crops.resolve_video(row["video_path"]), [0.5])[0]
        x, y = 0.05 + (i % 5) * (tw + gap), 0.28 + (i // 5) * (th + 0.3)
        sheet.image(x, y, tw, th, fit(frame, 480))
        sheet.text(x + tw / 2, y + th + 0.03, f"{plain(row['label'])}  ·  {row['parent_label'].replace('_', ' ')}", 6, ha="center", color=MUTE)

    y0 = 0.28 + 2 * (th + 0.3) + 0.08
    word = rng.choice(by_cat["Greetings"])
    sheet.text(0.05, y0, f"(b) INCLUDE, one clip over time: “{plain(word['label'])}”", 7.5, weight="bold")
    strip = frames_at(crops.resolve_video(word["video_path"]), np.linspace(0.05, 0.95, 8))
    sw, sh, sgap = 0.8, 0.45, 0.07
    for i, frame in enumerate(strip):
        sheet.image(0.05 + i * (sw + sgap), y0 + 0.24, sw, sh, fit(frame, 360))

    y1 = y0 + 0.24 + sh + 0.22
    sheet.text(0.05, y1, "(c) CISLR", 7.5, weight="bold")
    names = sorted(p.name for p in crops.CISLR_DIR.glob("*.mp4"))
    cis = [crops.CISLR_DIR / n for n in random.Random(5).sample(names, 6)]
    cw = 1.0
    for i, path in enumerate(cis):
        frame = frames_at(path, [0.5])[0]
        sheet.image(0.05 + i * (cw + 0.2), y1 + 0.24, cw, cw, fit(frame, 300))
    sheet.save(out / "fig_ex_dataset.png")


# ----------------------------------------------------------------------------- Figure 2: crops
def figure_crops(out):
    rng = random.Random(3)
    index = {int(r["row"]): r for r in read_rows(CACHE / "index.csv")}
    sources = np.load(CACHE / "sources.npy", mmap_mode="r")
    geometry = np.load(CACHE / "geometry.npy", mmap_mode="r").astype(np.float32)
    cache = np.load(CACHE / "crops.npy", mmap_mode="r")
    train_rows = [r for r in read_rows(SPLIT) if r["split"] == "train"]
    paths = {r["video_path"] for r in train_rows}
    row_of = {r["video_path"]: int(k) for k, r in index.items() for _ in [0]}
    train_idx = sorted(row_of[p] for p in paths if p in row_of)
    direct = [i for i in train_idx if (np.asarray(sources[i]) == 1).mean() >= 0.97]
    move = {i: float(geometry[i, :, :2, :2].std(axis=0).sum()) for i in direct}
    main = max(direct, key=lambda i: move[i])
    categories = {}
    for i in direct:
        categories.setdefault(index[i]["video_path"].split("/")[0], []).append(i)
    others = [rng.choice(categories[c]) for c in rng.sample(sorted(categories), 4)]

    sheet = Sheet(W_IN, 6.1)
    info = index[main]
    sheet.text(0.05, 0.03, f"(a) One training clip (“{plain(info['label'])}”)",
               7.5, weight="bold")
    frames = crops.read_sampled_frames(crops.resolve_video(info["video_path"]), 32)
    steps = list(range(0, 32, 4))
    tw, th, gap, left = 0.78, 0.44, 0.075, 0.52
    for c, t in enumerate(steps):
        canvas = frames[t].copy()
        for s, key in enumerate(("left", "right", "face")):
            cx, cy, size = geometry[main, t, s]
            side = size * 1280
            x0, y0 = int(cx * 1280 - side / 2), int(cy * 720 - side / 2)
            cv2.rectangle(canvas, (x0, y0), (int(x0 + side), int(y0 + side)), BGR[key], 9)
        sheet.image(left + c * (tw + gap), 0.27, tw, th, fit(canvas, 400))
        for s in range(3):
            sheet.image(left + c * (tw + gap), 0.27 + th + 0.07 + s * (tw + 0.05), tw, tw, cv2.cvtColor(np.asarray(cache[main, t, s]), cv2.COLOR_RGB2BGR))
    for s, name in enumerate(("left hand", "right hand", "face")):
        sheet.text(left - 0.05, 0.27 + th + 0.07 + s * (tw + 0.05) + tw / 2, name, 6.2, ha="right", va="center", color=(BLUE, ORNG, GRN)[s])
    sheet.text(left - 0.05, 0.27 + th / 2, "frame", 6.2, ha="right", va="center", color=MUTE)
    sheet.text(left, 0.27 + th + 0.07 + 3 * (tw + 0.05) + 0.0, "frames 1, 5, 9, …, 29 of the 32 stored; the model reads 16 of them", 6, color=MUTE)

    y1 = 0.27 + th + 0.07 + 3 * (tw + 0.05) + 0.26
    sheet.text(0.05, y1, "(b) Four other words (left hand, right hand, face)", 7.5, weight="bold")
    cw = 0.5
    for g, row in enumerate(others):
        gx = 0.05 + g * (3 * cw + 0.12 + 0.2)
        for s in range(3):
            sheet.image(gx + s * (cw + 0.02), y1 + 0.24, cw, cw, cv2.cvtColor(np.asarray(cache[row, 16, s]), cv2.COLOR_RGB2BGR))
        sheet.text(gx + 1.5 * cw, y1 + 0.24 + cw + 0.03, plain(index[row]["label"]), 6.2, ha="center", color=MUTE)

    y2 = y1 + 0.24 + cw + 0.3
    sheet.text(0.05, y2, "(c) Crops by how the box was obtained", 7.5, weight="bold")
    titles = {1: "direct landmark box", 2: "forearm region rescue", 3: "interpolated from neighbour", 0: "no box in the clip"}
    pw = 0.75
    for col, code in enumerate((1, 2, 3, 0)):
        seen, found = set(), []
        streams = (0, 1) if code == 2 else (0, 1, 2)
        order = list(range(len(sources)))
        random.Random(9 + code).shuffle(order)
        for i in order:
            if code != 0 and i not in train_idx:
                continue
            arr = np.asarray(sources[i])
            hits = np.argwhere(arr == code)
            hits = [(t, s) for t, s in hits if s in streams]
            if hits and i not in seen:
                found.append((i, *hits[len(hits) // 2]))
                seen.add(i)
            if len(found) == 2:
                break
        gx = 0.05 + col * (2 * pw + 0.1 + 0.18)
        for k, (i, t, s) in enumerate(found):
            sheet.image(gx + k * (pw + 0.04), y2 + 0.24, pw, pw, cv2.cvtColor(np.asarray(cache[i, t, s]), cv2.COLOR_RGB2BGR))
            sheet.text(gx + k * (pw + 0.04) + pw / 2, y2 + 0.24 + pw + 0.02, ("left hand", "right hand", "face")[s], 5.8, ha="center", color=MUTE)
        sheet.text(gx + pw + 0.02, y2 + 0.24 + pw + 0.2, titles[code], 6.2, ha="center", color=INK)
    sheet.save(out / "fig_ex_crops.png")


# ----------------------------------------------------------------------------- Figures 3 and 4: GAN front ends
def test_clips():
    return [r["video_path"] for r in read_rows(TEST)]


def box_side(box):
    return max(box[2] - box[0], box[3] - box[1])


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def figure_gan(out, picks):
    kinds = ["clean", "degraded", "Bicubic", "SR-L1", "SR-GAN v2", "SR-GAN v3"]
    heads = ["1080p source", "180p input, JPEG 40", "Bicubic", "SR-L1", "SR-GAN v2", "SR-GAN v3"]
    sheet = Sheet(W_IN, 0.42 + 1.28 * len(picks))
    tile, gap, left = 1.06, 0.07, 0.24
    for c, head in enumerate(heads):
        sheet.text(left + c * (tile + gap) + tile / 2, 0.06, head, 6.3, ha="center", color=INK, weight="bold")
    for r, (path, t, side) in enumerate(picks):
        clip = crops.resolve_video(path)
        clean = crops.read_sampled_frames(clip, 32)
        box = detect_clip(clean)[t][side][0]
        x0, y0, x1, y1 = box
        ref = clean[t][y0:y1, x0:x1]
        deg = cv2.resize(clean[t], (320, 180), interpolation=cv2.INTER_AREA)
        deg = cv2.imdecode(cv2.imencode(".jpg", deg, [cv2.IMWRITE_JPEG_QUALITY, 40])[1], cv2.IMREAD_COLOR)
        lowres = cv2.resize(deg[y0 // 4:max(y0 // 4 + 2, y1 // 4), x0 // 4:max(x0 // 4 + 2, x1 // 4)], (tile_px := 220, 220), interpolation=cv2.INTER_NEAREST)
        tiles = [(cv2.resize(ref, (220, 220), interpolation=cv2.INTER_AREA), None), (lowres, None)]
        for kind in kinds[2:]:
            frame = frontend_frames(clip, kind)[t]
            region = frame[y0:y1, x0:x1]
            tiles.append((cv2.resize(region, (220, 220), interpolation=cv2.INTER_AREA), psnr(region, ref)))
        y = 0.3 + r * (tile + 0.22)
        for c, (img, score) in enumerate(tiles):
            sheet.image(left + c * (tile + gap), y, tile, tile, img)
            if score is not None:
                sheet.text(left + c * (tile + gap) + tile / 2, y + tile + 0.02, f"{score:.1f} dB", 5.8, ha="center", color=MUTE)
        word = plain(path.split("/")[1])
        sheet.text(0.1, y + tile / 2, f"{word}, {side} hand", 5.8, ha="center", va="center", rot=90, color=MUTE)
    sheet.save(out / "fig_ex_gan.png")


def scan_for_detection(limit=36):
    cache = Path("scratch") / "example_scan.json"
    if cache.exists():
        return json.loads(cache.read_text())
    clips = test_clips()
    random.Random(21).shuffle(clips)
    rows = []
    for path in clips[:limit]:
        clip = crops.resolve_video(path)
        clean = detect_clip(frontend_frames(clip, "clean"))
        counts = {"clean": [hands_found(r) for r in clean]}
        for kind in ("Bicubic", "SR-L1", "SR-GAN v2", "SR-GAN v3"):
            counts[kind] = [hands_found(r) for r in detect_clip(frontend_frames(clip, kind))]
        rows.append({"path": path, "counts": counts})
        print("scanned", path, flush=True)
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(rows))
    return rows


def choose_detection_frames(rows):
    """Candidate frames: every learned front end finds more hands than bicubic (improved), or all agree on two (parity)."""
    improved, parity = [], []
    for row in rows:
        c = row["counts"]
        for t in range(2, 30):
            if c["clean"][t] == 2 and all(c[k][t] > c["Bicubic"][t] for k in ("SR-L1", "SR-GAN v2", "SR-GAN v3")):
                improved.append((row["path"], t))
            if all(c[k][t] == 2 for k in c):
                parity.append((row["path"], t))
    used = set()

    def raised(cands, n):
        found = []
        for path, t in cands:
            if path in used:
                continue
            record = detect_clip(frontend_frames(crops.resolve_video(path), "clean"))[t]
            ys = [(record[k][0][1] + record[k][0][3]) / 2 for k in ("left", "right") if record[k][0]]
            if len(ys) == 2 and max(ys) < 430:
                found.append((path, t))
                used.add(path)
            if len(found) == n:
                break
        return found

    return raised(improved, 8), raised(parity, 8)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def compute_row(path, t, kinds):
    clip = crops.resolve_video(path)
    results = {}
    for kind in kinds:
        frames = frontend_frames(clip, kind)
        results[kind] = (frames[t], detect_clip(frames)[t])
    return results


def hand_state(record, clean_record, side):
    """'ok' if the box overlaps the 1080p detection (IoU >= 0.3), 'wrong' if a box is elsewhere, else 'missed'."""
    box, source, _ = record[side]
    if box is None or source not in (crops.SOURCE_DIRECT, crops.SOURCE_ROI):
        return "missed"
    return "ok" if iou(box, clean_record[side][0]) >= 0.3 else "wrong"


def n_ok(res, kind):
    return sum(hand_state(res[kind][1], res["clean"][1], s) == "ok" for s in ("left", "right"))


def is_improved(res):
    return n_ok(res, "clean") == 2 and all(n_ok(res, k) > n_ok(res, "Bicubic") for k in ("SR-L1", "SR-GAN v2", "SR-GAN v3"))


def is_parity(res):
    return all(n_ok(res, k) == 2 for k in res)


def figure_detect(out, cands):
    kinds = ["clean", "Bicubic", "SR-L1", "SR-GAN v2", "SR-GAN v3"]
    heads = ["1080p source", "Bicubic", "SR-L1", "SR-GAN v2", "SR-GAN v3"]
    improved, parity = cands
    picks, used = [], set()
    for pool, need, test in ((improved, 2, is_improved), (parity, 1, is_parity)):
        got = 0
        for path, t in pool:
            if path in used:
                continue
            res = compute_row(path, t, kinds)
            if test(res):
                picks.append((path, t, res))
                used.add(path)
                got += 1
                print("accepted", path, t, flush=True)
            if got == need:
                break
    sheet = Sheet(W_IN, 0.4 + 1.5 * len(picks))
    tw, th, gap, left = 1.33, 1.0, 0.06, 0.24
    for c, head in enumerate(heads):
        sheet.text(left + c * (tw + gap) + tw / 2, 0.06, head, 6.5, ha="center", weight="bold")
    for r, (path, t, results) in enumerate(picks):
        boxes = [b for key in ("left", "right", "face") for b in [results["clean"][1][key][0]] if b]
        x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half_w = min(max((x1 - x0) * 0.62, (y1 - y0) * 0.62 * 4 / 3, 190), 480)
        half_h = half_w * 3 / 4
        cx = min(max(cx, half_w), 1280 - half_w); cy = min(max(cy, half_h), 720 - half_h)
        win = (int(cx - half_w), int(cy - half_h), int(cx + half_w), int(cy + half_h))
        y = 0.3 + r * (th + 0.5)
        for c, kind in enumerate(kinds):
            frame, record = results[kind]
            canvas = frame.copy()
            status = []
            for key in ("left", "right"):
                box, source, points = record[key]
                state = hand_state(record, results["clean"][1], key)
                if state != "missed":
                    cv2.rectangle(canvas, box[:2], box[2:], BGR[key], 4)
                    for px, py in points or []:
                        cv2.circle(canvas, (int(px), int(py)), 3, BGR[key], -1)
                status.append(f"{key[0].upper()} " + {"ok": "found", "wrong": "misplaced", "missed": "missed"}[state])
            cropped = canvas[win[1]:win[3], win[0]:win[2]]
            sheet.image(left + c * (tw + gap), y, tw, th, cv2.resize(cropped, (440, 330), interpolation=cv2.INTER_AREA))
            sheet.text(left + c * (tw + gap) + tw / 2, y + th + 0.03, "  ·  ".join(status), 6, ha="center",
                       color=RED if ("missed" in " ".join(status) or "misplaced" in " ".join(status)) else MUTE)
        word = plain(path.split("/")[1])
        sheet.text(0.1, y + th / 2, word, 6, ha="center", va="center", rot=90, color=MUTE)
    sheet.save(out / "fig_ex_detect.png")


def pick_gan_crops(rows):
    """Three test clips, each with a hand that the clean detector finds directly in the middle frame."""
    clips = test_clips()
    random.Random(8).shuffle(clips)
    picks = []
    for path in clips:
        clean = crops.read_sampled_frames(crops.resolve_video(path), 32)
        record = detect_clip(clean)[16]
        for side in ("right", "left"):
            box, source, _ = record[side]
            if source == crops.SOURCE_DIRECT and box is not None and 110 <= box_side(box) <= 330:
                picks.append((path, 16, side))
                break
        if len(picks) == 3:
            return picks
    return picks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="report/final/latex/figures")
    parser.add_argument("--only", default="all")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    crops.CROP_SIZE = 128
    if args.only in ("all", "dataset"):
        figure_dataset(out)
    if args.only in ("all", "crops"):
        figure_crops(out)
    if args.only in ("all", "gan"):
        figure_gan(out, pick_gan_crops(None))
    if args.only in ("all", "detect"):
        figure_detect(out, choose_detection_frames(scan_for_detection()))


if __name__ == "__main__":
    main()
