"""Build the demo clip set the ISL Reader app offers when run with islvit.demo.

The clips are held-out test clips (splits/_test472_paths.csv: recording sessions the
model never trained on) from the INCLUDE dataset (Sridhar et al., ACM MM 2020,
CC-BY-4.0). Each one is transcoded to a 720p H.264 MP4 that any browser plays, and
that MP4 -- not the original -- is what the app sends to the model, so what the
audience sees is exactly what is classified.

Selection, in order:
1. candidates: test clips all three released INT4 models get right (masked-50, 6-view TTA),
   one clip per word, spread across the dataset's categories;
2. each candidate is sent to the running demo server (/predict, all 262 words, as the app
   does) and kept only if the server's top word is the right one, above the app's
   confidence threshold;
3. optionally one "hard case": a look-alike word the server is unsure about, to show the
   Not-sure flow honestly.

    python -m islvit.demo                          # in one terminal
    python -m islvit.demo_samples --count 10        # writes demo_samples/ and its manifest
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import av
import cv2
import numpy as np

from islvit.data import crops

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "demo_samples"
SEEDS = (0, 1, 2)
LOOKALIKES = {"you", "you (plural)", "big large", "wide", "narrow", "short", "small little", "good", "bad", "wet",
              "Dress", "T-Shirt"}


def word_of(label: str) -> str:
    return re.sub(r"^\d+\.\s*", "", label)


def candidates() -> list[tuple[str, str]]:
    """(video_path, label) of test clips every released seed gets right, grouped so categories alternate."""
    results = [json.loads((ROOT / f"runs/f16_262w_lm_s{s}_qat4/masked50_clean.json").read_text()) for s in SEEDS]
    hits = [dict(zip(r["video_paths"], r["hits_masked_tta"])) for r in results]
    by_category = defaultdict(dict)
    for path in results[0]["video_paths"]:
        if all(h[path] for h in hits):
            category, label = path.split("/")[0], path.split("/")[1]
            by_category[category].setdefault(label, path)          # first clip of each word
    out, pools = [], [list(v.items()) for _, v in sorted(by_category.items())]
    while any(pools):                                               # round-robin over categories
        for pool in pools:
            if pool:
                label, path = pool.pop(0)
                out.append((path, label))
    return out


def transcode(src: Path, dst: Path, height: int = 720) -> None:
    """Source video -> H.264/yuv420p MP4 at ``height`` (no audio, moov atom up front)."""
    with av.open(str(src)) as inp:
        stream = inp.streams.video[0]
        rate = stream.average_rate or 25
        with av.open(str(dst), "w", options={"movflags": "+faststart"}) as out:
            enc = None
            for frame in inp.decode(stream):
                img = frame.to_ndarray(format="bgr24")
                h, w = img.shape[:2]
                if h > height:
                    img = cv2.resize(img, (int(round(w * height / h)) // 2 * 2, height), interpolation=cv2.INTER_AREA)
                if enc is None:
                    enc = out.add_stream("libx264", rate=rate)
                    enc.width, enc.height, enc.pix_fmt = img.shape[1], img.shape[0], "yuv420p"
                    enc.options = {"crf": "18", "preset": "medium"}
                for packet in enc.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
                    out.mux(packet)
            for packet in enc.encode():
                out.mux(packet)


def poster(video: Path, dst: Path) -> None:
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2))
    ok, frame = cap.read()
    cap.release()
    if ok:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (320, int(round(h * 320 / w))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), frame, [cv2.IMWRITE_JPEG_QUALITY, 82])


def predict(server: str, video: Path) -> dict:
    request = urllib.request.Request(f"{server}/predict", data=video.read_bytes(), method="POST",
                                     headers={"X-Filename": video.name, "X-TTA": "1"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def lookalike_clips() -> list[tuple[str, str]]:
    """Every test clip of a look-alike word, in file order (the hard-case search space)."""
    with (ROOT / "splits/_test472_paths.csv").open(encoding="utf-8") as h:
        test = [r["video_path"] for r in csv.DictReader(h)]
    out = []
    for path in test:
        label = path.split("/")[1]
        if word_of(label) in LOOKALIKES:
            out.append((path, label))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the demo clip set")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--server", default="http://127.0.0.1:7860")
    parser.add_argument("--tau", type=float, default=0.40, help="the app's default confidence threshold")
    parser.add_argument("--hard-case", action="store_true",
                        help="make the last clip a look-alike word the model is unsure on, with the right word "
                             "still in its shortlist, so the demo can show the Not-sure flow")
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    for old in list(OUT.glob("*.mp4")) + list(OUT.glob("*.jpg")):
        old.unlink()
    want = args.count - (1 if args.hard_case else 0)
    kept, hard, tried = [], None, 0

    def attempt(path, label):
        nonlocal tried
        src = crops.resolve_video(path)
        if src is None:
            return None
        stem = f"{tried + 1:02d}_" + re.sub(r"[^A-Za-z0-9]+", "_", word_of(label)).strip("_").lower()
        mp4 = OUT / f"{stem}.mp4"
        transcode(src, mp4)
        tried += 1
        result = predict(args.server, mp4)
        top = result["candidates"][0]
        entry = {"file": mp4.name, "poster": mp4.with_suffix(".jpg").name, "label": label, "word": word_of(label),
                 "category": path.split("/")[0], "source": f"INCLUDE {path}", "server_top": top["word"],
                 "server_p": round(top["p"], 4), "in_shortlist": label in [c["word"] for c in result["candidates"]]}
        return mp4, entry

    # 1. confident, correct clips from words all three released seeds get right
    pool = candidates()
    print(f"{len(pool)} candidate words where all three seeds are right")
    for path, label in pool:
        if len(kept) >= want:
            break
        got = attempt(path, label)
        if not got:
            continue
        mp4, entry = got
        ok = entry["server_top"] == label and entry["server_p"] >= args.tau
        print(f"  {entry['word']:16s} -> {word_of(entry['server_top']):16s} {100 * entry['server_p']:5.1f}%  {'kept' if ok else 'skipped'}")
        if ok:
            poster(mp4, mp4.with_suffix(".jpg"))
            kept.append(entry)
        else:
            mp4.unlink(missing_ok=True)

    # 2. one hard case: a look-alike word below the threshold whose right word is still in the top five
    if args.hard_case:
        for path, label in lookalike_clips():
            got = attempt(path, label)
            if not got:
                continue
            mp4, entry = got
            ok = entry["server_p"] < args.tau and entry["in_shortlist"]
            print(f"  hard case? {entry['word']:12s} -> {word_of(entry['server_top']):12s} {100 * entry['server_p']:5.1f}%  "
                  f"right word in top 5: {entry['in_shortlist']}  {'kept' if ok else 'skipped'}")
            if ok:
                poster(mp4, mp4.with_suffix(".jpg"))
                hard = {**entry, "hard_case": True}
                break
            mp4.unlink(missing_ok=True)

    samples = kept + ([hard] if hard else [])
    manifest = {"note": "Held-out INCLUDE test clips (Sridhar et al., ACM MM 2020, CC-BY-4.0), transcoded to 720p. "
                        "The confident clips are words all three released models classify correctly; the hard case "
                        "is a look-alike word the model is unsure on.",
                "tau": args.tau, "samples": samples}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(samples)} samples to {OUT} (tried {tried})")


if __name__ == "__main__":
    main()
