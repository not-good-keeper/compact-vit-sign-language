"""End-to-end test of the application's prediction route with several models.

This drives the same HTTP route the interface uses (POST /predict on the server
started by ``python -m islvit.demo``) with raw video bytes, and compares each
answer with the offline evaluation of the same model on the same clip.

What it establishes, and what it does not:

* It exercises the server path (decode, MediaPipe detection, cropping, landmark
  extraction, six-view TTA, 50-word restriction) for each model, which the
  browser screens call. It does not click through the browser screens.
* Agreement is measured on correctness (right / wrong) against the per-clip
  ``hits_masked_tta`` of each model's masked50_clean.json, because those files
  store hits, not predicted labels.
* The clip set is one held-out test clip per deployed word (the first by path),
  50 clips in all, drawn from the 472-clip test split. It is a sample for an
  integration check, not a replacement for the 472-clip evaluation.

Output: report/final/data/app_integration_<tag>.json

Usage (repository root, islvit environment, INCLUDE_raw/ present):
    python report/final/scripts/app_integration_test.py --run release/isl_vit_tiny_lm_int4_s0.pt --tag int4_lm_s0
    python report/final/scripts/app_integration_test.py --run runs/f16_262w_lm_s0 --tag fp32_lm_s0 \
        --offline runs/f16_262w_lm_s0/masked50_clean.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "report" / "final" / "data"
SPLIT = ROOT / "splits" / "vocab50clean__session-disjoint.csv"
PY = sys.executable


def clip_list(limit_per_word: int = 1) -> list[dict]:
    with SPLIT.open(encoding="utf-8") as h:
        rows = sorted((r for r in csv.DictReader(h) if r["split"] == "test"), key=lambda r: r["video_path"])
    seen: dict[str, int] = {}
    chosen = []
    for r in rows:
        if seen.get(r["label"], 0) < limit_per_word:
            seen[r["label"]] = seen.get(r["label"], 0) + 1
            chosen.append(r)
    return chosen


def wait_ready(port: int, proc: subprocess.Popen, timeout: float = 300) -> dict:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"server exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/info", timeout=5) as r:
                return json.loads(r.read())
        except OSError:
            time.sleep(2)
    raise SystemExit("server did not become ready")


def post(port: int, path: Path) -> tuple[dict, float]:
    data = path.read_bytes()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/predict", data=data, method="POST",
                                 headers={"X-Filename": path.name, "X-TTA": "1",
                                          "Content-Type": "application/octet-stream"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        body = json.loads(r.read())
    return body, time.perf_counter() - t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--offline", default=None, help="masked50_clean.json of the same model")
    ap.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    ap.add_argument("--port", type=int, default=7871)
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (latency runs)")
    ap.add_argument("--sr", default=None)
    ap.add_argument("--video-dir", default=None,
                    help="read <video_path>.mkv from this folder instead of INCLUDE_raw (degraded copies)")
    args = ap.parse_args()

    clips = clip_list()
    if args.limit:
        clips = clips[: args.limit]
    offline = None
    if args.offline:
        o = json.loads((ROOT / args.offline).read_text(encoding="utf-8"))
        offline = dict(zip(o["video_paths"], o["hits_masked_tta"]))

    cmd = [PY, "-m", "islvit.demo", "--run", args.run, "--port", str(args.port), "--device", args.device]
    if args.sr:
        cmd += ["--sr", args.sr]
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    log = (OUT_DIR / f"app_integration_{args.tag}.server.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        info = wait_ready(args.port, proc)
        results = []
        for r in clips:
            src = (Path(args.video_dir) / Path(r["video_path"]).with_suffix(".mkv")) if args.video_dir \
                else ROOT / "INCLUDE_raw" / r["video_path"]
            body, seconds = post(args.port, src)
            top = body["candidates"][0]["word"]
            row = {"video_path": r["video_path"], "label": r["label"], "top": top,
                   "correct": top == r["label"], "confidence": body["confidence"],
                   "in_top5": any(c["word"] == r["label"] for c in body["candidates"]),
                   "detection_rate": body["detection_rate"], "views": body["views"],
                   "frames": body["frames"], "deployed_only": body.get("deployed_only", False),
                   "seconds": seconds}
            if offline is not None:
                row["offline_correct"] = bool(offline[r["video_path"]])
            results.append(row)
            print(f"{r['label']:28s} -> {top:28s} {row['confidence']:.2f} {seconds:5.1f}s", flush=True)
    finally:
        proc.terminate()
        proc.wait(timeout=30)
        log.close()

    n = len(results)
    summary = {"run": args.run, "tag": args.tag, "device": args.device, "sr": args.sr,
               "server_info": {k: info.get(k) for k in ("run", "n_frames", "img_size", "params_m", "device")},
               "n_clips": n, "correct": sum(x["correct"] for x in results),
               "top5": sum(x["in_top5"] for x in results),
               "answered_at_tau_0.40": sum(x["confidence"] >= 0.40 for x in results),
               "all_views_6": all(x["views"] == 6 for x in results),
               "all_deployed_only": all(x["deployed_only"] for x in results),
               "median_seconds": sorted(x["seconds"] for x in results)[n // 2]}
    if offline is not None:
        summary["agree_with_offline"] = sum(x["correct"] == x["offline_correct"] for x in results)
        summary["offline_correct"] = sum(x["offline_correct"] for x in results)
    out = OUT_DIR / f"app_integration_{args.tag}.json"
    out.write_text(json.dumps({"summary": summary, "clips": results}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
