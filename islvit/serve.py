"""A local web UI that runs the real model on a real video.

Started with the standard library only -- no Flask, no FastAPI, no npm. A panel
or a collaborator can run this straight from a clone with nothing installed
beyond what training already needs.

Two decisions worth stating:

**It reuses the training preprocessing verbatim.** ``predict.extract`` and
``predict.views`` are the same functions that built the crop cache, so the
browser path and the benchmark path cannot drift. Writing a separate inference
preprocessor is the standard way a model that scores well offline fails on a
real clip.

**It defaults to CPU.** Loading a second model onto a GPU that is mid-training
is how a multi-hour run died once already in this project. Inference is ~24 ms
per view on a desktop CPU, so six TTA views finish well inside a second; pass
``--device cuda`` only when nothing else is using the card.

Uploads arrive as the raw request body rather than multipart form data, which
keeps the parser to two lines and removes a whole class of encoding bug.

Usage::

    python -m islvit.serve --run runs/f16_clean_s0
    python -m islvit.serve --run runs/f16_clean_s0 --port 8000 --device cuda
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from islvit.eval import load_run
from islvit.models.isl_vit import count_parameters
from islvit.predict import extract, extract_landmarks, views

STATE: dict = {}


def classify(video_path: Path, use_tta: bool = True) -> dict:
    """Run one video through the model and report everything the UI needs."""
    model, config, classes = STATE["model"], STATE["config"], STATE["classes"]
    device = STATE["device"]

    clip, detected, geometry, sources = extract(video_path, STATE["crop_size"])
    landmarks = extract_landmarks(video_path, interp=config.get("lm_interp", False)) \
        if config.get("landmarks") else None

    # Detection rate on the two hand streams is the single best predictor of
    # whether a prediction means anything -- with no hands found the model is
    # classifying background. The UI shows it next to the answer for that reason.
    hands = sources[:, :2]
    genuine = float(((hands == 1) | (hands == 2)).mean())

    # Per-stream provenance, which the diagnostics screen breaks out. A stream
    # sitting at 80 % "interpolated" looks healthy under a single coverage number
    # but is mostly stale boxes carried from neighbouring frames.
    names = ("missing", "direct", "roi", "interp")
    provenance = {}
    for index, stream in enumerate(("left_hand", "right_hand", "face")):
        counts = np.bincount(sources[:, index].astype(int), minlength=4)
        share = counts / max(1, counts.sum())
        provenance[stream] = {name: float(value) for name, value in zip(names, share)}
        provenance[stream]["genuine"] = float(share[1] + share[2])

    total, n = None, 0
    with torch.no_grad():
        for crops, det, geo, extras in views(clip, detected, geometry,
                                             config["n_frames"], config["img_size"], use_tta,
                                             landmarks):
            extras = {key: value.to(device) for key, value in extras.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(crops.to(device), det.to(device), geo.to(device), **extras)
            total = logits.float().softmax(1) if total is None else total + logits.float().softmax(1)
            n += 1
    probability = (total / n).squeeze(0).cpu()

    top = probability.topk(min(5, len(classes)))
    return {
        "frames": int(clip.shape[0]),
        "views": n,
        "detection_rate": genuine,
        "provenance": provenance,
        "confidence": float(probability.max()),
        "candidates": [
            {"word": classes[i], "p": float(p)}
            for p, i in zip(top.values.tolist(), top.indices.tolist())
        ],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter console
        print(f"  {self.address_string()} {fmt % args}", flush=True)

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict):
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/info":
            self._json(200, STATE["info"])
        else:
            self._json(404, {"error": "not found"})

    def _enrol(self, data: bytes, suffix: str):
        """Save a recorded take under custom/<signer>/<word>/ for later ingest.

        This is the layout islvit.data.ingest_custom and crops.py --corpus custom
        already expect, so a take recorded here needs no renaming to become
        training or signer-disjoint evaluation data.
        """
        signer = (self.headers.get("X-Signer") or "").strip()
        word = (self.headers.get("X-Word") or "").strip()
        if not signer or not word:
            self._json(400, {"error": "X-Signer and X-Word headers are required"})
            return
        # Path components come from the browser, so keep them to a safe charset
        # rather than trusting them into a filesystem path.
        safe = lambda s: "".join(c for c in s if c.isalnum() or c in " _-.").strip()
        signer, word = safe(signer), safe(word)
        if not signer or not word:
            self._json(400, {"error": "signer and word must contain usable characters"})
            return
        root = Path("custom").resolve()
        folder = (root / signer / word).resolve()
        # Charset filtering alone left "../../etc" as "....etc" -- harmless here,
        # but the property that actually matters is containment, so assert it
        # against the resolved path rather than trusting the sanitiser.
        if root not in folder.parents and folder != root:
            self._json(400, {"error": "signer/word must stay inside custom/"})
            return
        folder.mkdir(parents=True, exist_ok=True)
        take = len(list(folder.glob("take*"))) + 1
        path = folder / f"take{take}{suffix}"
        path.write_bytes(data)
        self._json(200, {"saved": str(path), "take": take,
                         "takes_for_word": len(list(folder.glob("take*")))})

    def do_POST(self):
        if self.path not in ("/predict", "/enrol", "/correct"):
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            self._json(400, {"error": "empty upload"})
            return
        suffix = Path(self.headers.get("X-Filename", "clip.mp4")).suffix or ".mp4"
        data = self.rfile.read(length)

        # OpenCV needs a path, so the bytes land in a temp file that is removed
        # whether or not decoding succeeds.
        if self.path == "/enrol":
            self._enrol(data, suffix)
            return
        if self.path == "/correct":
            # A user correcting a prediction is the cheapest labelled example this
            # project can get, so it is written to disk rather than only logged.
            try:
                record = json.loads(data.decode("utf-8"))
            except ValueError:
                self._json(400, {"error": "correction must be JSON"})
                return
            record["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            log = Path("runs") / "corrections.jsonl"
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            self._json(200, {"logged": str(log)})
            return

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(data)
            path = Path(handle.name)
        try:
            self._json(200, classify(path, use_tta=self.headers.get("X-TTA", "1") == "1"))
        except SystemExit as exc:
            # extract() raises SystemExit when a clip cannot be decoded or no
            # frames come back. That is a user-facing condition, not a crash.
            self._json(422, {"error": str(exc)})
        except Exception:
            traceback.print_exc()
            self._json(500, {"error": "inference failed; see server console"})
        finally:
            path.unlink(missing_ok=True)


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ISL Reader</title>
<style>
:root{
 --ink:#12141b;--ink2:#3d4356;--muted:#6b7285;--faint:#98a0b3;
 --line:#e2e4ec;--line2:#c8ccd8;--bg:#eef0f5;--card:#fff;--card2:#f6f7fb;
 --brand:#0d5c5b;--brand2:#0a4746;--wash:#e2efee;
 --pos:#1d6b3f;--posw:#e5f1e9;--warn:#8a5a0c;--warnw:#f8f0dc;--neg:#9a352f;--negw:#f9e7e5;
 --serif:Georgia,"Times New Roman",serif;
 --sans:"Segoe UI",system-ui,-apple-system,Roboto,Arial,sans-serif;
 --mono:"Cascadia Mono",Consolas,Menlo,monospace;
 --sh:0 1px 2px rgba(18,20,27,.04),0 10px 30px -18px rgba(18,20,27,.3);
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}
.app{display:grid;grid-template-columns:232px 1fr;min-height:100vh}
@media(max-width:900px){.app{grid-template-columns:1fr}aside{position:static!important;height:auto!important}}

/* ---------- rail ---------- */
aside{background:var(--card);border-right:1px solid var(--line);padding:20px 14px;position:sticky;top:0;height:100vh;overflow:auto}
.brand{display:flex;align-items:baseline;gap:8px;padding:0 8px 16px;border-bottom:1px solid var(--line);margin-bottom:14px}
.brand h1{font-family:var(--serif);font-size:21px;font-weight:600;letter-spacing:-.01em}
.brand span{font-family:var(--mono);font-size:10px;color:var(--brand);background:var(--wash);padding:2px 6px;border-radius:3px}
nav button{display:grid;grid-template-columns:22px 1fr;gap:9px;align-items:center;width:100%;text-align:left;
 background:none;border:0;border-radius:6px;padding:9px 9px;cursor:pointer;color:var(--ink2);font-size:13.5px;font-family:inherit}
nav button:hover{background:var(--card2);color:var(--ink)}
nav button.on{background:var(--wash);color:var(--brand);font-weight:650}
nav button .n{font-family:var(--mono);font-size:10.5px;color:var(--faint)}
nav button.on .n{color:var(--brand)}
nav button:focus-visible{outline:2px solid var(--brand);outline-offset:-2px}
.railfoot{margin-top:16px;padding:10px 9px;border-top:1px solid var(--line);font-family:var(--mono);font-size:10.5px;color:var(--faint);line-height:1.7}
.railfoot b{color:var(--ink2);font-weight:600}
.dotlive{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--faint);margin-right:5px}
.dotlive.on{background:var(--pos)}

/* ---------- main ---------- */
main{padding:26px 30px 60px;max-width:1180px}
.screen{display:none}
.screen.on{display:block;animation:in .18s ease-out}
@keyframes in{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
@media(prefers-reduced-motion:reduce){.screen.on{animation:none}}
.head{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;margin-bottom:18px;padding-bottom:12px;border-bottom:2px solid var(--brand)}
.head h2{font-family:var(--serif);font-size:25px;font-weight:600;letter-spacing:-.01em}
.head .sub{font-size:13px;color:var(--muted);margin-left:auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:18px 20px;margin-bottom:16px;box-shadow:var(--sh)}
h3{font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-bottom:11px}
.cols{display:grid;grid-template-columns:1.1fr .9fr;gap:16px;align-items:start}
@media(max-width:1000px){.cols{grid-template-columns:1fr}}

/* ---------- controls ---------- */
button.b{font-family:var(--sans);font-size:13.5px;border:1px solid var(--line2);background:#fff;color:var(--ink2);
 border-radius:7px;padding:9px 15px;cursor:pointer;transition:.12s}
button.b:hover:not(:disabled){border-color:var(--brand);color:var(--brand);background:var(--wash)}
button.b.primary{background:var(--brand);border-color:var(--brand);color:#fff}
button.b.primary:hover:not(:disabled){background:var(--brand2)}
button.b.rec{background:var(--neg);border-color:var(--neg);color:#fff}
button.b:disabled{opacity:.4;cursor:not-allowed}
button.b:focus-visible{outline:2px solid var(--brand);outline-offset:2px}
.row{display:flex;gap:9px;flex-wrap:wrap;align-items:center}
input[type=range]{width:100%;accent-color:var(--brand)}
input[type=text]{font-family:var(--sans);font-size:13.5px;border:1px solid var(--line2);border-radius:6px;padding:8px 11px;width:100%}
input[type=text]:focus{outline:2px solid var(--brand);outline-offset:-1px;border-color:var(--brand)}
select{font-family:var(--sans);font-size:13.5px;border:1px solid var(--line2);border-radius:6px;padding:8px 10px;background:#fff}
label.sw{display:flex;align-items:center;gap:7px;font-size:13.5px;color:var(--ink2);cursor:pointer}
.tl{display:flex;justify-content:space-between;font-size:11.5px;color:var(--muted);margin-top:3px}

/* ---------- drop / video ---------- */
#drop{border:2px dashed var(--line2);border-radius:8px;padding:22px 16px;text-align:center;background:var(--card2);cursor:pointer;transition:.15s}
#drop:hover,#drop.hot{border-color:var(--brand);background:var(--wash)}
#drop .big{font-size:14.5px;font-weight:600;margin-bottom:3px}
#drop .sm{font-size:12.5px;color:var(--muted)}
video{width:100%;border-radius:7px;background:#0d0f14;display:block;max-height:280px;object-fit:contain}

/* ---------- crop health ---------- */
.health{display:grid;gap:9px;margin-top:4px}
.hrow{display:grid;grid-template-columns:78px 1fr 44px;gap:10px;align-items:center;font-size:12.5px}
.hrow .nm{color:var(--ink2)}
.hbar{height:8px;border-radius:5px;background:var(--card2);overflow:hidden;display:flex}
.hbar i{display:block;height:100%}
.hbar .d{background:var(--pos)} .hbar .r{background:#5aa27a} .hbar .p{background:var(--warn)} .hbar .m{background:var(--neg)}
.hrow .pc{font-family:var(--mono);font-size:11.5px;text-align:right;color:var(--muted)}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:var(--muted);margin-top:9px}
.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px;vertical-align:-1px}

/* ---------- result ---------- */
.word{font-family:var(--serif);font-size:42px;font-weight:600;line-height:1.05}
.conf{font-family:var(--mono);font-size:16px;color:var(--brand);float:right;margin-top:14px}
.bar{height:8px;border-radius:5px;background:var(--card2);overflow:hidden;margin:9px 0 18px}
.bar i{display:block;height:100%;background:var(--brand);border-radius:5px;transition:width .35s}
.alt{display:grid;grid-template-columns:1fr 52px;gap:10px;align-items:center;margin-bottom:9px;font-size:13.5px}
.alt .ab{height:5px;border-radius:3px;background:var(--card2);overflow:hidden;margin-top:4px}
.alt .ab i{display:block;height:100%;background:var(--brand);opacity:.5;border-radius:3px}
.alt .pv{font-family:var(--mono);font-size:12px;color:var(--muted);text-align:right}
.abstain{text-align:center;padding:16px 10px}
.abstain .q{font-family:var(--serif);font-size:46px;color:var(--warn);line-height:1}
.abstain .t{font-family:var(--serif);font-size:21px;margin:9px 0 4px}
.abstain .s{font-size:13px;color:var(--muted)}
.pickrow{display:flex;gap:8px;flex-wrap:wrap;justify-content:center;margin-top:16px}

/* ---------- strip / chips ---------- */
.strip{display:flex;gap:7px;flex-wrap:wrap;align-items:center;min-height:38px;padding:9px 11px;background:var(--card2);border:1px solid var(--line);border-radius:7px}
.chip{font-family:var(--serif);font-size:15px;background:var(--wash);border:1px solid var(--brand);color:var(--ink);border-radius:14px;padding:3px 12px}
.chip.ab{background:var(--warnw);border-color:var(--warn);color:var(--warn);font-family:var(--sans);font-size:12.5px}
.strip .ph{color:var(--faint);font-size:13px;font-style:italic}

/* ---------- notes ---------- */
.note{border-left:3px solid var(--brand);background:var(--wash);padding:10px 13px;border-radius:0 6px 6px 0;font-size:12.8px;color:var(--ink2);margin-top:12px}
.note.warn{border-left-color:var(--warn);background:var(--warnw)}
.note.neg{border-left-color:var(--neg);background:var(--negw)}
.note.pos{border-left-color:var(--pos);background:var(--posw)}
.note b{color:var(--ink)}

/* ---------- tables / vocab ---------- */
table{border-collapse:collapse;width:100%;font-size:12.8px}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line)}
th{font-family:var(--mono);font-size:9.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:600}
td.n{text-align:right;font-family:var(--mono)}
tr.ab td{color:var(--warn)}
.vgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(178px,1fr));gap:8px}
.vc{border:1px solid var(--line);border-radius:7px;padding:9px 11px;background:var(--card2)}
.vc .w{font-family:var(--serif);font-size:15px;margin-bottom:5px}
.vc .rb{height:5px;border-radius:3px;background:#e7e9f0;overflow:hidden}
.vc .rb i{display:block;height:100%}
.vc .mt{display:flex;justify-content:space-between;font-family:var(--mono);font-size:10.5px;color:var(--muted);margin-top:4px}
.g{background:var(--pos)}.y{background:var(--warn)}.r{background:var(--neg)}
.empty{font-size:12.8px;color:var(--faint);font-style:italic;padding:8px 0}
.stat{display:flex;gap:20px;flex-wrap:wrap}
.stat div{font-family:var(--mono);font-size:11.5px;color:var(--muted)}
.stat b{display:block;font-family:var(--sans);font-size:21px;color:var(--ink);font-weight:650}

/* ---------- enrolment ---------- */
.eg{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:7px;max-height:290px;overflow:auto;padding-right:4px}
.ec{border:1px solid var(--line);border-radius:6px;padding:7px 9px;background:var(--card2);cursor:pointer;font-size:12.5px}
.ec:hover{border-color:var(--brand);background:var(--wash)}
.ec.on{border-color:var(--brand);background:var(--wash);box-shadow:inset 0 0 0 1px var(--brand)}
.ec .w{margin-bottom:4px}
.pips{display:flex;gap:3px}
.pips i{width:100%;height:4px;border-radius:2px;background:#e0e3ec}
.pips i.f{background:var(--pos)}
.stages{margin-top:12px;display:none}
.stages.on{display:block}
.stage{display:flex;align-items:center;gap:9px;font-size:12.8px;color:var(--muted);padding:3px 0}
.dot{width:9px;height:9px;border-radius:50%;border:1.5px solid var(--line2);flex:0 0 auto}
.stage.done .dot{background:var(--pos);border-color:var(--pos)}
.stage.done{color:var(--ink2)}
.stage.now .dot{background:var(--brand);border-color:var(--brand);animation:pulse 1s infinite}
.stage.now{color:var(--brand);font-weight:600}
@keyframes pulse{50%{opacity:.3}}
@media(prefers-reduced-motion:reduce){.stage.now .dot{animation:none}}
</style></head><body>

<div class="app">
<aside>
  <div class="brand"><h1>ISL Reader</h1><span>v1</span></div>
  <nav id="nav">
    <button data-s="live" class="on"><span class="n">1</span>Live capture</button>
    <button data-s="notsure"><span class="n">2</span>Not sure</button>
    <button data-s="compose"><span class="n">3</span>Sentence composer</button>
    <button data-s="settings"><span class="n">4</span>Settings</button>
    <button data-s="diag"><span class="n">5</span>Detection diagnostics</button>
    <button data-s="vocab"><span class="n">6</span>Vocabulary browser</button>
    <button data-s="enrol"><span class="n">7</span>Enrolment / capture</button>
  </nav>
  <div class="railfoot" id="railfoot">loading…</div>
</aside>

<main>

<!-- ============ 1 · LIVE CAPTURE ============ -->
<section class="screen on" id="s-live">
  <div class="head"><h2>Live capture</h2><span class="sub" id="liveSub"></span></div>
  <div class="cols">
    <div>
      <div class="card">
        <h3>Camera preview</h3>
        <video id="preview" playsinline muted></video>
        <div class="row" style="margin-top:12px">
          <button class="b" id="btnCam">Turn camera on</button>
          <button class="b rec" id="btnRec" disabled>● Record 3 s</button>
          <button class="b" id="btnFile">Upload a clip</button>
        </div>
        <input type="file" id="file" accept="video/*" hidden>
        <div id="drop" tabindex="0" role="button" style="margin-top:12px">
          <div class="big">…or drop a video here</div>
          <div class="sm">.mp4 · .mov · .webm — one isolated sign per clip</div>
        </div>
        <div class="stages" id="stages">
          <div class="stage" data-k="0"><span class="dot"></span>Upload</div>
          <div class="stage" data-k="1"><span class="dot"></span>Detect hands &amp; face</div>
          <div class="stage" data-k="2"><span class="dot"></span>Crop 3 streams, sample frames</div>
          <div class="stage" data-k="3"><span class="dot"></span>Encode &amp; score</div>
        </div>
      </div>

      <div class="card">
        <h3>Crop health — this clip</h3>
        <div class="health" id="health"><div class="empty">No clip processed yet.</div></div>
        <div class="legend">
          <span><i class="d" style="background:var(--pos)"></i>direct</span>
          <span><i class="r" style="background:#5aa27a"></i>ROI-rescued</span>
          <span><i class="p" style="background:var(--warn)"></i>interpolated</span>
          <span><i class="m" style="background:var(--neg)"></i>missing</span>
        </div>
      </div>
    </div>

    <div>
      <div class="card">
        <h3>Recognised word</h3>
        <div id="result"><div class="empty">Record or upload a clip to begin.</div></div>
        <div id="warn"></div>
      </div>
      <div class="card">
        <h3>Sentence strip</h3>
        <div class="strip" id="strip"><span class="ph">words you accept land here</span></div>
        <div class="row" style="margin-top:11px">
          <button class="b" id="stripUndo">Undo last</button>
          <button class="b" id="stripClear">Clear</button>
          <button class="b primary" id="stripGo" style="margin-left:auto">Open composer →</button>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ============ 2 · NOT SURE ============ -->
<section class="screen" id="s-notsure">
  <div class="head"><h2>Not sure</h2><span class="sub">what happens below the confidence threshold</span></div>
  <div class="cols">
    <div class="card">
      <h3>Current state</h3>
      <div id="nsBody"><div class="empty">Nothing pending. This screen fills when a clip scores below τ.</div></div>
    </div>
    <div>
      <div class="card">
        <h3>Why this screen exists</h3>
        <div class="note"><b>Confidence gating is part of the interface, not a post-hoc filter.</b>
        At τ = 0.40 the model answers roughly three clips in four; on the rest it says so rather
        than guessing. Measured on the held-out set, accuracy on the clips it does answer rises
        from 75.8 % to 90.7 %.</div>
        <div class="note warn"><b>It is not perfect.</b> Confident errors still get through —
        on a spot check of held-out clips, two wrong answers scored 77 % and 59 %. The gate
        reduces confident mistakes; it does not eliminate them.</div>
      </div>
      <div class="card">
        <h3>Corrections collected</h3>
        <div class="stat"><div><b id="nCorr">0</b>this session</div><div><b id="nCorrTot">—</b>logged to disk</div></div>
        <div class="note pos">Every correction is appended to <code>runs/corrections.jsonl</code> with the
        clip's confidence and top-5. A user fixing a wrong answer is the cheapest labelled example
        this project can obtain.</div>
      </div>
    </div>
  </div>
</section>

<!-- ============ 3 · SENTENCE COMPOSER ============ -->
<section class="screen" id="s-compose">
  <div class="head"><h2>Sentence composer</h2><span class="sub">word-level assembly — not grammatical translation</span></div>
  <div class="cols">
    <div>
      <div class="card">
        <h3>Sequence</h3>
        <div class="strip" id="seq" style="min-height:52px"><span class="ph">nothing composed yet</span></div>
        <div class="row" style="margin-top:12px">
          <label class="sw"><input type="checkbox" id="dedup" checked> Collapse immediate repeats</label>
          <button class="b" id="seqSpeak" style="margin-left:auto">Speak</button>
          <button class="b" id="seqCopy">Copy</button>
          <button class="b" id="seqExport">Export JSON</button>
        </div>
        <div class="note warn"><b>No grammar translation yet.</b> This is the recognised word
        sequence in the order it was signed. ISL word order is not English word order, so the
        strip above is not an English sentence — the translation module is future work.</div>
      </div>
    </div>
    <div class="card">
      <h3>Word-level history</h3>
      <table><thead><tr><th>Word</th><th class="n">Conf</th><th class="n">Hands</th><th class="n">Time</th></tr></thead>
      <tbody id="hist"><tr><td colspan="4" class="empty">Nothing yet.</td></tr></tbody></table>
    </div>
  </div>
</section>

<!-- ============ 4 · SETTINGS ============ -->
<section class="screen" id="s-settings">
  <div class="head"><h2>Settings</h2><span class="sub">every control here changes a measured trade-off</span></div>
  <div class="cols">
    <div>
      <div class="card">
        <h3>Confidence threshold τ</h3>
        <input type="range" id="tau" min="0" max="90" step="5" value="40">
        <div class="tl"><span>answers more</span><span id="tauVal">τ = 0.40</span><span>answers surer</span></div>
        <div class="note" id="tauNote"></div>
      </div>
      <div class="card">
        <h3>Test-time augmentation</h3>
        <div class="row">
          <label class="sw"><input type="radio" name="tta" value="6" checked> 6 views</label>
          <label class="sw"><input type="radio" name="tta" value="1"> 1 view (faster)</label>
        </div>
        <div class="note">Six views = 3 temporal phases × 2 horizontal flips, averaged.
        Measured at <b>+3.2 points</b> top-1 on the held-out set, for six forward passes
        instead of one.</div>
      </div>
      <div class="card">
        <h3>Detector health floor</h3>
        <input type="range" id="detFloor" min="0" max="90" step="5" value="25">
        <div class="tl"><span>never warn</span><span id="detVal">warn below 25 %</span><span>strict</span></div>
        <div class="note">Below this genuine-detection rate the result is flagged as unreliable.
        On INCLUDE the detector finds a real hand in 91 % (left) and 85 % (right) of frames.</div>
      </div>
    </div>
    <div>
      <div class="card">
        <h3>Vocabulary tier</h3>
        <select id="tier"></select>
        <div class="note warn" id="tierNote"><b>Only the loaded model can serve requests.</b>
        Switching tiers needs the server restarted with a different <code>--run</code>; this
        list shows which trained runs exist on disk.</div>
      </div>
      <div class="card">
        <h3>Ensemble</h3>
        <div class="note warn"><b>Not available in this server.</b> One model is loaded.
        The 75.8 % headline is a five-model ensemble at ~18.5 MB; the single model
        loaded here measures <b>73.3 %</b> with TTA. Running five models on CPU would take
        roughly five times as long per clip.</div>
      </div>
      <div class="card">
        <h3>Loaded model</h3>
        <table><tbody id="modelTbl"></tbody></table>
      </div>
    </div>
  </div>
</section>

<!-- ============ 5 · DIAGNOSTICS ============ -->
<section class="screen" id="s-diag">
  <div class="head"><h2>Detection diagnostics</h2><span class="sub">where the crops actually came from</span></div>
  <div class="cols">
    <div class="card">
      <h3>Provenance across this session</h3>
      <div class="health" id="diagHealth"><div class="empty">No clips processed yet.</div></div>
      <div class="legend">
        <span><i style="background:var(--pos)"></i>direct — full-frame detector found it</span>
        <span><i style="background:#5aa27a"></i>ROI-rescued — second pass on an upscaled crop</span>
        <span><i style="background:var(--warn)"></i>interpolated — box carried from a neighbour</span>
        <span><i style="background:var(--neg)"></i>missing</span>
      </div>
      <div class="note" id="diagNote"></div>
    </div>
    <div>
      <div class="card">
        <h3>Per-clip breakdown</h3>
        <table><thead><tr><th>#</th><th class="n">L hand</th><th class="n">R hand</th><th class="n">Face</th><th class="n">Result</th></tr></thead>
        <tbody id="diagTbl"><tr><td colspan="5" class="empty">Nothing yet.</td></tr></tbody></table>
      </div>
      <div class="card">
        <h3>Why this matters</h3>
        <div class="note"><b>An interpolated box is a stale box.</b> It shows where a hand
        recently was, usually background the hand has already left. A stream sitting at 80 %
        interpolated looks fine under a single coverage number and is mostly not hand.</div>
        <div class="note neg">Measured consequence: adding 76 labelled clips from a corpus where
        the detector fires on only 25 % of left-hand frames <b>cost 5.7 points</b>. Detection
        quality dominates data quantity.</div>
      </div>
    </div>
  </div>
</section>

<!-- ============ 6 · VOCABULARY ============ -->
<section class="screen" id="s-vocab">
  <div class="head"><h2>Vocabulary browser</h2><span class="sub" id="vocabSub"></span></div>
  <div class="card">
    <div class="row" style="margin-bottom:14px">
      <input type="text" id="vFilter" placeholder="Filter words…" style="max-width:260px">
      <label class="sw"><input type="checkbox" id="vWeak"> Weak classes only (&lt; 50 % recall)</label>
      <span class="sub" style="margin-left:auto;font-size:12.5px;color:var(--muted)" id="vCount"></span>
    </div>
    <div class="vgrid" id="vgrid"></div>
    <div class="note warn" id="vNote"></div>
  </div>
</section>

<!-- ============ 7 · ENROLMENT ============ -->
<section class="screen" id="s-enrol">
  <div class="head"><h2>Enrolment / capture</h2><span class="sub">guided recording — 3 takes per word</span></div>
  <div class="cols">
    <div>
      <div class="card">
        <h3>Signer</h3>
        <input type="text" id="signer" placeholder="signer id, e.g. kushal">
        <div class="note"><b>Why the signer id matters.</b> INCLUDE ships no signer labels, so
        signer-independence can only be approximated on it by clustering recording sessions.
        Takes recorded here are stored per signer, which makes a genuinely signer-disjoint
        test set possible for the first time.</div>
      </div>
      <div class="card">
        <h3>Record</h3>
        <video id="ePreview" playsinline muted></video>
        <div class="row" style="margin-top:11px">
          <button class="b" id="eCam">Turn camera on</button>
          <button class="b rec" id="eRec" disabled>● Record take</button>
          <span id="eTarget" style="margin-left:auto;font-size:13px;color:var(--muted)">pick a word →</span>
        </div>
        <div id="eStatus"></div>
      </div>
    </div>
    <div class="card">
      <h3>Progress — <span id="eProg">0 / 0</span> takes</h3>
      <div class="eg" id="egrid"></div>
      <div class="note"><b>Target: 3 signers × 50 words × 3 takes ≈ 450 clips.</b>
      Files are written to <code>custom/&lt;signer&gt;/&lt;word&gt;/takeN.webm</code>, which is the
      layout <code>islvit.data.ingest_custom</code> already reads.</div>
    </div>
  </div>
</section>

</main>
</div>

<script>
(function(){
"use strict";
var $=function(id){return document.getElementById(id)};
var info=null, tau=0.40, ttaViews=6, detFloor=0.25;
var hist=[], pending=null, strip=[], corrections=0, enrolCounts={}, enrolWord=null;
var camStream=null, eStream=null;

/* ---------- boot ---------- */
fetch("/info").then(function(r){return r.json()}).then(function(d){
  info=d;
  $("railfoot").innerHTML=
    '<span class="dotlive on"></span><b>'+esc(d.run)+'</b><br>'+
    d.classes.length+' signs · '+d.n_frames+'f / '+d.img_size+'px<br>'+
    d.params_m.toFixed(2)+' M params · '+d.device;
  $("liveSub").textContent=d.classes.length+" signs · "+d.device.toUpperCase();
  $("modelTbl").innerHTML=
    row("Run",d.run)+row("Classes",d.classes.length)+row("Frames",d.n_frames)+
    row("Crop size",d.img_size+" px")+row("Parameters",d.params_m.toFixed(2)+" M")+
    row("Device",d.device);
  var t=$("tier");
  t.innerHTML=(d.tiers||[]).map(function(x){
    return '<option'+(x===d.run?' selected':'')+'>'+esc(x)+'</option>'}).join("")
    ||'<option>'+esc(d.run)+'</option>';
  buildVocab(); buildEnrol(); updateTau();
}).catch(function(){ $("railfoot").innerHTML='<span class="dotlive"></span>server unreachable' });

function row(k,v){return '<tr><td>'+esc(k)+'</td><td class="n">'+esc(String(v))+'</td></tr>'}
function esc(s){var d=document.createElement("div");d.textContent=s;return d.innerHTML}
function clean(w){return String(w).replace(/^\d+\.\s*/,"")}
function now(){return new Date().toLocaleTimeString([],{hour:"2-digit",minute:"2-digit",second:"2-digit"})}

/* ---------- nav ---------- */
$("nav").addEventListener("click",function(e){
  var b=e.target.closest("button[data-s]"); if(!b) return; show(b.dataset.s);
});
function show(name){
  document.querySelectorAll("nav button").forEach(function(b){b.classList.toggle("on",b.dataset.s===name)});
  document.querySelectorAll(".screen").forEach(function(s){s.classList.toggle("on",s.id==="s-"+name)});
}

/* ---------- settings ---------- */
$("tau").addEventListener("input",function(){ tau=this.value/100; updateTau(); if(hist.length) renderLast(); });
function updateTau(){
  $("tauVal").textContent="τ = "+tau.toFixed(2);
  var msg = tau<=0.25 ? "Very permissive — the model will answer almost everything, including clips it has little basis for."
          : tau<=0.45 ? "<b>Measured operating point.</b> At τ = 0.40 the model answers about 75 % of held-out clips at 90.7 % accuracy."
          : tau<=0.65 ? "Cautious. Around 64 % coverage at roughly 93 % accuracy on the held-out set."
          : "Very cautious — high accuracy on the few clips it commits to, many more “not sure” responses.";
  $("tauNote").innerHTML=msg;
}
document.querySelectorAll("input[name=tta]").forEach(function(r){
  r.addEventListener("change",function(){ ttaViews=+this.value })});
$("detFloor").addEventListener("input",function(){
  detFloor=this.value/100; $("detVal").textContent="warn below "+this.value+" %";
  if(hist.length) renderLast();
});

/* ---------- capture ---------- */
function stage(i){ $("stages").classList.add("on");
  document.querySelectorAll("#stages .stage").forEach(function(el,k){
    el.classList.toggle("done",k<i); el.classList.toggle("now",k===i) }); }
function stagesOff(){ $("stages").classList.remove("on");
  document.querySelectorAll("#stages .stage").forEach(function(el){el.classList.remove("done","now")}); }

$("btnFile").addEventListener("click",function(){$("file").click()});
$("drop").addEventListener("click",function(){$("file").click()});
$("drop").addEventListener("keydown",function(e){if(e.key==="Enter"||e.key===" "){e.preventDefault();$("file").click()}});
["dragenter","dragover"].forEach(function(ev){$("drop").addEventListener(ev,function(e){e.preventDefault();this.classList.add("hot")})});
["dragleave","drop"].forEach(function(ev){$("drop").addEventListener(ev,function(e){e.preventDefault();this.classList.remove("hot")})});
$("drop").addEventListener("drop",function(e){var f=e.dataTransfer.files[0]; if(f){localPreview(f);send(f,f.name)}});
$("file").addEventListener("change",function(){var f=this.files[0]; if(f){localPreview(f);send(f,f.name)}});

function localPreview(f){var v=$("preview"); v.srcObject=null; v.src=URL.createObjectURL(f); v.loop=true; v.play().catch(function(){})}

$("btnCam").addEventListener("click",function(){
  navigator.mediaDevices.getUserMedia({video:{width:1280,height:720},audio:false}).then(function(s){
    camStream=s; var v=$("preview"); v.src=""; v.srcObject=s; v.play();
    $("btnRec").disabled=false; this.textContent="Camera on"; this.disabled=true;
  }.bind(this)).catch(function(e){alert("Camera unavailable: "+e.message)});
});
$("btnRec").addEventListener("click",function(){ record(camStream,$("btnRec"),function(blob){ send(blob,"clip.webm") }) });

function record(stream,btn,done){
  if(!stream) return;
  var chunks=[], rec=new MediaRecorder(stream,{mimeType:"video/webm"});
  rec.ondataavailable=function(e){if(e.data.size)chunks.push(e.data)};
  rec.onstop=function(){ done(new Blob(chunks,{type:"video/webm"})) };
  rec.start();
  var left=3, label=btn.textContent; btn.disabled=true; btn.textContent="● Recording 3…";
  var t=setInterval(function(){ left--; btn.textContent="● Recording "+left+"…";
    if(left<=0){clearInterval(t); rec.stop(); btn.disabled=false; btn.textContent=label} },1000);
}

function send(blob,name){
  stage(0);
  setTimeout(function(){stage(1)},250); setTimeout(function(){stage(2)},900);
  fetch("/predict",{method:"POST",body:blob,
    headers:{"X-Filename":name,"X-TTA":ttaViews===6?"1":"0"}})
  .then(function(r){ stage(3); return r.json().then(function(j){return{ok:r.ok,j:j}}) })
  .then(function(res){
    stagesOff();
    if(!res.ok){
      $("result").innerHTML='<div class="abstain"><div class="q">!</div>'+
        '<div class="t">Could not read that clip</div><div class="s">'+esc(res.j.error||"unknown")+'</div></div>';
      $("warn").innerHTML=""; return;
    }
    var d=res.j; d.when=now(); hist.push(d);
    renderLast(); renderHealth(d); renderDiag(); renderHistory();
  })
  .catch(function(e){ stagesOff(); alert("Request failed: "+e.message) });
}

/* ---------- rendering ---------- */
function renderLast(){
  var d=hist[hist.length-1]; if(!d) return;
  var top=d.candidates[0], answered=d.confidence>=tau;
  if(answered){
    $("result").innerHTML='<span class="conf">'+(top.p*100).toFixed(1)+'%</span>'+
      '<div class="word">'+esc(clean(top.word))+'</div>'+
      '<div class="bar"><i style="width:'+(top.p*100).toFixed(1)+'%"></i></div>'+
      d.candidates.slice(1).map(function(c){
        return '<div class="alt"><div>'+esc(clean(c.word))+
          '<div class="ab"><i style="width:'+(c.p*100).toFixed(1)+'%"></i></div></div>'+
          '<div class="pv">'+(c.p*100).toFixed(1)+'%</div></div>'}).join("")+
      '<div class="row" style="margin-top:14px"><button class="b primary" id="accept">Accept → sentence</button>'+
      '<button class="b" id="wrong">Wrong — correct it</button></div>';
    $("accept").onclick=function(){ addWord(clean(top.word)); };
    $("wrong").onclick=function(){ pending=d; show("notsure"); renderNotSure(); };
    $("nsBody").innerHTML='<div class="empty">Nothing pending.</div>';
  }else{
    $("result").innerHTML='<div class="abstain"><div class="q">?</div>'+
      '<div class="t">Didn\'t catch that</div><div class="s">best guess '+esc(clean(top.word))+
      ' at '+(top.p*100).toFixed(1)+'%, below τ = '+tau.toFixed(2)+'</div>'+
      '<div class="pickrow"><button class="b primary" id="goNs">Show shortlist →</button></div></div>';
    $("goNs").onclick=function(){ pending=d; show("notsure"); renderNotSure(); };
    pending=d; renderNotSure();
  }
  $("warn").innerHTML = d.detection_rate<detFloor
    ? '<div class="note neg"><b>Hands were rarely found</b> ('+(d.detection_rate*100).toFixed(0)+
      '% of frames). This prediction is about background, not a sign.</div>'
    : (d.detection_rate<0.6
      ? '<div class="note warn">Detection was patchy ('+(d.detection_rate*100).toFixed(0)+'%). Treat with caution.</div>':"");
}

function bars(p){
  return '<div class="hbar">'+
    '<i class="d" style="width:'+(p.direct*100)+'%"></i>'+
    '<i class="r" style="width:'+(p.roi*100)+'%"></i>'+
    '<i class="p" style="width:'+(p.interp*100)+'%"></i>'+
    '<i class="m" style="width:'+(p.missing*100)+'%"></i></div>';
}
function renderHealth(d){
  var names={left_hand:"L hand",right_hand:"R hand",face:"Face"};
  $("health").innerHTML=Object.keys(names).map(function(k){
    var p=d.provenance[k];
    return '<div class="hrow"><span class="nm">'+names[k]+'</span>'+bars(p)+
      '<span class="pc">'+(p.genuine*100).toFixed(0)+'%</span></div>'}).join("");
}
function renderDiag(){
  if(!hist.length) return;
  var names={left_hand:"L hand",right_hand:"R hand",face:"Face"}, agg={};
  Object.keys(names).forEach(function(k){
    agg[k]={direct:0,roi:0,interp:0,missing:0,genuine:0};
    hist.forEach(function(d){ ["direct","roi","interp","missing","genuine"].forEach(function(f){
      agg[k][f]+=d.provenance[k][f]/hist.length }) });
  });
  $("diagHealth").innerHTML=Object.keys(names).map(function(k){
    return '<div class="hrow"><span class="nm">'+names[k]+'</span>'+bars(agg[k])+
      '<span class="pc">'+(agg[k].genuine*100).toFixed(0)+'%</span></div>'}).join("");
  var worst=Math.min(agg.left_hand.genuine,agg.right_hand.genuine);
  $("diagNote").className="note"+(worst<detFloor?" neg":(worst<0.6?" warn":" pos"));
  $("diagNote").innerHTML= worst<detFloor
    ? "<b>Genuine detection is below the floor.</b> Results in this session are not trustworthy."
    : worst<0.6 ? "Detection is patchy on at least one hand stream. Expect weaker predictions."
    : "<b>Detection is healthy</b> across "+hist.length+" clip"+(hist.length>1?"s":"")+
      " — comparable to the ~91 % / ~85 % measured on INCLUDE.";
  $("diagTbl").innerHTML=hist.slice().reverse().map(function(d,i){
    var a=d.confidence>=tau;
    return '<tr'+(a?'':' class="ab"')+'><td class="n">'+(hist.length-i)+'</td>'+
      '<td class="n">'+(d.provenance.left_hand.genuine*100).toFixed(0)+'%</td>'+
      '<td class="n">'+(d.provenance.right_hand.genuine*100).toFixed(0)+'%</td>'+
      '<td class="n">'+(d.provenance.face.genuine*100).toFixed(0)+'%</td>'+
      '<td class="n">'+(a?esc(clean(d.candidates[0].word)):"not sure")+'</td></tr>'}).join("");
}

function renderNotSure(){
  var d=pending; if(!d){ $("nsBody").innerHTML='<div class="empty">Nothing pending.</div>'; return }
  var answered=d.confidence>=tau;
  $("nsBody").innerHTML=
    (answered?'<div class="note warn">This clip was answered at '+(d.confidence*100).toFixed(0)+
      '%. Pick the right word below to log a correction.</div>'
     :'<div class="abstain" style="padding-top:4px"><div class="q">?</div><div class="t">Didn\'t catch that</div>'+
      '<div class="s">top score '+(d.confidence*100).toFixed(1)+'% — below τ = '+tau.toFixed(2)+'</div></div>')+
    '<h3 style="margin-top:14px">Top-5 shortlist</h3>'+
    d.candidates.map(function(c,i){
      return '<div class="alt"><div>'+esc(clean(c.word))+
        '<div class="ab"><i style="width:'+(c.p*100).toFixed(1)+'%"></i></div></div>'+
        '<div class="pv">'+(c.p*100).toFixed(1)+'%</div></div>'}).join("")+
    '<div class="pickrow" style="justify-content:flex-start">'+
      d.candidates.slice(0,2).map(function(c,i){
        return '<button class="b primary" data-pick="'+i+'">Pick '+(i+1)+': '+esc(clean(c.word))+'</button>'}).join("")+
      '<button class="b" id="again">Sign again</button></div>';
  $("nsBody").querySelectorAll("[data-pick]").forEach(function(b){
    b.onclick=function(){ var c=d.candidates[+b.dataset.pick]; correct(d,c.word); addWord(clean(c.word)); };
  });
  $("again").onclick=function(){ pending=null; show("live"); };
}

function correct(d,word){
  corrections++; $("nCorr").textContent=corrections;
  fetch("/correct",{method:"POST",body:JSON.stringify({
      chosen:word, predicted:d.candidates[0].word, confidence:d.confidence,
      detection_rate:d.detection_rate, candidates:d.candidates, run:info&&info.run})})
    .then(function(r){return r.json()})
    .then(function(j){ if(j.logged) $("nCorrTot").textContent="✓ "+j.logged })
    .catch(function(){ $("nCorrTot").textContent="(not saved)" });
}

function addWord(w){ strip.push({w:w,t:now()}); pending=null; renderStrip(); show("live"); }
function renderStrip(){
  var seq=dedup();
  $("strip").innerHTML = strip.length
    ? strip.map(function(x){return '<span class="chip">'+esc(x.w)+'</span>'}).join("")
    : '<span class="ph">words you accept land here</span>';
  $("seq").innerHTML = seq.length
    ? seq.map(function(x){return '<span class="chip">'+esc(x.w)+'</span>'}).join("")
    : '<span class="ph">nothing composed yet</span>';
}
function dedup(){
  if(!$("dedup").checked) return strip;
  return strip.filter(function(x,i){ return i===0 || x.w!==strip[i-1].w });
}
$("dedup").addEventListener("change",renderStrip);
$("stripUndo").addEventListener("click",function(){strip.pop();renderStrip()});
$("stripClear").addEventListener("click",function(){strip=[];renderStrip()});
$("stripGo").addEventListener("click",function(){show("compose")});
$("seqCopy").addEventListener("click",function(){
  navigator.clipboard.writeText(dedup().map(function(x){return x.w}).join(" "))
    .then(function(){ $("seqCopy").textContent="Copied"; setTimeout(function(){$("seqCopy").textContent="Copy"},1200) })
    .catch(function(){ alert("Clipboard unavailable") });
});
$("seqSpeak").addEventListener("click",function(){
  var text=dedup().map(function(x){return x.w}).join(" ");
  if(!text) return;
  if(!("speechSynthesis" in window)){ alert("Speech synthesis unavailable in this browser"); return }
  speechSynthesis.cancel(); speechSynthesis.speak(new SpeechSynthesisUtterance(text));
});
$("seqExport").addEventListener("click",function(){
  var payload={run:info&&info.run, tau:tau, exported:new Date().toISOString(),
               sequence:dedup(), history:hist.map(function(d){return{
                 word:d.candidates[0].word, confidence:d.confidence,
                 detection_rate:d.detection_rate, when:d.when}})};
  var a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:"application/json"}));
  a.download="isl-session-"+Date.now()+".json"; a.click(); URL.revokeObjectURL(a.href);
});

function renderHistory(){
  $("hist").innerHTML = hist.length ? hist.slice().reverse().map(function(d){
    var a=d.confidence>=tau;
    return '<tr'+(a?'':' class="ab"')+'><td>'+(a?esc(clean(d.candidates[0].word)):"(not sure)")+
      '</td><td class="n">'+(d.confidence*100).toFixed(0)+'%</td><td class="n">'+
      (d.detection_rate*100).toFixed(0)+'%</td><td class="n">'+d.when+'</td></tr>'}).join("")
    : '<tr><td colspan="4" class="empty">Nothing yet.</td></tr>';
}

/* ---------- vocabulary ---------- */
function buildVocab(){
  var pc=info.per_class;
  $("vocabSub").textContent=info.classes.length+" signs"+(pc?" · per-word recall from the held-out set":"");
  $("vNote").innerHTML = pc
    ? "<b>Recall is per word on the 472 held-out clips</b>, session-disjoint. A green word is one the model actually gets right; red words it essentially cannot do. Two words sit at 0 % — both are short single motions with little handshape signature."
    : "<b>Per-word recall not available.</b> Generate it with <code>python -m islvit.tta --run "+esc(info.run)+" --dump-per-class</code>.";
  renderVocab();
  $("vFilter").addEventListener("input",renderVocab);
  $("vWeak").addEventListener("change",renderVocab);
}
function renderVocab(){
  var pc=info.per_class, q=$("vFilter").value.toLowerCase(), weakOnly=$("vWeak").checked;
  var items=(pc||info.classes.map(function(c){return{word:c,recall:null,support:null}}))
    .filter(function(c){ return clean(c.word).toLowerCase().indexOf(q)>=0 })
    .filter(function(c){ return !weakOnly || (c.recall!==null && c.recall<0.5) })
    .sort(function(a,b){ return (a.recall===null?-1:a.recall)-(b.recall===null?-1:b.recall) });
  $("vCount").textContent=items.length+" shown";
  $("vgrid").innerHTML=items.map(function(c){
    var r=c.recall, cls=r===null?"":(r>=0.8?"g":(r>=0.5?"y":"r"));
    return '<div class="vc"><div class="w">'+esc(clean(c.word))+'</div>'+
      '<div class="rb"><i class="'+cls+'" style="width:'+(r===null?0:r*100).toFixed(0)+'%"></i></div>'+
      '<div class="mt"><span>'+(r===null?"no data":(r*100).toFixed(0)+"% recall")+'</span>'+
      '<span>'+(c.support===null?"":c.support+" clips")+'</span></div></div>'}).join("")
    || '<div class="empty">No words match.</div>';
}

/* ---------- enrolment ---------- */
function buildEnrol(){
  $("egrid").innerHTML=info.classes.map(function(c){
    return '<div class="ec" data-w="'+esc(c)+'"><div class="w">'+esc(clean(c))+'</div>'+
      '<div class="pips"><i></i><i></i><i></i></div></div>'}).join("");
  $("egrid").addEventListener("click",function(e){
    var c=e.target.closest(".ec"); if(!c) return;
    enrolWord=c.dataset.w;
    $("egrid").querySelectorAll(".ec").forEach(function(x){x.classList.toggle("on",x===c)});
    $("eTarget").innerHTML='recording <b>'+esc(clean(enrolWord))+'</b>';
    $("eRec").disabled=!eStream;
  });
  updateEnrolProgress();
}
function updateEnrolProgress(){
  var done=0;
  Object.keys(enrolCounts).forEach(function(w){ done+=Math.min(3,enrolCounts[w]) });
  $("eProg").textContent=done+" / "+(info.classes.length*3);
  $("egrid").querySelectorAll(".ec").forEach(function(c){
    var n=enrolCounts[c.dataset.w]||0;
    c.querySelectorAll(".pips i").forEach(function(p,i){ p.classList.toggle("f",i<n) });
  });
}
$("eCam").addEventListener("click",function(){
  navigator.mediaDevices.getUserMedia({video:{width:1280,height:720},audio:false}).then(function(s){
    eStream=s; var v=$("ePreview"); v.srcObject=s; v.play();
    this.textContent="Camera on"; this.disabled=true; $("eRec").disabled=!enrolWord;
  }.bind(this)).catch(function(e){alert("Camera unavailable: "+e.message)});
});
$("eRec").addEventListener("click",function(){
  var signer=$("signer").value.trim();
  if(!signer){ $("eStatus").innerHTML='<div class="note neg">Enter a signer id first.</div>'; return }
  if(!enrolWord){ $("eStatus").innerHTML='<div class="note neg">Pick a word to record.</div>'; return }
  record(eStream,$("eRec"),function(blob){
    fetch("/enrol",{method:"POST",body:blob,
      headers:{"X-Filename":"take.webm","X-Signer":signer,"X-Word":enrolWord}})
    .then(function(r){return r.json().then(function(j){return{ok:r.ok,j:j}})})
    .then(function(res){
      if(!res.ok){ $("eStatus").innerHTML='<div class="note neg">'+esc(res.j.error)+'</div>'; return }
      enrolCounts[enrolWord]=res.j.takes_for_word; updateEnrolProgress();
      $("eStatus").innerHTML='<div class="note pos">Saved <code>'+esc(res.j.saved)+
        '</code> — take '+res.j.take+' of 3 for this word.</div>';
    }).catch(function(e){ $("eStatus").innerHTML='<div class="note neg">'+esc(e.message)+'</div>' });
  });
});
})();
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Local web UI for ISL-ViT-Tiny")
    parser.add_argument("--run", type=str, required=True, help="a runs/<tag> directory")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--crop-size", type=int, default=128, help="must match the training cache")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="CPU by default: a second model on a GPU that is mid-training has "
        "killed a multi-hour run in this project before",
    )
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("  cuda requested but unavailable; falling back to cpu")
        device = "cpu"

    run_dir = Path(args.run)
    model, config, classes = load_run(run_dir, device)
    model.eval()
    STATE.update(
        model=model, config=config, classes=classes, device=device, crop_size=args.crop_size,
        info={
            "run": run_dir.name,
            "classes": classes,
            "n_frames": config["n_frames"],
            "img_size": config["img_size"],
            "params_m": count_parameters(model)["total_M"],
            "device": device,
            # Per-word recall from the held-out set, if it has been dumped. The
            # vocabulary browser shows it so a user can see which signs are
            # actually reliable instead of trusting the mean.
            "per_class": (
                json.loads((run_dir / "per_class.json").read_text(encoding="utf-8"))["classes"]
                if (run_dir / "per_class.json").exists() else None
            ),
            "tiers": sorted(
                p.parent.name for p in Path("runs").glob("*/summary.json")
                if p.parent.name.startswith(("vocab", "f16_", "f32_", "f24_"))
            ),
        },
    )

    print(f"\n  ISL Reader — {run_dir.name}, {len(classes)} signs, "
          f"{config['n_frames']}f/{config['img_size']}px on {device}")
    print(f"  open  http://127.0.0.1:{args.port}/\n")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
