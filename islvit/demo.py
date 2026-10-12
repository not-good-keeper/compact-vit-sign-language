"""Run the ISL Reader app -- the new UI with real predictions -- for a demo on another computer.

The interface lives in islvit/ui (islvit/ui/build.py bundles it into one self-contained
index.html) and the backend is islvit/serve.py, unchanged: this launcher only swaps
serve's built-in page for the new one and picks the address the server listens on.
Every request (/info, /predict, /correct, /enrol) is served by serve.py's own handler.

Reaching it from your own computer
----------------------------------
SSH tunnel (default; nothing is exposed to the network). The server listens on
127.0.0.1 of this machine, and the tunnel carries it to yours::

    # on this machine
    python -m islvit.demo
    # on your computer -- the same SSH login you already use, plus -L
    ssh -N -L 7860:127.0.0.1:7860 <user>@<this machine>
    # then open http://localhost:7860 in your browser

Tailscale (both computers on the same tailnet). Bind to this machine's tailnet
address only, so the app is reachable from your devices and not from the LAN::

    python -m islvit.demo --host tailscale
    # open http://<tailnet-ip>:7860 -- Windows may ask once to allow Python through the firewall

The model runs on the CPU by default and CUDA is hidden from the process: the GPU on
this machine is often mid-training, and a second model there has killed long runs before.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UI = ROOT / "islvit" / "ui"
DEFAULT_RUN = ROOT / "release" / "isl_vit_tiny_lm_int4_s0.pt"


def build_page() -> str:
    """The bundled page, rebuilt first if any UI source is newer than it."""
    page = UI / "index.html"
    sources = [p for p in (UI / "src").rglob("*") if p.is_file()] + [UI / "build.py"]
    if not page.exists() or max(p.stat().st_mtime for p in sources) > page.stat().st_mtime:
        print("  UI sources changed; rebuilding islvit/ui/index.html")
        runpy.run_path(str(UI / "build.py"), run_name="__main__")
    return page.read_text(encoding="utf-8")


SAMPLES = ROOT / "demo_samples"


def load_samples() -> dict:
    """The demo clip manifest (islvit.demo_samples), with the set of files it may serve."""
    path = SAMPLES / "manifest.json"
    if not path.exists():
        return {"samples": [], "files": set()}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["samples"] = [s for s in manifest["samples"] if (SAMPLES / s["file"]).exists()]
    files = {s["file"] for s in manifest["samples"]} | {s["poster"] for s in manifest["samples"] if (SAMPLES / s["poster"]).exists()}
    return {**manifest, "files": files}


def restrict_to_deployed(serve) -> None:
    """Answer from the 50 deployed words, the way the evaluation scores the model.

    islvit.mask50 averages the six views' probabilities over all 262 trained words and
    then sets every non-deployed word to zero before taking the top word. serve.classify
    has no such step, so the live app answered from all 262 words (about 72 % on the
    held-out clips instead of 85.9 %). This replaces it with the same computation; the
    masked vector is renormalised for display, which does not change which word is top.
    """
    import torch
    from islvit.mask50 import deployed_columns

    def classify(video_path, use_tta: bool = True) -> dict:
        model, config, classes = serve.STATE["model"], serve.STATE["config"], serve.STATE["classes"]
        if "keep" not in serve.STATE:
            keep = torch.zeros(len(classes), dtype=torch.bool)
            keep[torch.as_tensor(deployed_columns(classes))] = True
            serve.STATE["keep"] = keep
        keep = serve.STATE["keep"]
        clip, detected, geometry, sources = serve.extract(video_path, serve.STATE["crop_size"])
        landmarks = serve.extract_landmarks(video_path, interp=config.get("lm_interp", False)) \
            if config.get("landmarks") else None
        hands = sources[:, :2]
        genuine = float(((hands == 1) | (hands == 2)).mean())
        names = ("missing", "direct", "roi", "interp")
        provenance = {}
        for index, stream in enumerate(("left_hand", "right_hand", "face")):
            counts = np.bincount(sources[:, index].astype(int), minlength=4)
            share = counts / max(1, counts.sum())
            provenance[stream] = {name: float(value) for name, value in zip(names, share)}
            provenance[stream]["genuine"] = float(share[1] + share[2])
        total, n = None, 0
        with torch.no_grad():
            for crops, det, geo, extras in serve.views(clip, detected, geometry, config["n_frames"],
                                                       config["img_size"], use_tta, landmarks):
                extras = {key: value.to(serve.STATE["device"]) for key, value in extras.items()}
                logits = model(crops.to(serve.STATE["device"]), det.to(serve.STATE["device"]),
                               geo.to(serve.STATE["device"]), **extras)
                probs = logits.float().softmax(1)
                total = probs if total is None else total + probs
                n += 1
        probability = (total / n).squeeze(0).cpu() * keep           # mask50's step
        probability = probability / probability.sum()
        top = probability.topk(5)
        return {"frames": int(clip.shape[0]), "views": n, "detection_rate": genuine, "provenance": provenance,
                "confidence": float(probability.max()), "deployed_only": True,
                "candidates": [{"word": classes[i], "p": float(p)} for p, i in zip(top.values.tolist(), top.indices.tolist())]}

    serve.classify = classify
    print("  answering from the 50 deployed words (as evaluated); --all-words for all 262")


def tailscale_ip() -> str:
    try:
        out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError) as error:
        raise SystemExit(f"--host tailscale: could not ask Tailscale for this machine's address ({error})")
    ip = out.stdout.split()[0] if out.stdout.split() else ""
    if not ip.startswith("100."):
        raise SystemExit(f"--host tailscale: unexpected address {ip!r}; is Tailscale connected?")
    return ip


def main() -> None:
    parser = argparse.ArgumentParser(description="ISL Reader demo: the new UI with real predictions")
    parser.add_argument("--run", default=str(DEFAULT_RUN), help="release .pt or runs/<tag> directory")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1",
                        help="127.0.0.1 for an SSH tunnel (default), 'tailscale' for this machine's tailnet "
                             "address, or an explicit address")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--all-words", action="store_true",
                        help="answer from all trained words (262) instead of the 50 deployed ones; the 50-word "
                             "restriction is what the reported 85.9 %% measures, and is the default")
    parser.add_argument("--sr", default=None,
                        help="SR-GAN generator (e.g. runs/sr_gan/generator.pt) used in place of bicubic upscaling "
                             "for video below 720p, as in the GAN study; off by default")
    args = parser.parse_args()

    host = tailscale_ip() if args.host == "tailscale" else args.host
    if args.device == "cpu":
        import torch
        torch.cuda.is_available = lambda: False   # keep every stage (MediaPipe crops, upscaler) off the GPU

    sys.path.insert(0, str(ROOT))
    from islvit import serve

    serve.PAGE = build_page()
    if not args.all_words:
        restrict_to_deployed(serve)
    if args.sr:
        # serve.classify calls predict.extract / extract_landmarks without an upscaler; bind one, so
        # sub-720p clips go through the GAN front end exactly as in the comparative study.
        import functools
        serve.extract = functools.partial(serve.extract, upscaler=args.sr)
        serve.extract_landmarks = functools.partial(serve.extract_landmarks, upscaler=args.sr)
        print(f"  SR-GAN front end: {args.sr} (applies to video below 720p)")
    samples = load_samples()
    print(f"  {len(samples['samples'])} demo clips from {SAMPLES.relative_to(ROOT)}/" if samples["samples"]
          else "  no demo clips (python -m islvit.demo_samples builds them)")

    class DemoHandler(serve.Handler):
        """serve.py's handler plus two read-only routes for the demo clip strip:
        GET /samples (the manifest) and GET /samples/<file> (a clip or its poster).
        Only files named in the manifest are served, so nothing else on disk is reachable."""

        def do_GET(self):
            if self.path == "/samples":
                self._json(200, {k: v for k, v in samples.items() if k != "files"})
            elif self.path.startswith("/samples/"):
                name = self.path[len("/samples/"):]
                if name not in samples["files"]:
                    self._json(404, {"error": "not found"})
                    return
                kind = "video/mp4" if name.endswith(".mp4") else "image/jpeg"
                self._send(200, (SAMPLES / name).read_bytes(), kind)
            else:
                super().do_GET()

    # serve.main() always binds 127.0.0.1 with its own handler; route both through the demo's.
    base = serve.ThreadingHTTPServer

    class Server(base):
        def __init__(self, address, handler):
            super().__init__((host, address[1]), DemoHandler)
            print(f"  listening on http://{host}:{address[1]}/")
            if host == "127.0.0.1":
                print(f"  from your computer:  ssh -N -L {address[1]}:127.0.0.1:{address[1]} <user>@<this machine>"
                      f"   then open http://localhost:{address[1]}")

    serve.ThreadingHTTPServer = Server
    sys.argv = ["islvit.serve", "--run", args.run, "--port", str(args.port), "--device", args.device]
    serve.main()


if __name__ == "__main__":
    main()
