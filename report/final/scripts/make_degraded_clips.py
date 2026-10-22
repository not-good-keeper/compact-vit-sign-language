"""Write 180p / JPEG-40 copies of the 50 app-test clips for the super-resolution app test.

Each frame of the 1080p source goes through islvit.sr.degrade.degrade_frame, the
same deterministic degradation the GAN study applies inside the frame reader,
and is stored losslessly (FFV1 in Matroska, bgr0), so the server decodes exactly
the degraded frames and no second codec is added. Output folder is local only.

Usage (repository root):
    python report/final/scripts/make_degraded_clips.py OUT_DIR
"""
from __future__ import annotations

import sys
from pathlib import Path

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app_integration_test import clip_list  # noqa: E402
from islvit.sr.degrade import degrade_frame  # noqa: E402

HEIGHT, QUALITY = 180, 40


def main() -> None:
    out_root = Path(sys.argv[1])
    for row in clip_list():
        src = ROOT / "INCLUDE_raw" / row["video_path"]
        dst = out_root / Path(row["video_path"]).with_suffix(".mkv")
        dst.parent.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(degrade_frame(frame, HEIGHT, QUALITY))
        cap.release()
        h, w = frames[0].shape[:2]
        with av.open(str(dst), "w") as container:
            stream = container.add_stream("ffv1", rate=int(round(fps)))
            stream.width, stream.height, stream.pix_fmt = w, h, "bgr0"
            for f in frames:
                vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="bgr24")
                for packet in stream.encode(vf):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        print(dst, len(frames), (w, h))


if __name__ == "__main__":
    main()
