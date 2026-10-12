"""CISLR: does a hand model run directly on the frame find hands the pose-led Holistic path misses?

CISLR signers are tightly framed with large hands, yet the Holistic path (pose first, then a
hand ROI around the pose wrist) finds a left hand in only ~25 % of frames. This counts hands
found per frame by Holistic and by a HandLandmarker run on the whole frame, on the same frames.
"""
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions

from islvit.data import crops

vision = mp.tasks.vision
rows = [r for r in csv.DictReader(open("splits/crosscorpus__cislr-test.csv", encoding="utf-8"))
        if r.get("corpus") == "cislr" and r["split"] == "test"]
vids = [Path("CISLR/CISLR_v1.5-a_videos") / r["video_path"] for r in rows[::15]][:40]
conf = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
direct = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(crops.HAND_MODEL)), running_mode=vision.RunningMode.IMAGE,
    num_hands=2, min_hand_detection_confidence=conf, min_hand_presence_confidence=conf))

tot = hol_any = hol_two = dir_any = dir_two = 0
for v in vids:
    cap = cv2.VideoCapture(str(v))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for idx in np.linspace(0, n - 1, 8).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        # native 300 px up to the 720 working height, letterboxed to 1280 wide like the pipeline
        canvas = np.zeros((720, 1280, 3), np.uint8)
        canvas[:, 280:1000] = cv2.resize(frame, (720, 720), interpolation=cv2.INTER_CUBIC)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        k = len(direct.detect(image).hand_landmarks)
        crops.reset_detectors()
        h = crops.get_holistic().detect(image)
        hk = int(bool(h.left_hand_landmarks)) + int(bool(h.right_hand_landmarks))
        dir_any += k > 0; dir_two += k >= 2; hol_any += hk > 0; hol_two += hk >= 2; tot += 1
print(f"{len(vids)} CISLR clips, {tot} frames, direct-detector confidence {conf}")
print(f"Holistic (pose-led):    any hand {hol_any / tot:.3f}   two hands {hol_two / tot:.3f}")
print(f"HandLandmarker direct:  any hand {dir_any / tot:.3f}   two hands {dir_two / tot:.3f}")
