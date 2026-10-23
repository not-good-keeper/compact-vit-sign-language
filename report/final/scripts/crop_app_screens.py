"""Crop the raw application screenshots into the figures used in the report.

Raw screenshots (report/final/data/app_screens_raw/, made by capture_app_screens.js) are cut to the region that shows
one feature, and a few are assembled from regions of the same screenshot. Nothing is redrawn or recoloured. The crop
boxes and SHA-256 of every output are written into report/final/data/app_screens_log.json under "figures".

    python report/final/scripts/crop_app_screens.py [OUT_DIR]
"""
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "report" / "final" / "data" / "app_screens_raw"
LOG = ROOT / "report" / "final" / "data" / "app_screens_log.json"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "report" / "final" / "latex" / "figures"
PAD = (8, 10, 18)

# name -> (raw file, box) for single crops
SINGLE = {
    "appfig_idle": ("app_01_capture_idle", (0, 0, 1440, 560)),
    "appfig_reading": ("app_02_capture_reading", (285, 105, 1420, 553)),
    "appfig_tray": ("app_05_capture_unsure", (285, 605, 1415, 1108)),
    "appfig_notsure": ("app_06_notsure", (285, 105, 1415, 625)),
    "appfig_sentence": ("app_07_sentence", (285, 105, 1415, 520)),
    "appfig_shortcuts": ("app_12_shortcuts", (455, 275, 985, 665)),
    "appfig_settings": ("app_08_settings", (285, 105, 1405, 1205)),
    "appfig_detection": ("app_09_detection", (285, 105, 1415, 1125)),
    "appfig_vocab": ("app_10_vocabulary", (285, 105, 1415, 645)),
    "appfig_enrol": ("app_11_enrolment", (285, 105, 1415, 990)),
}
# confident answer: stage and ring on top, candidates and crop health beneath
STACKED = {
    "appfig_confident": ("app_03_capture_confident", (285, 105, 1420, 553), (285, 1122, 1405, 1388)),
}
# case strips: lens (with any warning) | candidates over crop health
STRIPS = {
    "appfig_case_dress": ("app_04_capture_lower_confidence", (1050, 105, 1420, 600), (285, 1122, 785, 1388), (800, 1122, 1405, 1388)),
    "appfig_case_wide": ("app_05_capture_unsure", (1050, 105, 1420, 600), (285, 1122, 785, 1388), (800, 1122, 1405, 1388)),
    "appfig_case_180_bicubic": ("app_13_capture_180p_bicubic", (1050, 105, 1420, 640), (285, 1125, 785, 1390), (800, 1125, 1405, 1390)),
    "appfig_case_180_srgan": ("app_14_capture_180p_srgan_v2", (1050, 105, 1420, 640), (285, 1122, 785, 1388), (800, 1122, 1405, 1388)),
    "appfig_case_cislr": ("app_15_capture_cislr_low_detection", (1050, 105, 1420, 640), (285, 1165, 785, 1431), (800, 1165, 1405, 1431)),
}


def load(name):
    return Image.open(RAW / f"{name}.png").convert("RGB")


def stack(images, gap=10, width=None):
    width = width or max(i.width for i in images)
    height = sum(i.height for i in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), PAD)
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height + gap
    return canvas


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = json.loads(LOG.read_text(encoding="utf-8"))
    figures = {}
    for name, (raw, box) in SINGLE.items():
        image = load(raw).crop(box)
        image.save(OUT / f"{name}.png")
        figures[name] = {"raw": raw + ".png", "boxes": [box], "size": image.size}
    for name, (raw, top, bottom) in STACKED.items():
        src = load(raw)
        image = stack([src.crop(top), src.crop(bottom)], width=max(top[2] - top[0], bottom[2] - bottom[0]))
        image.save(OUT / f"{name}.png")
        figures[name] = {"raw": raw + ".png", "boxes": [top, bottom], "size": image.size}
    for name, (raw, lens, cand, health) in STRIPS.items():
        src = load(raw)
        lens_i, cand_i, health_i = src.crop(lens), src.crop(cand), src.crop(health)
        right = stack([cand_i, health_i], gap=10, width=health_i.width)
        canvas = Image.new("RGB", (lens_i.width + 10 + right.width, max(lens_i.height, right.height)), PAD)
        canvas.paste(lens_i, (0, 0))
        canvas.paste(right, (lens_i.width + 10, 0))
        canvas.save(OUT / f"{name}.png")
        figures[name] = {"raw": raw + ".png", "boxes": [lens, cand, health], "size": canvas.size}
    for name, entry in figures.items():
        entry["sha256"] = sha(OUT / f"{name}.png")
        entry["raw_sha256"] = sha(RAW / entry["raw"])
        print(name, entry["size"])
    log["figures"] = figures
    LOG.write_text(json.dumps(log, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
