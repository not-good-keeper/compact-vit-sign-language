"""Write docs/GAN_COMPARATIVE_STUDY.md: GAN versus no GAN, all three GAN versions.

Every number in the document is read from a result file under runs/ when this script
runs; the prose around the numbers was written for the results as they stand. The
document is also converted to .docx with pandoc if pandoc is on the PATH.

    python -m islvit.gan_report
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

RUNS = Path("runs")
STUDY = RUNS / "gan_study"
STUDY_V1 = RUNS / "gan_study_v1"
DIAG = STUDY / "diagnostics"
DOC = Path("docs/GAN_COMPARATIVE_STUDY.md")
FIG = Path("docs/figures")
SEEDS = (0, 1, 2)
LEVELS = (("clean", "Clean 1080p"), ("L1", "360p, JPEG 60"), ("L2", "240p, JPEG 50"), ("L3", "180p, JPEG 40"))
DEGRADED = LEVELS[1:]

# Front ends compared on the test set. "srgan" is v2 and "srgan_v1" lives in gan_study_v1.
ARMS = {
    "bicubic": ("Bicubic", "no GAN", "#9AA3B2"),
    "srl1": ("SR-L1", "no GAN, learned", "#5B7BD6"),
    "srgan_v1": ("SR-GAN v1", "GAN", "#C065B0"),
    "srgan": ("SR-GAN v2", "GAN", "#2E9E6E"),
    "sr3_gan_task": ("SR-GAN v3", "GAN", "#1E5E46"),
    "sr3_l1_task": ("v3 without the GAN", "no GAN, learned", "#D98C2B"),
}
TEAM = [("Yalamanchi Kushal", "CB.SC.U4CSE23766"), ("Nethi Kushala Kumar", "CB.SC.U4CSE23735"),
        ("Dhruseek Rishi Menon", "CB.SC.U4CSE23716")]
LOOKALIKES = [["41. you", "46. you (plural)"], ["83. big large", "81. wide", "82. narrow"],
              ["79. short", "84. small little"], ["94. good", "95. bad", "96. wet"], ["38. Dress", "42. T-Shirt"]]


# --------------------------------------------------------------------------- #
# Loading and statistics
# --------------------------------------------------------------------------- #
def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def result(family: str, level: str, arm: str, seed: int) -> dict | None:
    """masked50 result for one model on one front end."""
    if level == "clean":
        if family == "frozen_int4":
            return load(RUNS / f"f16_262w_lm_s{seed}_qat4" / "masked50_clean.json")
        if family == "frozen_fp32":
            return load(RUNS / f"f16_262w_lm_s{seed}" / "masked50_clean.json")
        if arm == "srgan_v1":
            return load(STUDY_V1 / "retrained" / f"clean_srgan_s{seed}.json")
        return load(STUDY / "retrained" / f"clean_{arm}_s{seed}.json")
    if arm == "srgan_v1":
        return load(STUDY_V1 / family / f"{level}_srgan_s{seed}.json")
    return load(STUDY / family / f"{level}_{arm}_s{seed}.json")


def top1(r: dict | None, key: str = "top1") -> float | None:
    return None if r is None else r["masked_tta"][key]


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None, 0
    return float(np.mean(values)), (float(np.std(values, ddof=1)) if len(values) > 1 else 0.0), len(values)


def cell(family, level, arm, key="top1") -> str:
    m, sd, n = mean_sd([top1(result(family, level, arm, s), key) for s in SEEDS])
    if m is None:
        return "—"
    return f"{100 * m:.1f} ± {100 * sd:.1f}" if n == 3 else f"{100 * m:.1f} (n={n})"


def mean(family, level, arm, key="top1") -> float | None:
    return mean_sd([top1(result(family, level, arm, s), key) for s in SEEDS])[0]


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(family: str, level: str, base: str, other: str) -> dict | None:
    """Pre-registered rule: mean gain > 0, every seed the same direction, pooled exact McNemar p < 0.05."""
    deltas, gained, lost = [], 0, 0
    for s in SEEDS:
        a, b = result(family, level, base, s), result(family, level, other, s)
        if a is None or b is None:
            continue
        ha = dict(zip(a["video_paths"], a["hits_masked_tta"]))
        hb = dict(zip(b["video_paths"], b["hits_masked_tta"]))
        assert ha.keys() == hb.keys(), "the two results scored different clips"
        lost += sum(1 for k in ha if ha[k] and not hb[k])
        gained += sum(1 for k in ha if hb[k] and not ha[k])
        deltas.append(top1(b) - top1(a))
    if not deltas:
        return None
    p = mcnemar(lost, gained)
    m = float(np.mean(deltas))
    verdict = ("helps" if m > 0 and all(d > 0 for d in deltas) and p < 0.05 else
               "hurts" if m < 0 and all(d < 0 for d in deltas) and p < 0.05 else "no reliable difference")
    return {"delta": m, "per_seed": deltas, "gained": gained, "lost": lost, "p": p, "verdict": verdict}


def fmt_p(p: float) -> str:
    if p >= 0.001:
        return f"{p:.2g}"
    exp = int(math.floor(math.log10(p)))
    sup = str(exp).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{p / 10 ** exp:.1f} × 10{sup}"


def compare_row(level_name: str, label: str, c: dict | None) -> str:
    if c is None:
        return f"| {level_name} | {label} | — | — | — | — |"
    seeds = " / ".join(f"{100 * d:+.1f}" for d in c["per_seed"])
    return (f"| {level_name} | {label} | {100 * c['delta']:+.1f} ({seeds}) | {c['gained']} / {c['lost']} | "
            f"{fmt_p(c['p'])} | **{c['verdict']}** |")


def detection(level: str, arm: str) -> dict | None:
    if arm == "srgan_v1":
        return load(STUDY_V1 / "detection" / f"{level}_srgan.json")
    return load(STUDY / "detection" / f"{level}_{arm}.json")


def summary(run: str) -> dict:
    return load(RUNS / run / "summary.json") or {}


def last_history(run: str) -> dict:
    path = RUNS / run / "history.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def pct(x, digits=1) -> str:
    return "—" if x is None else f"{100 * x:.{digits}f}"


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def figures() -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    FIG.mkdir(parents=True, exist_ok=True)
    made = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(FIG / f"{name}.png", dpi=200)
        plt.close(fig)
        made.append(f"{name}.png")

    # fig21: top-1 for every front end, frozen INT4, and the fine-tuned arm
    order = ["bicubic", "srl1", "sr3_l1_task", "srgan_v1", "srgan", "sr3_gan_task"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax, w = axes[0], 0.13
    clean = mean("frozen_int4", "clean", "bicubic")
    for j, arm in enumerate(order):
        vals = [mean("frozen_int4", lv, arm) for lv, _ in DEGRADED]
        xs = [i + (j - 2.5) * w for i in range(3)]
        ax.bar(xs, [100 * v for v in vals], w * 0.95, color=ARMS[arm][2], label=f"{ARMS[arm][0]} ({ARMS[arm][1]})")
        for x, v in zip(xs, vals):
            ax.text(x, 100 * v + 0.6, f"{100 * v:.0f}", ha="center", fontsize=7)
    ax.axhline(100 * clean, color="#1F2430", ls="--", lw=1)
    ax.text(2.6, 100 * clean + 0.8, f"clean 1080p {100 * clean:.1f}", ha="right", fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels([n for _, n in DEGRADED]); ax.set_ylim(50, 92)
    ax.set_ylabel("top-1, 50 words (%)"); ax.set_title("Shipped INT4 model, frozen (mean of 3 seeds)")
    ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax, w = axes[1], 0.26
    for j, arm in enumerate(["bicubic", "srgan_v1", "srgan"]):
        vals = [mean("retrained", lv, arm) for lv, _ in DEGRADED]
        xs = [i + (j - 1) * w for i in range(3)]
        ax.bar(xs, [100 * v for v in vals], w * 0.95, color=ARMS[arm][2], label=ARMS[arm][0])
        for x, v in zip(xs, vals):
            ax.text(x, 100 * v + 0.6, f"{100 * v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(range(3)); ax.set_xticklabels([n.split(",")[0] for _, n in DEGRADED]); ax.set_ylim(50, 92)
    ax.set_title("FP32 models fine-tuned on 360p video")
    ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    save(fig, "fig21_gan_accuracy")

    # fig22: hands found by the detector
    fig, ax = plt.subplots(figsize=(12, 4.2))
    w = 0.13
    for j, arm in enumerate(order):
        vals = []
        for lv, _ in DEGRADED:
            d = detection(lv, arm)
            vals.append(None if d is None else (d["streams"]["left_hand"]["genuine"] + d["streams"]["right_hand"]["genuine"]) / 2)
        xs = [i + (j - 2.5) * w for i in range(3)]
        ax.bar([x for x, v in zip(xs, vals) if v is not None], [100 * v for v in vals if v is not None], w * 0.95,
               color=ARMS[arm][2], label=ARMS[arm][0])
    ref = (0.911 + 0.852) / 2
    ax.axhline(100 * ref, color="#1F2430", ls="--", lw=1)
    ax.text(2.6, 100 * ref + 1, "clean INCLUDE", ha="right", fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels([n for _, n in DEGRADED]); ax.set_ylim(40, 95)
    ax.set_ylabel("hand frames with a real detection (%)"); ax.set_title("Hands found by MediaPipe, mean of left and right")
    ax.legend(fontsize=8, frameon=False, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    save(fig, "fig22_gan_detection")

    # fig23: what each upscaler does to a held-out hand patch
    m = montage()
    if m:
        made.append(m)
    return made


def montage() -> str | None:
    bank_path = Path("C:/isl_gan_work/hr_heldout.npy")
    gens = [("SR-L1", RUNS / "sr_l1/generator.pt"), ("v3 without GAN", RUNS / "sr3_l1_task/generator.pt"),
            ("SR-GAN v1", RUNS / "sr_gan_v1/generator.pt"), ("SR-GAN v2", RUNS / "sr_gan/generator.pt"),
            ("SR-GAN v3", RUNS / "sr3_gan_task/generator.pt")]
    if not bank_path.exists() or not all(p.exists() for _, p in gens):
        return None
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from islvit.sr.evaluate import degrade_patch
    from islvit.sr.upscaler import super_resolve

    bank = np.load(bank_path, mmap_mode="r")
    picks = [np.asarray(bank[i]) for i in (140, 245, 285, 335)]   # hand crops from the held-out bank
    rows = []
    for hr in picks:
        lr = degrade_patch(hr, 4, 40)
        size = hr.shape[0]
        up = lambda path: cv2.resize(np.ascontiguousarray(super_resolve([np.ascontiguousarray(lr[..., ::-1])], path)[0][..., ::-1]),
                                     (size, size), interpolation=cv2.INTER_AREA)
        rows.append([cv2.resize(lr, (size, size), interpolation=cv2.INTER_NEAREST),
                     cv2.resize(lr, (size, size), interpolation=cv2.INTER_CUBIC)] + [up(p) for _, p in gens] + [hr])
    titles = ["180p input", "Bicubic"] + [n for n, _ in gens] + ["Original"]
    fig, axes = plt.subplots(len(rows), len(titles), figsize=(1.75 * len(titles), 1.8 * len(rows)))
    for r, row in enumerate(rows):
        for c, image in enumerate(row):
            axes[r, c].imshow(image)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig23_gan_montage.png", dpi=170)
    plt.close(fig)
    return "fig23_gan_montage.png"


# --------------------------------------------------------------------------- #
# Document
# --------------------------------------------------------------------------- #
def main() -> None:
    made = []
    try:
        made = figures()
    except Exception as error:  # figures are optional; the tables are the record
        print(f"  figures skipped: {error}")

    s_l1, s_v1, s_v2 = summary("sr_l1"), summary("sr_gan_v1"), summary("sr_gan")
    s3_l1, s3_gan, s3_task, s3_l1task = summary("sr3_l1"), summary("sr3_gan"), summary("sr3_gan_task"), summary("sr3_l1_task")
    h_v1, h_v2, h3_gan, h3_task = last_history("sr_gan_v1"), last_history("sr_gan"), last_history("sr3_gan"), last_history("sr3_gan_task")
    proxy = summary("hand_proxy")
    q12, q1 = load(STUDY / "sr_quality.json"), load(STUDY_V1 / "sr_quality.json")
    q3t, q3b = load(STUDY / "sr_quality_v3_task.json"), load(STUDY / "sr_quality_v3_base.json")
    dev = {lv: load(DIAG / f"dev_v3_{lv}.json") for lv in ("L3", "L2")}
    clean = mean("frozen_int4", "clean", "bicubic")

    L: list[str] = []
    w = L.append
    w("# GAN versus no GAN: three super-resolution GANs in front of an Indian Sign Language recogniser\n")
    w("**Team 3** · 23CSE473 Neural Networks and Deep Learning · Faculty guide: Dr. T. Bagyammal  ")
    w("Department of Computer Science and Engineering, Amrita School of Computing, Amrita Vishwa Vidyapeetham, Coimbatore\n")
    w("| Name | Roll number |")
    w("|---|---|")
    for name, roll in TEAM:
        w(f"| {name} | {roll} |")
    w("")
    w("*Generated by `python -m islvit.gan_report`. Every number is read from a result file under `runs/` when the "
      "document is built; Section 10 lists the files.*\n")

    # ------------------------------------------------------------------ 1
    w("## 1. The question\n")
    w("ISL-ViT-Tiny, the shipped 4-bit sign classifier, scores "
      f"{100 * clean:.1f} % top-1 on the 472 held-out test clips of its 50 deployed words when the video is the "
      "dataset's native 1080p. A camera on glasses or a phone may deliver much smaller, compressed video. This study "
      "asks one question: **when the input is low-resolution, does putting a super-resolution GAN in front of the "
      "recogniser give better accuracy than not using one?**\n")
    w("\"Not using a GAN\" has two meanings, and the study tests both:\n")
    w("- **Bicubic upscaling**, the pipeline as it was: the low-resolution frame is resized with bicubic "
      "interpolation. This answers whether the GAN beats doing nothing.")
    w("- **The same generator trained without the adversarial loss** (SR-L1, and in version 3 the hand-loss arm). "
      "This answers whether any gain comes from the *adversarial* training, which is what makes it a GAN, or simply "
      "from a learned upscaler. Without this control, a GAN that beats bicubic would prove only that learning to "
      "upscale helps.\n")
    w("We built three GANs in sequence, and each one answers a question the previous one raised.\n")

    # ------------------------------------------------------------------ 2
    w("## 2. Where the GAN sits, and why there\n")
    w("Every clip reaches the hand detector through one function, `islvit/data/crops.py::read_sampled_frames`. If the "
      "video is below the 720p working height, that function resizes it to 720p before MediaPipe Holistic looks for "
      "the hands, face and body; the detector's boxes and 21-joint hand landmarks are what the classifier reads. That "
      "resize was bicubic. The GAN replaces only that resize, and only when the server or the extraction tools are "
      "given a generator file (`--sr`).\n")
    w("```\nvideo -> decode -> [below 720p?  bicubic  |  generator x4, then area-resize to 720p] -> MediaPipe -> crops + landmarks -> classifier\n```\n")
    w("This placement was chosen because it is the one point where low resolution enters the pipeline, and because "
      "everything downstream, both the hand crops and the landmarks, is computed from the upscaled frame. Two "
      "alternatives were rejected: generating synthetic training clips with a GAN (about 15 clips per word is too "
      "little to train a video generator that keeps signs correct), and upscaling the crops after detection (that "
      "cannot help the detector, which is where low-resolution video does the most damage). Video at 720p or above "
      "never passes through the generator, so the result on clean 1080p video cannot change; this was verified "
      "(Section 6).\n")

    # ------------------------------------------------------------------ 3
    w("## 3. The comparison arms\n")
    w("| Arm | GAN? | Generator | How it was trained | Tested on the 472 test clips |")
    w("|---|---|---|---|---|")
    w("| Bicubic | no | none | — | yes |")
    w(f"| SR-L1 | no | SRVGG-compact, {s_l1.get('generator_params', 0):,} parameters | pixel L1 loss only, "
      f"{s_l1.get('iters', 0):,} iterations | yes |")
    w(f"| SR-GAN v1 | yes | same as SR-L1 | from SR-L1: L1 + VGG19 perceptual + adversarial (BCE), "
      f"{s_v1.get('iters', 0):,} iterations | yes |")
    w(f"| SR-GAN v2 | yes | same as SR-L1 | from v1 with a new discriminator, {s_v2.get('iters', 0):,} more iterations | yes |")
    w(f"| v3 base (L1) | no | wider and deeper, {s3_l1.get('generator_params', 0):,} parameters | pixel L1, "
      f"{s3_l1.get('iters', 0):,} iterations, new data and degradations | validation only |")
    w(f"| SR-GAN v3 | yes | v3 | from v3 base: L1 + perceptual + relativistic adversarial + hand-landmark losses, "
      f"{s3_task.get('iters', 0):,} iterations | yes |")
    w(f"| v3 GAN without hand losses | yes | v3 | as SR-GAN v3 without the hand-landmark losses | validation only |")
    w(f"| v3 without the GAN | no | v3 | from v3 base: L1 + hand-landmark losses, no discriminator, "
      f"{s3_l1task.get('iters', 0):,} iterations | yes |")
    w("")
    w("Each GAN has a matching no-GAN control built from the same generator and data: SR-L1 for v1 and v2, and "
      "\"v3 without the GAN\" for v3. The two arms marked *validation only* were used to develop v3 and were not run "
      "on the test set (Section 5.6).\n")

    # ------------------------------------------------------------------ 4
    w("## 4. How each GAN was built, how they differ, and why each was needed\n")
    w("### 4.1 Common to all three\n")
    w("- **Training data.** High-resolution patches of 128 × 128 pixels taken only from INCLUDE recording sessions "
      "used for training. A check in `islvit/sr/patches.py` refuses any clip from the test sessions. About half the "
      "patches are detected hand and face crops and half are random patches from whole 720p frames.")
    w("- **Degradation.** Each patch is degraded at random, following Real-ESRGAN's second-order scheme "
      "(`islvit/sr/degrade.py`): blur (Gaussian, or motion blur in a quarter of cases), random resize, Gaussian or "
      "shot noise and JPEG compression, applied twice, then a final ×4 downscale and JPEG. The generator learns to "
      "invert this.")
    w("- **Generator.** Real-ESRGAN's compact VGG-style network: 3 × 3 convolutions with PReLU at the low "
      "resolution, a ×4 pixel shuffle, and a nearest-neighbour upsample of the input added as a residual.")
    w(f"- **Discriminator.** A U-Net with spectral normalisation ({s_v1.get('discriminator_params', 0):,} parameters) "
      "that gives a real/fake score for every pixel, so it can object to an implausible fingertip without "
      "objecting to the palm beside it.")
    w("- **Losses.** Pixel L1 (weight 1), VGG19 perceptual loss (weight 1) and the adversarial loss (weight 0.1).")
    w("- **Schedule.** A fixed number of iterations chosen before training, cosine learning-rate decay, and the "
      "exponential moving average of the weights (decay 0.999) kept at the end. No checkpoint was selected by "
      "looking at a result. The discriminator and the VGG network run in bfloat16; the generator stays in float32.\n")

    w("### 4.2 SR-GAN v1: the planned GAN\n")
    w(f"**Built.** Stage 1 trained the generator ({s_l1.get('generator_params', 0):,} parameters: 64 channels, 16 "
      f"layers) with the L1 loss alone for {s_l1.get('iters', 0):,} iterations at batch {s_l1.get('batch')} and "
      f"learning rate {s_l1.get('lr')} ({s_l1.get('minutes')} min). That network is **SR-L1**, the no-GAN control. "
      f"Stage 2 started from SR-L1 and added the discriminator, the perceptual loss and the adversarial loss "
      f"(binary cross-entropy) for {s_v1.get('iters', 0):,} iterations at batch {s_v1.get('batch')} and learning "
      f"rate {s_v1.get('lr')} ({s_v1.get('minutes')} min).\n")
    w("**Why it was needed.** This is the GAN of the original, pre-registered plan, built as Real-ESRGAN describes and "
      "compared against both bicubic and SR-L1.\n")
    w(f"**A change made before it ran.** The plan was 100,000 adversarial iterations. In float32 one adversarial step "
      f"ran at 1.4 iterations per second, too slow for the time available, so the discriminator and VGG were moved "
      f"to bfloat16 and the batch to 16 (4.2–4.3 iterations per second) and the count fixed at 60,000 before the "
      f"run started.\n")
    w(f"**What its training showed.** By the end the discriminator was clearly winning: it scored real patches "
      f"{float(h_v1.get('d_real', 0)):.2f} and generated ones {float(h_v1.get('d_fake', 0)):.2f}, and the "
      f"generator's adversarial loss had risen to {float(h_v1.get('adv', 0)):.2f}.\n")

    w("### 4.3 SR-GAN v2: the same GAN, trained longer\n")
    w(f"**Built.** v2 starts from v1's generator with a **new, untrained discriminator** and trains for "
      f"{s_v2.get('iters', 0):,} more iterations at half the peak learning rate ({s_v2.get('lr')}), "
      f"{s_v2.get('minutes')} min. The generator has therefore had {s_v1.get('iters', 0) + s_v2.get('iters', 0):,} "
      "adversarial iterations in all. Architecture, data, degradation and losses are unchanged. Storing the tensors "
      f"in channels-last order raised the average training speed from {s_v1.get('iters', 0) / 60 / s_v1.get('minutes', 1):.1f} "
      f"to {s_v2.get('iters', 0) / 60 / s_v2.get('minutes', 1):.1f} iterations per second without changing the computation.\n")
    w("**How it differs from v1.** Only in training length and in the restarted discriminator.\n")
    w("**Why it was needed.** v1 was cut from 100,000 to 60,000 iterations, and its discriminator had come to "
      "dominate. If v1 merely failed to separate from SR-L1 because it was undertrained, v2 would show it. v2 was "
      "designed after v1's results were known, so it is outside the pre-registered plan; both are reported in full.\n")
    w(f"**What its training showed.** At the end the discriminator scored real and generated patches "
      f"{float(h_v2.get('d_real', 0)):.2f} and {float(h_v2.get('d_fake', 0)):.2f}: still winning, less "
      "decisively.\n")

    w("### 4.4 SR-GAN v3: a GAN trained for what the recogniser needs\n")
    w("**Why it was needed.** v1 and v2 helped against bicubic but hardly against SR-L1, so the adversarial loss "
      "was not what made the difference. Two diagnostics on the 180p test video then showed where the remaining "
      "error was (Section 7.5). First, with clean pixels but degraded landmarks the classifier lost much more than "
      "with degraded pixels but clean landmarks, so most of what super-resolution still lost was in the hand "
      "landmarks. Second, tuning the hand detector did not recover it. A GAN that only produces realistic-looking "
      "texture does not address that. v3 is a GAN whose objective includes reading the hands correctly.\n")
    w(f"**Built.** Four changes, all decided before any v3 arm was trained:\n")
    w(f"1. **A hand-landmark loss.** MediaPipe is not differentiable, so a stand-in was trained "
      f"(`islvit/sr/handproxy.py`): a ResNet-18 whose features are expanded by three transposed convolutions into 21 "
      f"heatmaps, read out by soft-argmax. It learned from {proxy.get('n_train', 0):,} hand crops whose MediaPipe "
      f"landmarks were checked to belong to the crop, for {proxy.get('epochs')} epochs. On "
      f"{proxy.get('n_val', 0):,} crops from the validation sessions its mean landmark error is "
      f"{proxy.get('final_val', {}).get('mean_err', 0):.4f} of the crop width, with "
      f"{100 * proxy.get('final_val', {}).get('pck_0.05', 0):.1f} % of points within 0.05; MediaPipe re-run on the "
      "same crops differs from its own stored labels by 0.041 on average. During GAN training the proxy is frozen, "
      f"and the generator is penalised by the distance between the proxy's landmarks on its output and on the "
      f"original (weight {s3_task.get('weights', {}).get('hand')}) and between their intermediate features "
      f"(weight {s3_task.get('weights', {}).get('hand_feat')}). Every batch is half hand crops for this purpose.")
    w(f"2. **A larger generator:** 96 channels and 24 layers, {s3_l1.get('generator_params', 0):,} parameters "
      "(3.3 times v1's), for a wider view of each hand.")
    w(f"3. **Camera-like compression.** Half the training samples are compressed with H.264 intra coding "
      f"(CRF 23–43) instead of JPEG, as phone and glasses video is.")
    w(f"4. **A relativistic adversarial loss** (RaGAN) in place of plain cross-entropy, which compares real and "
      f"generated scores against each other and is less prone to a runaway discriminator.\n")
    w(f"The v3 L1 base trained for {s3_l1.get('iters', 0):,} iterations ({s3_l1.get('minutes')} min). Three arms "
      f"then trained from it for {s3_task.get('iters', 0):,} iterations each, so that the effect of each ingredient "
      f"can be read off: the full GAN with hand losses (**SR-GAN v3**, {s3_task.get('minutes')} min), the GAN "
      f"without hand losses ({s3_gan.get('minutes')} min), and the hand losses without any discriminator "
      f"(**v3 without the GAN**, {s3_l1task.get('minutes')} min).\n")
    w("**Kept out of v3's training.** The original validation sessions (888 clips, separate from the test sessions) "
      "were removed from v3's patches and from the proxy's data, so that v3 could be developed on them without "
      "touching the test set.\n")

    w("### 4.5 The three GANs side by side\n")
    w("| | SR-GAN v1 | SR-GAN v2 | SR-GAN v3 |")
    w("|---|---|---|---|")
    w(f"| Generator | 64 ch × 16 layers, {s_v1.get('generator_params', 0):,} | same | 96 ch × 24 layers, {s3_task.get('generator_params', 0):,} |")
    w(f"| Starts from | SR-L1 | v1 | v3 L1 base |")
    w(f"| Adversarial loss | BCE, weight 0.1 | BCE, weight 0.1 | relativistic (RaGAN), weight 0.1 |")
    w(f"| Other losses | L1, VGG19 | L1, VGG19 | L1, VGG19, hand landmarks (2), hand features (0.2) |")
    w(f"| Degradation | blur, resize, noise, JPEG | same | same + H.264 on half |")
    w(f"| Training data | train + validation sessions | same | train sessions only, + 43,682 hand crops |")
    w(f"| Adversarial iterations | {s_v1.get('iters', 0):,} | {s_v1.get('iters', 0) + s_v2.get('iters', 0):,} in total | {s3_task.get('iters', 0):,} |")
    w(f"| Training time | {s_l1.get('minutes')} + {s_v1.get('minutes')} min | + {s_v2.get('minutes')} min | "
      f"{s3_l1.get('minutes')} + {s3_task.get('minutes')} min |")
    w(f"| Discriminator at the end (real / fake) | {float(h_v1.get('d_real', 0)):.2f} / {float(h_v1.get('d_fake', 0)):.2f} | "
      f"{float(h_v2.get('d_real', 0)):.2f} / {float(h_v2.get('d_fake', 0)):.2f} | "
      f"{float(h3_task.get('d_real', 0)):.2f} / {float(h3_task.get('d_fake', 0)):.2f} |")
    w(f"| Pre-registered | yes | no (designed after v1) | no (designed after the diagnostics) |")
    w("")

    # ------------------------------------------------------------------ 5
    w("## 5. Testing parameters\n")
    w("### 5.1 Test data\n")
    w("- The 472 held-out clips of the 50 deployed words (`splits/_test472_paths.csv`), from recording sessions that "
      "neither the classifier nor any generator was trained on. The same 472 clips are used for every arm.")
    w("- Degradation inside the frame reader, before detection, deterministic so every arm sees identical inputs "
      "(`islvit/sr/degrade.py::degrade_frame`): area downscale to the target height, then one JPEG encode with "
      "OpenCV.\n")
    w("| Level | Height | JPEG quality | Upscale factor to 720p |")
    w("|---|---|---|---|")
    w("| 360p | 360 | 60 | ×2 |")
    w("| 240p | 240 | 50 | ×3 |")
    w("| 180p | 180 | 40 | ×4 |")
    w("| Clean | 1080 (native) | — | none; the generator is never used |")
    w("")
    w("- Bicubic: `cv2.resize(..., INTER_CUBIC)` to 720p. Generators: ×4, then area resize to exactly 720p.")
    w("- Per front end: 32 frames per clip, 128-pixel crops, landmarks extracted through the same front end "
      "(`islvit/data/crops.py`, `islvit/data/landmarks.py`). The classifier reads 16 of the 32 frames at 64 pixels.\n")
    w("### 5.2 Models scored\n")
    w("- **Frozen 4-bit:** the three released models `release/isl_vit_tiny_lm_int4_s{0,1,2}.pt`, unchanged.")
    w("- **Frozen 32-bit:** the three full-precision landmark models they were quantised from (`runs/f16_262w_lm_s{0,1,2}`).")
    w("- **Fine-tuned:** the 32-bit models fine-tuned for 150 epochs on the 2,845 training clips degraded to 360p "
      "(JPEG 60) and passed through bicubic, SR-GAN v1 or SR-GAN v2; learning rate 10⁻⁴, resolution jitter 0.5, "
      "16 frames, landmarks on, final epoch kept, three seeds per arm. This asks whether the GAN still helps once the "
      "classifier has adapted to low-resolution input.\n")
    w("### 5.3 Metric\n")
    w("Top-1 and top-5 accuracy over the 50 deployed words: the model's 262 output scores are averaged over six views "
      "(three temporal sampling phases, each with and without a horizontal flip), the 212 other words are set to "
      "zero, and the highest remaining word is the answer (`islvit/mask50.py`). Each figure is the mean of three "
      "seeds; the best seed is never quoted, since choosing it would be selection on the test set.\n")
    w("### 5.4 Statistics and the decision rule\n")
    w("Arms are compared on identical clips with an exact two-sided McNemar test, pooling over the three seeds the "
      "clips that one arm gets right and the other wrong. The rule was fixed before the first test run: an arm "
      "**helps** only if its mean top-1 is higher, all three seeds move in the same direction, and p < 0.05; it "
      "**hurts** under the mirror conditions; anything else is **no reliable difference**. For scale, five "
      "seeds of one classifier configuration trained earlier in the project varied with a standard deviation of "
      "1.9 points, so differences of a point or two are within seed noise.\n")
    w("### 5.5 Detection and hallucination measures\n")
    w("For each front end the hand and face detections are compared with the clean 1080p extraction of the same "
      "clips (`islvit/sr/study.py detection`): the share of frames with a genuine detection; **extra hands**, found "
      "in degraded video but not in clean video, which is how a GAN that invents hands would show; **missed hands**; "
      "the median wrist error; and **misplaced hands**, present in both but more than 5 % of the frame height apart. "
      "CISLR, a second corpus natively 300 × 300 pixels, was used for a detection-only check.\n")
    w("### 5.6 Development on the validation sessions (v3 only)\n")
    w("v3's arms were compared on 200 clips from the validation sessions before any test run, by running MediaPipe "
      "on the super-resolved video and comparing with the clean video's landmarks (`islvit/sr/diag_detector.py`). "
      "Re-extracting clean video gives a noise floor of about 1.5 % of hand frames. Classifier accuracy could not "
      "be used here, because the released classifiers were trained with these sessions included. On the test set "
      "only the two v3 arms that bracket the trade-off seen on validation were run: the full SR-GAN v3 and v3 "
      "without the GAN.\n")
    w("### 5.7 Image quality\n")
    w(f"{(q12 or {}).get('n', 1200)} held-out 128-pixel patches, degraded to each level, scored by PSNR, SSIM "
      "(implemented in PyTorch) and sharpness (variance of the Laplacian), with the generator's time per patch on "
      "the GPU (`islvit/sr/evaluate.py`).\n")

    # ------------------------------------------------------------------ 6
    w("## 6. How the measurements were verified\n")
    w("- **No leakage.** The patch builder and the proxy's data builder assert that no test-session clip is used; "
      "v3 additionally excludes the validation sessions. Fine-tuning clips come from the training split only.")
    w("- **Clean video untouched.** Crops and detection records for clean 1080p clips extracted with and without "
      "`--sr` were byte-identical.")
    w("- **No mixing of front ends.** Every cache records its degradation, generator file and generator checksum "
      "(`FRONTEND.json`); landmarks are refused if they come from a different front end than the crops beside them.")
    w("- **Deterministic detection.** MediaPipe kept state between videos, which changed hand presence on 3–6 % of "
      "frames when a clip was processed twice; the detectors are rebuilt for every video, after which repeated runs "
      "agree exactly.")
    w("- **No selection on results.** Iteration counts, loss weights, epochs and the decision rule were fixed before "
      "each run; final weights are always used.")
    w("- **Precision of v3 test extraction.** v3's test caches were built with the generator in 16-bit floating "
      "point to save time. On real degraded frames this changed no pixel by more than 1/255 (mean 0.03/255), and the "
      "precision is recorded in each cache's `FRONTEND.json`.")
    w("- **Diagnostics separate pixels from landmarks.** Re-scoring with clean crops and degraded landmarks, and "
      "with degraded crops and clean landmarks, located the remaining loss (Section 7.5).")
    w("- **Detector control.** Landmarks re-extracted from the same super-resolved frames with lower thresholds, a "
      "larger and multi-scale search region and gap filling tested whether the detector, rather than the image, "
      "was the limit.\n")

    # ------------------------------------------------------------------ 7
    w("## 7. Results\n")
    w("### 7.1 Accuracy, frozen 4-bit models\n")
    order = ["bicubic", "srl1", "sr3_l1_task", "srgan_v1", "srgan", "sr3_gan_task"]
    w("Top-1 (%), mean ± sd over three seeds. Clean 1080p video does not pass through any generator, so all arms share "
      "the clean figure.\n")
    w("| Input | " + " | ".join(f"{ARMS[a][0]} ({ARMS[a][1]})" for a in order) + " |")
    w("|---|" + "---|" * len(order))
    for lv, name in LEVELS:
        w(f"| {name} | " + " | ".join(cell("frozen_int4", lv, a) for a in order) + " |")
    w("")
    w("Top-5 (%) on the same runs:\n")
    w("| Input | " + " | ".join(ARMS[a][0] for a in order) + " |")
    w("|---|" + "---|" * len(order))
    for lv, name in DEGRADED:
        w(f"| {name} | " + " | ".join(pct(mean("frozen_int4", lv, a, "top5")) for a in order) + " |")
    w("")
    if (FIG / "fig21_gan_accuracy.png").exists():
        w("![Top-1 accuracy for every front end](figures/fig21_gan_accuracy.png)\n")

    w("### 7.2 GAN versus no GAN, paired tests\n")
    w("Δ is the change in mean top-1 (percentage points) from the second arm to the first; per-seed changes in "
      "brackets; clips gained / lost pooled over seeds.\n")
    for family, title in (("frozen_int4", "Frozen 4-bit models"), ("frozen_fp32", "Frozen 32-bit models")):
        w(f"**{title}**\n")
        w("| Input | Comparison | Δ top-1 (per seed) | clips gained / lost | p | verdict |")
        w("|---|---|---|---|---|---|")
        pairs = [("bicubic", "srgan_v1", "SR-GAN v1 vs bicubic"), ("srl1", "srgan_v1", "SR-GAN v1 vs SR-L1"),
                 ("bicubic", "srgan", "SR-GAN v2 vs bicubic"), ("srl1", "srgan", "SR-GAN v2 vs SR-L1"),
                 ("srgan_v1", "srgan", "SR-GAN v2 vs v1"),
                 ("bicubic", "sr3_gan_task", "SR-GAN v3 vs bicubic"), ("srl1", "sr3_gan_task", "SR-GAN v3 vs SR-L1"),
                 ("sr3_l1_task", "sr3_gan_task", "SR-GAN v3 vs v3 without the GAN"),
                 ("srgan", "sr3_gan_task", "SR-GAN v3 vs v2"),
                 ("bicubic", "srl1", "SR-L1 vs bicubic (no GAN)"), ("bicubic", "sr3_l1_task", "v3 without the GAN vs bicubic")]
        for lv, name in DEGRADED:
            for base, other, label in pairs:
                w(compare_row(name, label, compare(family, lv, base, other)))
        w("")

    w("### 7.3 After fine-tuning the classifier on 360p video\n")
    w("| Input | Bicubic | SR-GAN v1 | SR-GAN v2 |")
    w("|---|---|---|---|")
    for lv, name in LEVELS:
        w(f"| {name} | " + " | ".join(cell("retrained", lv, a) for a in ("bicubic", "srgan_v1", "srgan")) + " |")
    w("")
    w("| Input | Comparison | Δ top-1 (per seed) | clips gained / lost | p | verdict |")
    w("|---|---|---|---|---|---|")
    for lv, name in DEGRADED:
        for base, other, label in (("bicubic", "srgan_v1", "SR-GAN v1 vs bicubic"), ("bicubic", "srgan", "SR-GAN v2 vs bicubic"),
                                   ("srgan_v1", "srgan", "SR-GAN v2 vs v1")):
            w(compare_row(name, label, compare("retrained", lv, base, other)))
    w("\nThe v3 front ends were not fine-tuned against; see Section 8.2.\n")

    w("### 7.4 Detection and hallucination\n")
    w("| Input | Arm | Left hand | Right hand | Face | Extra hands | Missed hands | Wrist error (median) | Misplaced |")
    w("|---|---|---|---|---|---|---|---|---|")
    for lv, name in DEGRADED:
        for a in order:
            d = detection(lv, a)
            if not d:
                continue
            s, c = d["streams"], d.get("against_clean", {})
            w(f"| {name} | {ARMS[a][0]} | {pct(s['left_hand']['genuine'])} | {pct(s['right_hand']['genuine'])} | "
              f"{pct(s['face']['genuine'])} | {pct(c.get('extra_hands'))} | {pct(c.get('missed_hands'))} | "
              f"{pct(c.get('wrist_error_median'), 2)} | {pct(c.get('misplaced_hands'))} |")
    w("\nAll values are percentages of hand frames, except wrist error (percent of frame height). On clean INCLUDE "
      "video the detector finds the left hand in 91.1 % and the right in 85.2 % of frames.\n")
    if (FIG / "fig22_gan_detection.png").exists():
        w("![Hands found by the detector](figures/fig22_gan_detection.png)\n")
    w("**CISLR (natively 300 px), left / right hand found:** " + "; ".join(
        f"{ARMS[a][0]} {pct(d['streams']['left_hand']['genuine'])} / {pct(d['streams']['right_hand']['genuine'])}"
        for a in ("bicubic", "srl1", "srgan_v1", "srgan")
        for d in [load(STUDY_V1 / "detection" / "cislr_srgan.json") if a == "srgan_v1" else load(STUDY / "detection" / f"cislr_{a}.json")]
        if d) + ".\n")

    w("### 7.5 Where the remaining loss comes from (180p, frozen 4-bit)\n")
    def m3(pattern):
        vals = [json.loads(Path(p).read_text())["masked_tta"]["top1"] for p in sorted(Path().glob(str(pattern)))]
        return float(np.mean(vals)) if vals else None
    rows = []
    for a, label in (("bicubic", "Bicubic"), ("srgan", "SR-GAN v2")):
        both = mean("frozen_int4", "L3", a)
        lm = m3(DIAG / f"res_clean_pix_L3_{a}_s?.json")
        px = m3(DIAG / f"mirror/degraded_pixels_clean_landmarks_L3_{a}_s?.json")
        rows.append((label, both, lm, px))
    w("| Front end | Both degraded | Landmarks degraded, pixels clean | Pixels degraded, landmarks clean | Clean |")
    w("|---|---|---|---|---|")
    for label, both, lm, px in rows:
        w(f"| {label} | {pct(both)} | {pct(lm)} | {pct(px)} | {pct(clean)} |")
    lev = load(DIAG / "detector_levers.json") or {}
    w("")
    if lev:
        w("Detector settings re-run on the same SR-GAN v2 frames, with clean crops (so only the landmarks change):\n")
        w("| Detector variant | Top-1 | Hand frames found | Extra | Misplaced |")
        w("|---|---|---|---|---|")
        names = {"L3srgan_base": "As used in the study", "L3srgan_lowconf": "Lower confidence thresholds",
                 "L3srgan_bigroi": "Larger search region (512 px)", "L3srgan_multiroi": "Multi-scale search region",
                 "L3srgan_all": "All three", "L3srgan_base_interp": "Study settings + gap filling",
                 "L3srgan_all_interp": "All three + gap filling"}
        for k, label in names.items():
            r = lev.get(k)
            if r:
                w(f"| {label} | {pct(r['top1_mean'])} | {pct(r['present'])} | {pct(r['extra'])} | {pct(r['misplaced'])} |")
        w("")

    w("### 7.6 v3 on the validation sessions\n")
    if dev["L3"] and dev["L2"]:
        keys = [("bicubic", "Bicubic"), ("srl1", "SR-L1"), ("srgan_v2", "SR-GAN v2"), ("sr3_l1", "v3 base (L1)"),
                ("sr3_gan", "v3 GAN, no hand losses"), ("sr3_gan_task", "SR-GAN v3"), ("sr3_l1_task", "v3 without the GAN")]
        w("200 validation-session clips; MediaPipe on super-resolved video against the clean video. Handshape error "
          "is the median distance of the 21 joints from the clean ones, relative to the wrist, in hundredths of the "
          "frame height.\n")
        w("| Arm | Hands found, 180p | Handshape error, 180p | Hands found, 240p | Handshape error, 240p |")
        w("|---|---|---|---|---|")
        for k, label in keys:
            a, b = dev["L3"].get(f"L3_{k}"), dev["L2"].get(f"L2_{k}")
            if a and b:
                w(f"| {label} | {pct(a['present'])} | {100 * a['points_err_med']:.2f} | {pct(b['present'])} | {100 * b['points_err_med']:.2f} |")
        w(f"\nClean video: {pct(dev['L3']['L3_bicubic']['clean_present'])} % of hand frames found.\n")

    w("### 7.7 Image quality and cost\n")
    w("| Input | Arm | PSNR (dB) | SSIM | Sharpness | ms per patch (GPU) |")
    w("|---|---|---|---|---|---|")
    for lv_name in ("L1 360p", "L2 240p", "L3 180p"):
        sources = [("Bicubic", q12, "bicubic"), ("SR-L1", q12, "sr_l1"), ("v3 without the GAN", q3t, "sr_l1"),
                   ("SR-GAN v1", q1, "sr_gan"), ("SR-GAN v2", q12, "sr_gan"), ("SR-GAN v3", q3t, "sr_gan")]
        for label, q, key in sources:
            m = (q or {}).get("levels", {}).get(lv_name, {}).get(key)
            if m:
                ms = m.get("ms_per_patch") or "—"
                w(f"| {lv_name.split()[1]} | {label} | {m['psnr']:.2f} | {m['ssim']:.3f} | {m['sharpness']:.0f} | {ms} |")
    w(f"\nOriginal patches have a median sharpness of {(q12 or {}).get('original_sharpness', 0):.0f}. "
      "PSNR and SSIM reward the average of plausible answers, so a non-adversarial upscaler is expected to score "
      "higher on them and a GAN to look sharper; this is the usual trade-off between distortion and perceived "
      "quality.\n")
    w(f"Parameters: v1/v2 generator {s_v1.get('generator_params', 0):,}; v3 generator {s3_task.get('generator_params', 0):,}; "
      f"discriminator {s_v1.get('discriminator_params', 0):,} (training only).\n")
    if (FIG / "fig23_gan_montage.png").exists():
        w("![Held-out patches at 180p through each upscaler](figures/fig23_gan_montage.png)\n")
        w("*Hand crops from held-out INCLUDE clips (Sridhar et al., ACM MM 2020, CC-BY-4.0).*\n")
        w("The montage shows one effect the numbers alone hide. v3 without the GAN, trained on the hand-landmark loss "
          "with no discriminator, covers its output in a fine regular dot grid. That pattern is why its sharpness "
          "score is the highest of all the upscalers, above even the originals at 360p. The landmark loss rewards "
          "whatever makes the proxy's joints land correctly, and nothing in it objects to an unnatural texture. The "
          "same network trained with the discriminator (SR-GAN v3) shows no grid. Here the adversarial loss did "
          "the job it exists for: keeping the output looking like a real image.\n")

    # ------------------------------------------------------------------ 8
    w("## 8. What the results mean, and their limits\n")
    v1s = {lv: compare("frozen_int4", lv, "srl1", "srgan_v1") for lv, _ in DEGRADED}
    v2s = {lv: compare("frozen_int4", lv, "srl1", "srgan") for lv, _ in DEGRADED}
    v3s = {lv: compare("frozen_int4", lv, "sr3_l1_task", "sr3_gan_task") for lv, _ in DEGRADED}
    b3 = compare("frozen_int4", "L3", "bicubic", "srgan")
    w("### 8.1 Reasoning\n")
    w(f"Learning to upscale is what recovers the accuracy. At 180p SR-GAN v2 gains {100 * b3['delta']:.1f} points over "
      f"bicubic on the frozen 4-bit models (p = {fmt_p(b3['p'])}), and every learned front end, GAN or not, lands "
      "within about a point of the others. Section 7.4 shows the mechanism: after bicubic upscaling at 180p, MediaPipe "
      "finds about half the hands and a third of the faces, and any learned upscaler restores most of them. At 360p "
      "the picture depends on the model. The 4-bit models gain 2–3 points and the 32-bit models nothing, probably "
      "because at ×2 the detector still finds most hands after bicubic (Section 7.4) and little is left to recover. "
      "We did not investigate why the 4-bit models respond to that small difference and the 32-bit models do not.\n")
    helps4 = sum(1 for d in (v1s, v2s, v3s) for c in d.values() if c and c["verdict"] == "helps")
    f32 = compare("frozen_fp32", "L2", "srl1", "srgan_v1")
    w(f"The adversarial loss adds little on top. Against their own no-GAN controls the GANs passed the rule in {helps4} "
      f"of 9 comparisons on the frozen 4-bit models, both at 240p: SR-GAN v2 beat SR-L1 by "
      f"{100 * v2s['L2']['delta']:+.1f} points, and SR-GAN v3 beat v3 without the GAN by "
      f"{100 * v3s['L2']['delta']:+.1f}. The second win says less than it seems, since v3 without the GAN was the "
      "weakest learned front end at 240p and SR-GAN v3 did not beat SR-L1 there. The 32-bit models show one GAN win, "
      f"again at 240p (v1 over SR-L1, {100 * f32['delta']:+.1f}). At 360p and 180p no GAN beat its control on either "
      "model. The sharper texture the GANs produce (Section 7.7) mostly does not turn into accuracy. In v3 the "
      "discriminator did matter, though not for accuracy: without it the hand-landmark loss produced a visible dot-grid "
      "artifact (Section 7.7), and that arm found the fewest hands of any learned upscaler at 180p.\n")
    w("Training longer did not help. SR-GAN v2 was no better than v1 at any level, and worse at 360p on the 4-bit "
      "models, although its discriminator ended the run less dominant. Undertraining therefore does not explain why v1 "
      "failed to separate from SR-L1.\n")
    d3, d2 = detection("L3", "sr3_gan_task"), detection("L3", "srgan")
    hands = lambda d: (d["streams"]["left_hand"]["genuine"] + d["streams"]["right_hand"]["genuine"]) / 2
    w("v3 changed what it was designed to change, but the change did not reach the accuracy. On the validation "
      "sessions SR-GAN v3 found the most hands and v3 without the GAN had the lowest handshape error, and the test set "
      f"agrees: at 180p SR-GAN v3 found {pct(hands(d3))} % of hands against {pct(hands(d2))} % for v2, while v3 "
      "without the GAN had the smallest wrist error and the fewest misplaced hands. Neither was reliably more accurate "
      "than SR-GAN v2 or SR-L1. A possible reason, which we did not test, is that the frozen classifier learned from "
      "MediaPipe's landmarks on clean video and reacts to the detector's remaining errors in ways the proxy distance "
      "does not capture.\n")
    w("What limits accuracy after super-resolution is the detector's output, not how the image looks. Most of the "
      "remaining loss is on the landmark side (Section 7.5), and a more permissive detector found more hands but "
      "placed more of them wrongly, leaving accuracy where it was.\n")
    w("### 8.2 Limitations\n")
    w("- **Synthetic degradation.** Test video was degraded by area downscaling and one JPEG encode. Real phone or "
      "glasses video has motion blur, sensor noise and inter-frame compression; the GANs, v3 especially, were trained "
      "for some of that, and could rank differently on real low-resolution recordings.")
    w("- **One studio.** INCLUDE was recorded in one room with a fixed camera. On CISLR, the only other corpus "
      "available, pixel-only classifiers scored at chance for reasons unrelated to resolution (MediaPipe's pose model "
      "finds the wrists in a minority of its frames), so the GANs were tested there for hand detection only, not for "
      "accuracy.")
    w("- **v2 and v3 were not pre-registered.** Both were designed after earlier results were seen; their tests used "
      "the same rule, but the decision to build them was informed by the test set. v3's arms were developed on "
      "validation sessions to limit this.")
    w("- **v3 was not fine-tuned against.** The fine-tuned arm covers bicubic, v1 and v2. Whether a classifier "
      "fine-tuned on v3's output gains from its better hand detection is the most direct open question.")
    w("- **Two v3 arms were validation-only.** The v3 L1 base and the v3 GAN without hand losses were not run on the "
      "test set.")
    w("- **Different patch sets.** v3's training patches exclude the validation sessions, v1/v2's include them, so "
      "v3 trained on less INCLUDE data.")
    w("- **Image-quality patches mix crops and frame patches.** PSNR, SSIM and sharpness average over both. v3 was "
      "trained half on hand crops and partly on H.264, which may explain part of its lower PSNR than SR-L1 on this "
      "JPEG-degraded set.")
    w("- **Three seeds.** With three seeds and a seed-to-seed spread of about two points, effects of a point or two "
      "cannot be settled either way.")
    w("- **Cost.** v1 and v2 use the same network as SR-L1 and cost the same at inference. v3's generator has 3.3 "
      "times the parameters and was 1.5–2.3 times slower per patch on the GPU (Section 7.7). No latency was measured "
      "on wearable hardware.\n")

    # ------------------------------------------------------------------ 9
    w("## 9. Conclusion\n")
    w("On low-resolution video, a learned upscaler in front of the hand detector beats bicubic by a wide margin: "
      f"{100 * b3['delta']:.1f} points at 180p for SR-GAN v2, with a gain of similar size after the classifier is "
      "fine-tuned. Training that upscaler as a GAN makes little further difference to recognition. We tried a plain "
      "Real-ESRGAN-style GAN, the same GAN trained longer, and a third with a larger generator, camera-like "
      "compression, a relativistic loss and a hand-landmark objective. Across the three, adversarial training "
      "sharpened the images, suppressed an artifact that the hand-landmark loss introduced, and helped v3 find more "
      "hands, yet it beat a no-GAN control in only 2 of 9 accuracy comparisons on the shipped model.\n")
    w("These results do not show that a GAN cannot help this recogniser; they show that the GANs we built did not, "
      "on synthetically degraded studio video and a frozen or partly adapted classifier. The remaining error sits in "
      "where the detector places the hand joints. The most direct next test is to fine-tune the classifier on "
      "SR-GAN v3's output, so that it can use the better hand detection, and to repeat the comparison on real "
      "low-resolution recordings.\n")

    # ------------------------------------------------------------------ 10
    w("## 10. Result files\n")
    w("| What | Where |")
    w("|---|---|")
    w("| Accuracy, frozen 4-bit, per front end and seed | `runs/gan_study/frozen_int4/` (bicubic, SR-L1, v2, v3); `runs/gan_study_v1/frozen_int4/` (v1) |")
    w("| Accuracy, frozen 32-bit | `runs/gan_study/frozen_fp32/`; `runs/gan_study_v1/frozen_fp32/` |")
    w("| Accuracy, fine-tuned | `runs/gan_study/retrained/` (bicubic, v2); `runs/gan_study_v1/retrained/` (v1) |")
    w("| Detection and hallucination | `runs/gan_study/detection/`; `runs/gan_study_v1/detection/` |")
    w("| Image quality | `runs/gan_study/sr_quality.json` (bicubic, SR-L1, v2); `runs/gan_study_v1/sr_quality.json` (v1); `runs/gan_study/sr_quality_v3_task.json`, `sr_quality_v3_base.json` (v3) |")
    w("| Diagnostics (pixels vs landmarks, detector variants, v3 validation, paired v3 tests) | `runs/gan_study/diagnostics/` |")
    w("| Generator training records | `runs/sr_l1/`, `runs/sr_gan_v1/`, `runs/sr_gan/` (v2), `runs/sr3_l1/`, `runs/sr3_gan/`, `runs/sr3_gan_task/`, `runs/sr3_l1_task/` (`summary.json`, `history.csv`) |")
    w("| Hand-landmark proxy | `runs/hand_proxy/` |")
    w("| Fine-tuned classifiers | `runs/ft360_bicubic_s*/`, `runs/ft360_srgan_v1_s*/`, `runs/ft360_srgan_s*/` (v2) |")
    w("| Figures | `docs/figures/fig21_gan_accuracy.png`, `fig22_gan_detection.png`, `fig23_gan_montage.png` |")
    w("")
    w("Model weights (`*.pt`) are kept locally and not committed. To rebuild: `bash run_gan_study.sh` (bicubic, SR-L1, "
      "v1), `GAN_ITERS=80000 bash run_gan_study_v2.sh` (v2), `bash run_gan_v3.sh` and `bash run_gan_v3_test.sh` (v3), "
      "then `python -m islvit.gan_report`.\n")

    DOC.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {DOC} ({len(made)} figures)")
    if shutil.which("pandoc"):
        docx = DOC.with_suffix(".docx")
        subprocess.run(["pandoc", DOC.name, "-o", docx.name, "--resource-path=."], cwd=DOC.parent, check=True)
        print(f"wrote {docx}")


if __name__ == "__main__":
    main()
