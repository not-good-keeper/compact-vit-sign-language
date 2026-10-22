"""Charts for the Review 3 report and slides. Every number is read from a result file in runs/.

    python report/review3/make_charts.py
"""
import glob
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / "assets"
OUT.mkdir(exist_ok=True)
RUNS = ROOT / "runs"
INK, MUTED, GRID = "#1F2430", "#6B7280", "#E5E7EB"
COL = {"bicubic": "#A3AAB8", "srl1": "#5B7BD6", "srgan_v1": "#C065B0", "srgan": "#2E9E6E",
       "sr3_l1": "#7C8DB5", "sr3_gan": "#D98C2B", "sr3_gan_task": "#1E7A5A", "sr3_l1_task": "#8E5BB5"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})


def top1(pattern, key="masked_tta"):
    paths = sorted(glob.glob(str(pattern)))
    vals = [json.loads(Path(p).read_text())[key]["top1"] for p in paths]
    assert vals, pattern
    return 100 * st.mean(vals)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=220)
    plt.close(fig)
    print("wrote", name)


def grid(ax):
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)


# 0. The live Capture screenshot, cropped above the demo-clip strip.
shot = Image.open(OUT / "ui_capture.png")
shot.crop((0, 0, shot.size[0], 1040)).save(OUT / "ui_capture_crop.png")

# 1. Model variants on the 472-clip, 50-word session-disjoint test set (masked-50, six-view TTA, mean of 3 seeds)
models = [
    ("Pixel-only\nFP32", top1(RUNS / "f16_262w_s?/masked50_clean.json"), "14.55 MB", COL["bicubic"]),
    ("+ landmarks\nFP32", top1(RUNS / "f16_262w_lm_s?/masked50_clean.json"), "14.9 MB", COL["srl1"]),
    ("+ landmarks, 4-bit\npost-training", top1(RUNS / "f16_262w_lm_s?/ptq4_clean.json"), "~2 MB", COL["srgan_v1"]),
    ("+ landmarks, 4-bit\nQAT (shipped)", top1(RUNS / "f16_262w_lm_s?_qat4/masked50_clean.json"), "1.97 MiB", COL["srgan"]),
]
fig, ax = plt.subplots(figsize=(6.6, 3.5))
rects = ax.bar([m[0] for m in models], [m[1] for m in models], color=[m[3] for m in models], width=0.62)
for rect, m in zip(rects, models):
    x = rect.get_x() + rect.get_width() / 2
    ax.text(x, m[1] + 1.2, f"{m[1]:.1f}", ha="center", fontsize=9, color=INK)
    ax.text(x, 6, m[2], ha="center", color="white", fontsize=8.5, fontweight="bold")
ax.axhline(75, color="#D14343", ls=":", lw=1.2)
ax.text(3.42, 76.2, "target 75 %", color="#D14343", fontsize=8, ha="right")
ax.set_ylim(0, 100)
ax.set_ylabel("Top-1 accuracy (%)")
ax.set_title("Model variants on 472 held-out clips (mean of 3 seeds)", fontsize=10.5, color=INK)
grid(ax)
save(fig, "c1_models.png")

# 2. Super-resolution front end: frozen INT4 models and fine-tuned models
LEVELS = [("L1", "360p"), ("L2", "240p"), ("L3", "180p")]
clean = top1(RUNS / "f16_262w_lm_s?_qat4/masked50_clean.json")
arms = [("bicubic", "Bicubic (no GAN)", "gan_study", "bicubic"),
        ("srl1", "SR-L1 (no GAN)", "gan_study", "srl1"),
        ("sr3_l1_task", "v3 without the GAN", "gan_study", "sr3_l1_task"),
        ("srgan_v1", "SR-GAN v1", "gan_study_v1", "srgan"),
        ("srgan", "SR-GAN v2", "gan_study", "srgan"),
        ("sr3_gan_task", "SR-GAN v3", "gan_study", "sr3_gan_task")]
FT_ARMS = [("bicubic", "Bicubic", "gan_study"), ("srgan_v1", "SR-GAN v1", "gan_study_v1"), ("srgan", "SR-GAN v2", "gan_study")]
fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), gridspec_kw={"width_ratios": [1.35, 1]})
ax, w = axes[0], 0.135
for j, (key, label, study, name) in enumerate(arms):
    vals = [top1(RUNS / study / f"frozen_int4/{lv}_{name}_s?.json") for lv, _ in LEVELS]
    xs = [i + (j - 2.5) * w for i in range(3)]
    ax.bar(xs, vals, w * 0.95, color=COL[key], label=label)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.8, f"{v:.0f}", ha="center", fontsize=7.5, color=INK)
ax.axhline(clean, color=INK, ls="--", lw=1)
ax.text(2.55, clean + 1.0, f"clean 1080p {clean:.1f}", fontsize=8, ha="right", color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels([f"{n} + JPEG" for _, n in LEVELS]); ax.set_ylim(50, 92)
ax.set_ylabel("Top-1 accuracy (%)"); ax.set_title("Shipped INT4 model, frozen", fontsize=10.5, color=INK)
grid(ax); ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
ax, w = axes[1], 0.26
for j, (key, label, study) in enumerate(FT_ARMS):
    name = "srgan" if key == "srgan_v1" else key
    vals = [top1(RUNS / study / f"retrained/{lv}_{name}_s?.json") for lv, _ in LEVELS]
    xs = [i + (j - 1) * w for i in range(3)]
    ax.bar(xs, vals, w * 0.95, color=COL[key], label=label)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.8, f"{v:.0f}", ha="center", fontsize=7.5, color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels([n for _, n in LEVELS]); ax.set_ylim(50, 92)
ax.set_title("Fine-tuned on 360p video", fontsize=10.5, color=INK); grid(ax)
ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
save(fig, "c2_gan_accuracy.png")

# 3. Where the remaining loss sits at 180p: pixels versus landmarks
D = RUNS / "gan_study" / "diagnostics"
fig, ax = plt.subplots(figsize=(6.6, 3.4))
w = 0.3
cats = ["Both degraded", "Landmarks degraded\n(pixels clean)", "Pixels degraded\n(landmarks clean)"]
for j, (arm, label) in enumerate((("bicubic", "Bicubic"), ("srgan", "SR-GAN v2"))):
    vals = [top1(RUNS / f"gan_study/frozen_int4/L3_{arm}_s?.json"),
            top1(D / f"res_clean_pix_L3_{arm}_s?.json"),
            top1(D / f"mirror/degraded_pixels_clean_landmarks_L3_{arm}_s?.json")]
    xs = [i + (j - 0.5) * w for i in range(3)]
    ax.bar(xs, vals, w * 0.95, color=COL[arm], label=label)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.8, f"{v:.1f}", ha="center", fontsize=8, color=INK)
ax.axhline(clean, color=INK, ls="--", lw=1)
ax.text(2.45, clean + 0.9, f"clean {clean:.1f}", fontsize=8, ha="right")
ax.set_xticks(range(3)); ax.set_xticklabels(cats); ax.set_ylim(50, 97); ax.set_ylabel("Top-1 accuracy (%)")
ax.set_title("180p + JPEG 40: which input carries the loss", fontsize=10.5, color=INK)
grid(ax); ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
save(fig, "c3_decomposition.png")

# 4. v3 generators on the validation sessions (real MediaPipe on 200 clips, against clean landmarks)
dev = {lv: json.loads((D / f"dev_v3_{lv}.json").read_text()) for lv in ("L3", "L2")}
order = [("bicubic", "Bicubic", COL["bicubic"]), ("srl1", "SR-L1", COL["srl1"]),
         ("srgan_v2", "SR-GAN v2", COL["srgan"]), ("sr3_l1", "v3 base (L1)", COL["sr3_l1"]),
         ("sr3_gan", "v3 GAN, no hand loss", COL["sr3_gan"]), ("sr3_gan_task", "SR-GAN v3", COL["sr3_gan_task"]),
         ("sr3_l1_task", "v3 without the GAN", COL["sr3_l1_task"])]
fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6))
for ax, metric, title in ((axes[0], "present", "Hand frames found (%) - clean video 88.3"),
                          (axes[1], "points_err_med", "Handshape error, median (x100; lower is better)")):
    w = 0.38
    for j, lv in enumerate(("L3", "L2")):
        vals = [100 * dev[lv][f"{lv}_{k}"][metric] for k, _, _ in order]
        xs = [i + (j - 0.5) * w for i in range(len(order))]
        ax.bar(xs, vals, w * 0.95, color=[c for _, _, c in order], alpha=1.0 if lv == "L3" else 0.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([l for _, l, _ in order], rotation=35, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10, color=INK); grid(ax)
axes[0].set_ylim(30, 92)
axes[0].text(0.01, 0.97, "solid: 180p   faded: 240p", transform=axes[0].transAxes, fontsize=8, va="top", color=MUTED)
save(fig, "c4_v3_validation.png")

# 5. Leakage, same recipe: the random-split control (one seed, 200 clips) against the mean of the two
#    session-disjoint models trained with that recipe (472 clips). The 262-word model masked to 50 (75.6 %)
#    is a different model and is deliberately not used here.
rand = json.loads((RUNS / "baseline_random_s0/summary.json").read_text())["test"]["top1"] * 100
sess = 100 * st.mean(json.loads((RUNS / f"f16_clean_s{s}/summary.json").read_text())["test"]["top1"] for s in (0, 1))
fig, ax = plt.subplots(figsize=(4.4, 3.3))
rects = ax.bar(["Random video split", "Held-out sessions"], [rand, sess], color=[COL["srgan_v1"], COL["srgan"]], width=0.55)
for rect, v in zip(rects, [rand, sess]):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", fontsize=9, color=INK)
ax.set_ylim(0, 105); ax.set_ylabel("Top-1 accuracy (%)")
ax.set_title("Same recipe, two splits (50 words)", fontsize=10.5, color=INK); grid(ax)
save(fig, "c5_leakage.png")

# 6. Every plotted number, for the slide deck's native (editable) charts.
data = {
    "models": [{"label": m[0].replace("\n", " "), "top1": round(m[1], 3), "size": m[2]} for m in models],
    "frozen_int4": {"levels": [n for _, n in LEVELS], "clean": round(clean, 3), "arms": [
        {"label": label, "values": [round(top1(RUNS / study / f"frozen_int4/{lv}_{name}_s?.json"), 3) for lv, _ in LEVELS]}
        for _, label, study, name in arms]},
    "retrained": {"levels": [n for _, n in LEVELS], "arms": [
        {"label": label, "values": [round(top1(RUNS / study / f"retrained/{lv}_{'srgan' if key == 'srgan_v1' else key}_s?.json"), 3)
                                    for lv, _ in LEVELS]}
        for key, label, study in FT_ARMS]},
    "decomposition": {"categories": [c.replace("\n", " ") for c in cats], "clean": round(clean, 3), "arms": [
        {"label": label, "values": [round(top1(RUNS / f"gan_study/frozen_int4/L3_{arm}_s?.json"), 3),
                                    round(top1(D / f"res_clean_pix_L3_{arm}_s?.json"), 3),
                                    round(top1(D / f"mirror/degraded_pixels_clean_landmarks_L3_{arm}_s?.json"), 3)]}
        for arm, label in (("bicubic", "Bicubic"), ("srgan", "SR-GAN v2"))]},
    "v3_validation": {"labels": [l for _, l, _ in order],
                      **{f"{m}_{lv}": [round(100 * dev[lv][f"{lv}_{k}"][key], 3) for k, _, _ in order]
                         for m, key in (("present", "present"), ("shape_err", "points_err_med")) for lv in ("L3", "L2")}},
    "leakage": {"random": round(rand, 3), "sessions": round(sess, 3)},
}
(OUT / "chart_data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
print("wrote chart_data.json")
