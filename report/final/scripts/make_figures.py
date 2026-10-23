"""Quantitative figures for the final report, read from tracked result files.

Every plotted value comes from report/final/data/audit_results.json (written by
audit_results.py from runs/ and splits/) or directly from runs/*.json. No value
is typed in. Output: vector PDFs in report/final/latex/figures/.

Usage (repository root):
    python report/final/scripts/audit_results.py
    python report/final/scripts/make_figures.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "report" / "final" / "data" / "audit_results.json"
OUT = ROOT / "report" / "final" / "latex" / "figures"
RUNS = ROOT / "runs"

# Reference categorical palette (fixed order), text and grid tokens.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
MARKERS = ["o", "s", "^", "D", "v", "P"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 7.2,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.linewidth": 0.6, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def pct(x):
    return 100 * np.asarray(x, dtype=float)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def fig_variants(a: dict) -> None:
    """(a) top-1 per seed for the four controlled variants; (b) paired discordant clips."""
    variants = [("pixel_fp32", "Pixels\nFP32"), ("landmark_fp32", "Landmarks\nFP32"),
                ("landmark_ptq_int4", "Landmarks\nPTQ INT4"), ("landmark_qat_int4", "Landmarks\nQAT INT4")]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.15, 2.55), gridspec_kw={"width_ratios": [1.1, 1]})
    for i, (key, label) in enumerate(variants):
        seeds = pct([s["top1"] for s in a[key]["per_seed"]])
        m = pct(a[key]["mean"]["top1"])
        ax.plot([i - 0.22, i + 0.22], [m, m], color=INK, lw=1.6, solid_capstyle="round", zorder=3)
        for j, v in enumerate(seeds):
            ax.scatter(i + (j - 1) * 0.09, v, s=26, marker=MARKERS[j], color=C[0], edgecolor="white",
                       linewidth=0.6, zorder=4, label=f"seed {j}" if i == 0 else None)
        ax.text(i + 0.26, m, f"{m:.1f}", va="center", ha="left", color=INK, fontsize=8)
    ax.set_xticks(range(len(variants)), [v[1] for v in variants], fontsize=6.6)
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_ylim(70, 90)
    ax.set_yticks(range(70, 91, 5))
    ax.set_xlim(-0.5, len(variants) - 0.35)
    ax.legend(loc="lower right", fontsize=7, handletextpad=0.2, ncol=3, columnspacing=0.8)
    ax.set_title("(a) Masked-50, 6-view TTA, 472 test clips", fontsize=7.6, loc="left", color=INK)

    comps = [("paired_landmark_vs_pixel", "Landmarks vs pixels only"),
             ("paired_qat_vs_ptq", "QAT INT4 vs post-training INT4"),
             ("paired_qat_vs_fp32", "QAT INT4 vs FP32")]
    y = 0
    yt, yl = [], []
    for key, label in comps:
        bx.text(-44, y - 0.85, label, fontsize=7.2, color=INK, va="center", ha="left")
        for s, d in enumerate(a[key]["per_seed"]):
            bx.barh(y, d["gained"], color=C[2], height=0.7, edgecolor="white", linewidth=1,
                    label="gained" if (key, s) == (comps[0][0], 0) else None)
            bx.barh(y, -d["lost"], color=C[1], height=0.7, edgecolor="white", linewidth=1,
                    label="lost" if (key, s) == (comps[0][0], 0) else None)
            ptxt = f"p = {d['p']:.2g}"
            bx.text(max(d["gained"], 5) + 2, y, ptxt, va="center", fontsize=6.5, color=INK2)
            yt.append(y)
            yl.append(f"seed {s}")
            y += 1
        y += 1.0
    bx.set_yticks(yt, yl, fontsize=6.8)
    bx.set_ylim(y - 0.6, -1.6)
    bx.axvline(0, color=INK2, lw=0.6)
    bx.set_xlim(-45, 125)
    bx.set_xlabel("Test clips that changed outcome")
    bx.set_title("(b) Paired clip-level changes per seed", fontsize=7.6, loc="left", color=INK)
    bx.legend(loc="lower right", fontsize=7)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_variants")


def fig_protocol(a: dict) -> None:
    lm = a["leakage_matched"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.15, 2.2))
    rv = pct(lm["random_video"]["test"]["top1"])
    sd = pct([x["test"]["top1"] for x in lm["session_disjoint"]])
    ax.bar([0], [rv], width=0.55, color=C[1], edgecolor="white")
    ax.bar([1], [sd.mean()], width=0.55, color=C[0], edgecolor="white")
    for j, v in enumerate(sd):
        ax.scatter(1 + (j - 0.5) * 0.14, v, color=INK, s=14, zorder=4, marker=MARKERS[j])
    ax.text(0, rv + 1.5, f"{rv:.1f}\n(1 seed, n={lm['random_video']['test']['n']})", ha="center", fontsize=7.2)
    ax.text(1, sd.mean() + 4.5, f"{sd.mean():.1f}\n(2 seeds, n={lm['session_disjoint'][0]['test']['n']})",
            ha="center", fontsize=7.2)
    ax.set_xticks([0, 1], ["Random-video split", "Session-disjoint split"])
    ax.set_ylim(0, 115)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("(a) Same 50-word recipe, two split protocols", fontsize=7.6, loc="left")

    lad = a["protocol_ladder_8frame"]
    keys = [("full263_v2__random-video", "Random-video"), ("full263_v2__take-group", "Take-group"),
            ("sd_full263", "Session-disjoint")]
    vals = [pct(lad[k]["test"]["top1"]) for k, _ in keys]
    ns = [lad[k]["test"]["n"] for k, _ in keys]
    bx.bar(range(3), vals, width=0.55, color=[C[1], C[3], C[0]], edgecolor="white")
    for i, (v, n) in enumerate(zip(vals, ns)):
        bx.text(i, v + 1.5, f"{v:.1f}\n(n={n})", ha="center", fontsize=7.2)
    bx.set_xticks(range(3), [k[1] for k in keys])
    bx.set_ylim(0, 115)
    bx.set_yticks(range(0, 101, 20))
    bx.set_title("(b) Early 8-frame recipe, INCLUDE 262 words", fontsize=7.6, loc="left")
    fig.tight_layout(w_pad=2)
    save(fig, "fig_protocol")


def fig_ablation(a: dict) -> None:
    fs = a["frame_sweep_32cache"]
    by = defaultdict(list)
    for run, d in fs.items():
        by[d["n_frames"]].append(d)
    frames = sorted(by)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.15, 2.2))
    for k, (key, lab, mk, col) in enumerate((("test", "single view", "o", C[0]), ("test_tta", "6-view TTA", "s", C[1]))):
        means = [np.mean([pct(d[key]["top1"]) for d in by[f]]) for f in frames]
        ax.plot(frames, means, marker=mk, color=col, lw=1.6, ms=5, label=lab)
        for f in frames:
            if len(by[f]) > 1:
                for d in by[f]:
                    ax.scatter(f, pct(d[key]["top1"]), color=col, s=8, alpha=0.8, zorder=4)
    ax.set_xticks(frames)
    ax.set_xlabel("Frames per clip (sampled from a 32-frame cache)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("(a) Temporal length, 50 words, 472 clips", fontsize=7.6, loc="left")

    ra = a["resolution_ablation"]
    keys = ["abl64__take-group", "abl112__take-group", "abl128__take-group"]
    vals = [pct(ra[k]["test"]["top1"]) for k in keys]
    bx.bar(range(3), vals, width=0.55, color=C[0], edgecolor="white")
    for i, v in enumerate(vals):
        bx.text(i, v + 0.8, f"{v:.1f}", ha="center", fontsize=7.5)
    bx.set_xticks(range(3), ["64 px\n16 tokens", "112 px\n49 tokens", "128 px\n64 tokens"])
    bx.set_ylim(0, 60)
    bx.set_ylabel("Top-1 accuracy (%)")
    bx.set_title(f"(b) Crop size, INCLUDE-50 take-group, n={ra[keys[0]]['test']['n']}", fontsize=7.6, loc="left")
    fig.tight_layout(w_pad=2)
    save(fig, "fig_ablation")


def fig_gan(a: dict) -> None:
    g = a["gan"]["frozen_int4"]
    clean = a["landmark_qat_int4"]
    arms = [("bicubic", "Bicubic (no learning)"), ("srl1", "SR-L1 (no GAN)"), ("srgan_v1", "SR-GAN v1"),
            ("srgan", "SR-GAN v2"), ("sr3_l1_task", "v3 without GAN"), ("sr3_gan_task", "SR-GAN v3")]
    levels = ["L3", "L2", "L1"]
    xlab = ["180p", "240p", "360p", "clean\n1080p"]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(5.15, 5.6), gridspec_kw={"height_ratios": [1, 1.15]})
    for k, (arm, label) in enumerate(arms):
        m = [pct(g[lv][arm]["mean"]["top1"]) for lv in levels] + [pct(clean["mean"]["top1"])]
        ls = "--" if arm in ("bicubic", "srl1", "sr3_l1_task") else "-"
        ax.plot(range(4), m, marker=MARKERS[k], color=C[k], lw=1.3, ms=4.5, ls=ls, label=label)
    ax.set_xticks(range(4), xlab)
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_xlabel("Input video resolution (degraded before detection)")
    ax.set_ylim(58, 90)
    ax.legend(fontsize=6.6, loc="lower right", ncol=1, handlelength=2.2)
    ax.set_title("(a) Frozen INT4 model, mean of 3 seeds", fontsize=7.6, loc="left")

    pairs = [("srgan_v1", "srl1", "v1 vs SR-L1"), ("srgan", "srl1", "v2 vs SR-L1"),
             ("sr3_gan_task", "sr3_l1_task", "v3 vs v3 w/o GAN")]
    p = a["gan"]["paired_frozen_int4"]
    y, yt, yl = 0, [], []
    for lv, lvname in (("L1", "360p"), ("L2", "240p"), ("L3", "180p")):
        for k, (b, base, lab) in enumerate(pairs):
            r = p[f"{lv}:{b}_vs_{base}"]
            col = C[2] if r["verdict"] == "helps" else (C[1] if r["verdict"] == "hurts" else INK2)
            bx.plot([min(r["per_seed_points"]), max(r["per_seed_points"])], [y, y], color=GRID, lw=3,
                    solid_capstyle="round", zorder=2)
            bx.scatter(r["per_seed_points"], [y] * 3, s=7, color=INK2, zorder=3)
            bx.scatter(r["delta_points"], y, s=30, marker="D", color=col, edgecolor="white", zorder=4)
            bx.text(4.3, y, f"p={r['p']:.2g}", va="center", fontsize=6.4, color=INK2)
            yt.append(y)
            yl.append(f"{lvname}: {lab}")
            y += 1
        y += 0.5
    bx.axvline(0, color=INK2, lw=0.6)
    bx.set_yticks(yt, yl, fontsize=6.8)
    bx.invert_yaxis()
    bx.set_xlim(-4, 6)
    bx.set_xlabel("Change in top-1 from adding the GAN loss (points)")
    bx.set_title("(b) Each GAN against its own no-GAN control", fontsize=7.6, loc="left")
    bx.scatter([], [], s=30, marker="D", color=C[2], label="meets 'helps' rule")
    bx.scatter([], [], s=30, marker="D", color=INK2, label="no reliable difference")
    bx.scatter([], [], s=7, color=INK2, label="individual seeds")
    bx.legend(fontsize=6.4, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=3, columnspacing=0.8)
    fig.tight_layout(h_pad=1.0)
    save(fig, "fig_gan")


def fig_perclass(a: dict) -> None:
    hits = defaultdict(list)
    for f in a["landmark_qat_int4"]["files"]:
        d = json.loads((ROOT / f).read_text(encoding="utf-8"))
        for path, h in zip(d["video_paths"], d["hits_masked_tta"]):
            hits[path.split("/")[1]].append(h)
    words = sorted(hits, key=lambda w: (np.mean(hits[w]), w))
    recall = [100 * np.mean(hits[w]) for w in words]
    fig, ax = plt.subplots(figsize=(5.15, 2.3))
    ax.bar(range(len(words)), recall, color=C[0], width=0.75, edgecolor="white", linewidth=0.5)
    for t in (50, 90):
        ax.axhline(t, color=INK2, lw=0.6, ls=":")
    ax.set_xticks(range(len(words)), [w.split(". ", 1)[-1] for w in words], rotation=90, fontsize=5.6)
    ax.set_ylabel("Recall (%)")
    ax.set_xlim(-0.7, len(words) - 0.3)
    ax.set_ylim(0, 102)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig_perclass")
    return {w: r for w, r in zip(words, recall)}


def main() -> None:
    a = json.loads(DATA.read_text(encoding="utf-8"))
    fig_variants(a)
    fig_protocol(a)
    fig_ablation(a)
    fig_gan(a)
    rec = fig_perclass(a)
    low = [(w, round(r, 1)) for w, r in rec.items() if r < 50]
    print("per-class recall < 50 %:", low)
    print("words >= 90 %:", sum(r >= 90 for r in rec.values()), "of", len(rec))


if __name__ == "__main__":
    main()
