"""Generate every figure and derived statistic used in the technical report.

Run as a module::

    python -m islvit.figures

Writes PNGs to docs/figures/ and dumps the numeric tables it computed to stdout so
the report text can quote exact values rather than approximations.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("docs/figures")
RUNS = Path("runs")

# Validated categorical slots (see dataviz reference palette; checked with
# validate_palette.js -- all gates pass, contrast WARN relieved by the tables
# that accompany every figure in the report).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#4a3aa7", "#e34948"
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#b8b7b0", "#fcfcfb"

PROTOCOL_COLOR = {
    "random-video": ORANGE,
    "official": YELLOW,
    "take-group": BLUE,
    "session-disjoint": AQUA,
}


def style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=INK2, labelsize=9, length=3, width=1)
    ax.grid(True, color=MUTED, linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=12, fontweight="600", pad=12, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=10)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {OUT / name}")


def load_history(tag):
    path = RUNS / tag / "history.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return {
        "epoch": np.array([int(r["epoch"]) for r in rows]),
        "loss": np.array([float(r["train_loss"]) for r in rows]),
        "val": np.array([float(r["val_top1"]) for r in rows]),
        "val5": np.array([float(r["val_top5"]) for r in rows]),
        "ema": np.array([float(r["ema_top1"]) for r in rows]),
    }


def load_summary(tag):
    path = RUNS / tag / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------------- #
# Figure 1: the leakage evidence -- within-class MVI gap histogram
# --------------------------------------------------------------------------- #
def fig_gap_histogram():
    rows = list(csv.DictReader(open("splits/full263__take-group.csv", encoding="utf-8")))
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(int(re.search(r"MVI_(\d+)", row["video_path"]).group(1)))

    gaps = Counter()
    for ids in by_label.values():
        ids.sort()
        for a, b in zip(ids, ids[1:]):
            gaps[min(b - a, 45)] += 1

    xs = np.arange(1, 46)
    ys = np.array([gaps.get(int(x), 0) for x in xs])

    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = [RED if x <= 2 else (BLUE if x >= 20 else MUTED) for x in xs]
    ax.bar(xs, ys, color=colors, width=0.82)
    style(
        ax,
        "Within-class gaps between consecutive MVI ids",
        "gap between consecutive MVI ids (same class), 45 = 45 or more",
        "number of gaps",
    )
    ax.set_yscale("symlog", linthresh=10)
    ax.annotate(
        f"gap 1-2: {ys[0] + ys[1]:,} pairs\nback-to-back takes\n(near-duplicates)",
        xy=(1.5, ys[0]), xytext=(6, 900), color=INK, fontsize=9,
        arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2),
    )
    ax.annotate(
        "gap >= 20: a different\nrecording session",
        xy=(27, max(ys[19:]) if len(ys) > 19 else 10), xytext=(31, 180), color=INK, fontsize=9,
        arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2),
    )
    save(fig, "fig01_mvi_gap_histogram.png")
    print(f"    gap<=2: {ys[0] + ys[1]}  gap 3-19: {ys[2:19].sum()}  gap>=20: {ys[19:].sum()}")


# --------------------------------------------------------------------------- #
# Figure 2: leakage ladder
# --------------------------------------------------------------------------- #
LADDER = {
    "include50": [
        ("random-video", "include50_v2__random-video"),
        ("official", "include50_v2__official"),
        ("take-group", "abl64__take-group"),
        # sd_include50 is gone: it was trained against a session-disjoint split
        # whose blocks were clustered from the 943-clip INCLUDE-50 subset rather
        # than the whole corpus, so train and test shared 10 of 13 sessions.
        # frames8__sd is the same recipe on the corrected split.
        ("session-disjoint", "frames8__sd"),
    ],
    "full263": [
        ("random-video", "full263__random-video"),
        ("take-group", "full263_v2__take-group"),
        ("session-disjoint", "sd_full263"),
    ],
}


def fig_leakage_ladder():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (bench, entries) in zip(axes, LADDER.items()):
        names, values = [], []
        for scheme, tag in entries:
            summary = load_summary(tag)
            if summary:
                names.append(scheme)
                values.append(summary["test"]["top1"] * 100)
        bars = ax.bar(names, values, color=[PROTOCOL_COLOR[n] for n in names], width=0.62)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.1f}%",
                    ha="center", color=INK, fontsize=10, fontweight="600")
        style(ax, f"{bench}: test top-1 by split protocol", "", "test top-1 (%)")
        ax.set_ylim(0, 108)
        ax.tick_params(axis="x", rotation=18)
    save(fig, "fig02_leakage_ladder.png")


# --------------------------------------------------------------------------- #
# Figures 3-4: convergence
# --------------------------------------------------------------------------- #
def fig_convergence(bench, entries, fname, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for scheme, tag in entries:
        history = load_history(tag)
        if history is None:
            continue
        color = PROTOCOL_COLOR[scheme]
        ax1.plot(history["epoch"], history["loss"], color=color, lw=2, label=scheme)
        ax2.plot(history["epoch"], history["val"] * 100, color=color, lw=2, label=scheme)
    style(ax1, f"{title} - training loss", "epoch", "cross-entropy (label-smoothed)")
    style(ax2, f"{title} - validation top-1", "epoch", "val top-1 (%)")
    for ax in (ax1, ax2):
        legend = ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
        legend.set_title("")
    save(fig, fname)


# --------------------------------------------------------------------------- #
# Figure 5: resolution ablation
# --------------------------------------------------------------------------- #
def fig_resolution():
    points = [("64px / 16 tok", 0.824, "abl64__take-group", BLUE),
              ("112px / 49 tok", 2.342, "abl112__take-group", ORANGE),
              ("128px / 64 tok", 3.010, "abl128__take-group", AQUA)]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for label, gmacs, tag, color in points:
        summary = load_summary(tag)
        if not summary:
            continue
        top1 = summary["test"]["top1"] * 100
        ax.scatter([gmacs], [top1], s=190, color=color, zorder=3, edgecolor=SURFACE, linewidth=2)
        ax.annotate(f"  {label}\n  {top1:.1f}%", (gmacs, top1), color=INK, fontsize=9, va="center")
    style(ax, "Accuracy does not buy anything from extra tokens",
          "compute (GMACs per clip)", "take-group test top-1 (%)")
    ax.set_xlim(0, 3.9)
    ax.set_ylim(40, 50)
    save(fig, "fig05_resolution_ablation.png")


# --------------------------------------------------------------------------- #
# Figure 6: crop box provenance
# --------------------------------------------------------------------------- #
def fig_box_sources():
    sources = np.load("cache128/sources.npy", mmap_mode="r")
    done = np.load("cache128/done.npy")
    data = np.asarray(sources[np.where(done)[0]])
    streams = ["left hand", "right hand", "face"]
    labels = ["direct (full-frame)", "ROI rescue (2-stage)", "interpolated", "missing"]
    colors = [BLUE, AQUA, YELLOW, RED]

    fractions = np.array([[(data[:, :, s] == code).mean() * 100 for code in (1, 2, 3, 0)]
                          for s in range(3)])
    fig, ax = plt.subplots(figsize=(9, 3.4))
    left = np.zeros(3)
    for index, (label, color) in enumerate(zip(labels, colors)):
        values = fractions[:, index]
        ax.barh(streams, values, left=left, color=color, height=0.6, label=label)
        for row, (value, base) in enumerate(zip(values, left)):
            if value > 4:
                ax.text(base + value / 2, row, f"{value:.0f}%", ha="center", va="center",
                        color="white", fontsize=9, fontweight="600")
        left += values
    style(ax, "Where each crop's bounding box came from", "% of all sampled frames", "")
    ax.set_xlim(0, 100)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.18))
    save(fig, "fig06_box_sources.png")
    for stream, row in zip(streams, fractions):
        print(f"    {stream:11s} direct={row[0]:.1f} roi={row[1]:.1f} interp={row[2]:.1f} missing={row[3]:.1f}")


# --------------------------------------------------------------------------- #
# Figure 9: temporal ablation
# --------------------------------------------------------------------------- #
# Plotted as top-1 AND balanced together because on this split they disagree by
# up to 17 points -- showing either alone would imply a confidence the data does
# not support.
TEMPORAL = [(4, 0.414, "frames4__sd"), (8, 0.826, "frames8__sd"),
            (12, 1.238, "frames12__sd"), (16, 1.650, "frames16__sd")]
TEMPORAL_263 = [(4, 0.414, "sd263_f4"), (8, 0.826, "sd_full263"), (16, 1.650, "sd263_f16")]


def fig_temporal():
    """Both splits side by side -- the disagreement IS the finding.

    Plotting only the benchmark that resolved the question would hide that the
    underpowered split ranked T=4 first and the reliable one ranked it last.
    """
    panels = [("INCLUDE-50 SD (n=252, 45 cls)", TEMPORAL),
              ("INCLUDE-263 SD (n=1010, 154 cls)", TEMPORAL_263)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for ax, (title, entries) in zip(axes, panels):
        rows = [(t, g, load_summary(tag)) for t, g, tag in entries]
        rows = [(t, g, s) for t, g, s in rows if s]
        if not rows:
            continue
        frames = [t for t, _, _ in rows]
        for metric, color, label in (("top1", BLUE, "top-1"),
                                     ("balanced", ORANGE, "balanced"),
                                     ("top5", AQUA, "top-5")):
            ax.plot(frames, [s["test"][metric] * 100 for _, _, s in rows],
                    color=color, lw=2, marker="o", ms=7, label=label)
        style(ax, title, "frames sampled per clip", "test accuracy (%)")
        ax.set_xticks(frames)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2)
    save(fig, "fig09_temporal_ablation.png")

    for title, entries in panels:
        values = [load_summary(tag) for _, _, tag in entries]
        values = [v["test"]["balanced"] for v in values if v]
        if values:
            print(f"    temporal {title}: balanced {min(values) * 100:.1f}-{max(values) * 100:.1f}%")


# --------------------------------------------------------------------------- #
# Figure 7: per-class recall for the strict model
# --------------------------------------------------------------------------- #
def fig_per_class(tag, split_file, fname, title):
    import torch
    from torch.utils.data import DataLoader
    from islvit.data.dataset import build_datasets
    from islvit.eval import load_run

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config, classes = load_run(RUNS / tag, device)
    _, _, test_set = build_datasets(split_file, n_frames=config["n_frames"], img_size=config["img_size"])
    loader = DataLoader(test_set, batch_size=128, num_workers=2)

    correct = Counter()
    total = Counter()
    confusions = Counter()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["crops"].to(device), batch["detected"].to(device), batch["geometry"].to(device))
            predicted = logits.argmax(1).cpu()
            for true, pred in zip(batch["label"].tolist(), predicted.tolist()):
                total[true] += 1
                if true == pred:
                    correct[true] += 1
                else:
                    confusions[(classes[true], classes[pred])] += 1

    recalls = sorted((correct[c] / total[c]) * 100 for c in total)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(range(len(recalls)), recalls, color=AQUA, width=1.0)
    mean = float(np.mean(recalls))
    ax.axhline(mean, color=INK2, lw=1.6, ls="--")
    ax.text(1, mean + 3, f"mean per-class recall {mean:.1f}%", color=INK, fontsize=9)
    style(ax, title, "classes, sorted by recall", "per-class recall (%)")
    ax.set_ylim(0, 105)
    save(fig, fname)

    zero = sum(1 for c in total if correct[c] == 0)
    perfect = sum(1 for c in total if correct[c] == total[c])
    print(f"    {tag}: {len(total)} classes | {zero} at 0% | {perfect} at 100% | mean {mean:.1f}%")
    print(f"    top confusions: {confusions.most_common(6)}")
    return recalls


# --------------------------------------------------------------------------- #
def main():
    print("Generating report figures...")
    fig_gap_histogram()
    fig_leakage_ladder()
    fig_convergence("include50", LADDER["include50"], "fig03_convergence_include50.png",
                    "INCLUDE-50")
    fig_convergence("full263", LADDER["full263"], "fig04_convergence_full263.png",
                    "INCLUDE-263")
    fig_resolution()
    fig_box_sources()
    fig_temporal()
    fig_per_class("frames8__sd", "splits/include50__session-disjoint.csv",
                  "fig07_per_class_include50.png",
                  "Per-class recall, INCLUDE-50 session-disjoint")
    fig_per_class("sd_full263", "splits/full263__session-disjoint.csv",
                  "fig08_per_class_full263.png",
                  "Per-class recall, INCLUDE-263 session-disjoint")
    print("done.")


if __name__ == "__main__":
    main()
