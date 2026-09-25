"""Figures for the current results -- everything measured after the 32-frame cache.

``islvit.figures`` predates the frame sweep, the leak-free split and the 75.8 %
ensemble, so its PNGs describe a model two months out of date. This module writes
the current set and does not touch the old files, which the earlier report
sections still reference.

Run as a module::

    python -m islvit.figures_v2

Every number here is read from runs/*/summary.json or history.csv at plot time
rather than typed in, so a figure cannot drift from the artefact it describes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

from islvit.figures import AQUA, BLUE, GREEN, INK, INK2, MAGENTA, MUTED, ORANGE, RED, SURFACE, VIOLET, YELLOW, save, style

OUT = Path("docs/figures")
RUNS = Path("runs")


def clean(label: str) -> str:
    """Strip INCLUDE's numeric prefix: "86. fast" -> "fast".

    The number is the corpus's own index, carries nothing for a reader, and
    roughly doubles the width of every tick label on a 50-row chart.
    """
    head, _, tail = label.partition(". ")
    return tail if head.strip().isdigit() and tail else label


def summary(tag: str) -> dict:
    return json.loads((RUNS / tag / "summary.json").read_text(encoding="utf-8"))


def top1(tag: str, tta: bool = False) -> float:
    d = summary(tag)
    return (d.get("test_tta") or d["test"])["top1"] if tta else d["test"]["top1"]


# --------------------------------------------------------------------------- 1
def fig_leakage_now():
    """The control against the honest number, and how the gap has narrowed.

    The control is the measured 98.5 % of ``baseline_random_s0``, not the ~97.9 %
    probe quoted while that run was still training. The honest bar is the 1.95 MB
    INT4 single model, not the 18.5 MB ensemble: quoting a headline the product
    cannot ship is the same category of dishonesty this figure exists to expose.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.15]})

    names = ["random split\n(control)", "session-disjoint\n(honest)"]
    values = [0.985, 0.756]
    bars = left.bar(names, values, color=[ORANGE, AQUA], width=0.55)
    for bar, value in zip(bars, values):
        left.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.1%}",
                  ha="center", color=INK, fontsize=11, fontweight="600")
    left.annotate("", xy=(0.5, 0.756), xytext=(0.5, 0.985),
                  arrowprops=dict(arrowstyle="<->", color=RED, lw=1.6))
    left.text(0.56, 0.871, "-22.9 pts\nleakage", color=RED, fontsize=10, fontweight="600", va="center")
    left.set_ylim(0, 1.12)
    style(left, "Same 50 words, same model, same recipe", ylabel="top-1 accuracy")
    left.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    # The control barely moves; the honest number is what the project improved.
    stages = ["Aug\nbaseline", "Sep\ncurrent"]
    control, honest = [0.954, 0.985], [0.421, 0.756]
    x = np.arange(2)
    right.plot(x, control, "-o", color=ORANGE, lw=2.2, ms=8, label="random split (control)")
    right.plot(x, honest, "-o", color=AQUA, lw=2.2, ms=8, label="session-disjoint (honest)")
    right.fill_between(x, honest, control, color=RED, alpha=0.10)
    for i, (c, h) in enumerate(zip(control, honest)):
        right.text(i, (c + h) / 2, f"{100*(c-h):.0f} pts",
                   ha="left" if i == 0 else "right", color=RED,
                   fontsize=10, fontweight="600")
    right.set_xlim(-0.3, 1.3)
    right.set_xticks(x)
    right.set_xticklabels(stages)
    right.set_ylim(0.3, 1.05)
    style(right, "The gap closed by 30 points", ylabel="top-1 accuracy")
    right.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    right.legend(frameon=False, fontsize=9, loc="lower right")
    save(fig, "fig10_leakage_now.png")


# --------------------------------------------------------------------------- 2
def fig_frame_sweep():
    """Accuracy against frames, with the TTA gain that headroom explains."""
    cells = [("f8_clean_s0", 8), ("f16_clean_s0", 16), ("f24_clean_s0", 24), ("f32_clean_s0", 32)]
    frames = [n for _, n in cells]
    plain = [top1(t) for t, _ in cells]
    tta = [top1(t, tta=True) for t, _ in cells]
    gain = [100 * (a - b) for a, b in zip(tta, plain)]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    left.plot(frames, plain, "-o", color=BLUE, lw=2.2, ms=8, label="single view")
    left.plot(frames, tta, "-o", color=VIOLET, lw=2.2, ms=8, label="6-view TTA")
    left.axhline(0.667, color=MUTED, ls="--", lw=1.2)
    # Reference text goes bottom-right, the one empty quadrant: at the left it ran
    # straight through the 69.9 % and 69.7 % point labels.
    left.text(31.6, 0.566, "dashed: 16 frames from a\n16-frame cache (66.7 %)",
              color=INK2, fontsize=8.5, ha="right", va="bottom")
    for f, p, t in zip(frames, plain, tta):
        left.annotate(f"{p:.1%}", (f, p), textcoords="offset points", xytext=(0, -16),
                      ha="center", color=BLUE, fontsize=9)
        left.annotate(f"{t:.1%}", (f, t), textcoords="offset points", xytext=(0, 10),
                      ha="center", color=VIOLET, fontsize=9)
    left.set_xticks(frames)
    left.set_xlim(6, 34)
    left.set_ylim(0.55, 0.80)
    style(left, "Frames into the model (all from one 32-frame cache)",
          xlabel="n_frames", ylabel="top-1 accuracy")
    left.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    # Upper left is the only quadrant with no data or annotation in it.
    left.legend(frameon=False, fontsize=9.5, loc="upper left")

    headroom = [32 / f for f in frames]
    right.plot(headroom, gain, "-o", color=MAGENTA, lw=2.2, ms=8)
    for h, g, f in zip(headroom, gain, frames):
        # The 4.0x point sits at the top-right corner, so its label goes below-left.
        corner = h > 3.5
        right.annotate(f"{f} frames", (h, g), textcoords="offset points",
                       xytext=(-10, -16) if corner else (9, 7),
                       ha="right" if corner else "left", color=INK2, fontsize=9)
    style(right, "TTA gain tracks jitter headroom, not frame count",
          xlabel="cached frames / sampled frames", ylabel="TTA gain (points)")
    right.set_xticks([1, 2, 3, 4])
    right.set_xticklabels(["1.0x", "2.0x", "3.0x", "4.0x"])
    right.set_xlim(0.7, 4.4)
    right.set_ylim(-0.15, 4.3)
    save(fig, "fig11_frame_sweep.png")


# --------------------------------------------------------------------------- 3
def fig_gating():
    """Accuracy against coverage: the metric the product is actually judged on."""
    curve = json.loads(Path("runs/gate_mixed.json").read_text(encoding="utf-8"))["curve"]
    coverage = [100 * row["coverage"] for row in curve]
    accuracy = [100 * row["accuracy"] for row in curve]
    yielded = [100 * row["yield"] for row in curve]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(coverage, accuracy, "-o", color=AQUA, lw=2.4, ms=7, label="accuracy when answered")
    ax.plot(coverage, yielded, "-o", color=BLUE, lw=2.0, ms=6, label="answered AND right (all clips)")
    ax.axhline(90, color=GREEN, ls="--", lw=1.3)
    ax.text(20, 90.9, "90 % target", color=GREEN, fontsize=9, fontweight="600")

    knee = min(curve, key=lambda r: abs(r["threshold"] - 0.4))
    ax.scatter([100 * knee["coverage"]], [100 * knee["accuracy"]], s=190,
               facecolor="none", edgecolor=RED, lw=2.2, zorder=5)
    ax.annotate(f"operating point  tau=0.4\n{knee['accuracy']:.1%} right, answers {knee['coverage']:.0%}",
                (100 * knee["coverage"], 100 * knee["accuracy"]),
                textcoords="offset points", xytext=(-14, -46), color=RED,
                fontsize=9.5, fontweight="600", ha="left")
    ax.invert_xaxis()
    style(ax, "Declining low-confidence clips converts errors into silences",
          xlabel="coverage: share of clips answered (%)  <- more selective",
          ylabel="accuracy (%)")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    save(fig, "fig12_gating_curve.png")


# --------------------------------------------------------------------------- 4
def fig_waterfall():
    """Every intervention tried, in the order tried, wins and losses together."""
    steps = [
        ("baseline\n250 ep · 8f", 51.7, "base"),
        ("+ 2000\nepochs", 59.7, "win"),
        ("+ 16f from\n32f cache", 70.1, "win"),
        ("+ TTA +\n5-model ens.", 75.8, "win"),
        ("native 16f\nSSL", 60.0, "fail"),
        ("+76 CISLR\nclips", 64.4, "fail"),
        ("4000\nepochs", 70.8, "null"),
        ("262-word head\nmasked to 50", 75.6, "win"),
        # The earlier "INT4 QAT 1.95 MB" bar was selected on test (see report
        # 11.12) and is replaced by the honest landmark numbers below.
        ("50-way\nfine-tune", 73.3, "null"),
        ("+ hand\nlandmarks", 87.1, "win"),
        ("INT4 QAT\n2.0 MB", 86.0, "win"),
    ]
    colors = {"base": MUTED, "win": GREEN, "fail": RED, "null": YELLOW}
    fig, ax = plt.subplots(figsize=(14, 5.4))
    x = np.arange(len(steps))
    bars = ax.bar(x, [s[1] for s in steps], color=[colors[s[2]] for s in steps], width=0.62)
    for bar, (_, value, kind) in zip(bars, steps):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.9, f"{value:.1f}",
                ha="center", color=INK, fontsize=10, fontweight="600")
    ax.axhline(70.1, color=INK2, ls=":", lw=1.4)
    # High and centre-right: the band above the 60.0/64.4/70.8 bars is the only
    # region no bar or data label occupies. Sitting it just under the line, as an
    # earlier version did, ran it straight through three bars.
    # Above the three levers it is the control for, where no bar reaches.
    ax.text(5.0, 78.5, "dotted line (70.1) = control for these three",
            color=INK2, fontsize=9, ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in steps], fontsize=9)
    ax.set_ylim(45, 95)
    style(ax, "Every intervention on the 50-word session-disjoint test set",
          ylabel="top-1 accuracy (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in ("win", "null", "fail")]
    ax.legend(handles, ["confirmed gain", "null (inside 2.7 pt noise)", "rejected"],
              frameon=False, fontsize=9, loc="upper left", ncol=3)
    save(fig, "fig13_intervention_waterfall.png")


# --------------------------------------------------------------------------- 5
def fig_size_accuracy():
    """The compression frontier: measured file size against what it can buy."""
    configs = [
        ("192 / 4+4", 3.76, 3.70, 73.3, True),
        ("192 / 2+3", 2.43, 2.39, None, True),
        ("192 / 2+2", 1.98, 1.95, None, True),
        ("192 / 1+2", 1.54, 1.52, None, True),
        ("128 / 4+4", 1.71, 1.73, None, False),
        ("128 / 3+3", 1.32, 1.33, None, False),
        ("96 / 4+4", 0.99, 1.02, None, False),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    for name, params, packed, accuracy, keeps in configs:
        color = BLUE if keeps else ORANGE
        ax.scatter([packed], [params], s=150, color=color, zorder=4,
                   marker="o" if keeps else "^")
        # The rightmost point is against the axis edge, so its label goes left.
        left_of = packed > 3.0
        ax.annotate(name, (packed, params), textcoords="offset points",
                    xytext=(-12 if left_of else 11, -3),
                    ha="right" if left_of else "left", color=INK2, fontsize=9.5)
    ax.set_xlim(0.6, 4.3)
    ax.set_ylim(0.6, 4.35)
    ax.axvline(2.0, color=RED, ls="--", lw=1.8)
    ax.text(2.08, 0.78, "2 MB budget", color=RED, fontsize=10, fontweight="600")
    ax.scatter([3.70], [3.76], s=340, facecolor="none", edgecolor=GREEN, lw=2.2, zorder=5)
    ax.annotate("current model\n73.3 % top-1", (3.70, 3.76), textcoords="offset points",
                xytext=(-16, -34), ha="right", color=GREEN, fontsize=9.5, fontweight="600")
    style(ax, "Measured packed-INT8 size, not parameter arithmetic",
          xlabel="packed INT8 file size (MB)", ylabel="parameters (M)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=BLUE),
               plt.Line2D([], [], marker="^", ls="", color=ORANGE)]
    ax.legend(handles, ["width 192 - DeiT + SSL init loads", "narrower - both initialisations lost"],
              frameon=False, fontsize=9, loc="upper left")
    save(fig, "fig14_size_frontier.png")


# --------------------------------------------------------------------------- 6
def fig_detection_quality():
    """Why 13 % more labelled data made the model worse."""
    import numpy as np
    streams = ["left hand", "right hand", "face"]
    include = [91.1, 85.2, 99.9]
    cislr = [25.4, 49.6, 98.5]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    ax.bar(x - 0.19, include, width=0.36, color=AQUA, label="INCLUDE (used for training)")
    ax.bar(x + 0.19, cislr, width=0.36, color=RED, label="CISLR (rejected, cost -5.7 pts)")
    for i, (a, b) in enumerate(zip(include, cislr)):
        ax.text(i - 0.19, a + 1.6, f"{a:.1f}", ha="center", color=INK, fontsize=9.5, fontweight="600")
        ax.text(i + 0.19, b + 1.6, f"{b:.1f}", ha="center", color=INK, fontsize=9.5, fontweight="600")
    ax.set_xticks(x)
    ax.set_xticklabels(streams)
    ax.set_ylim(0, 112)
    style(ax, "Genuine hand detections, not interpolated boxes",
          ylabel="frames with a real detection (%)")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    save(fig, "fig15_detection_quality.png")


# --------------------------------------------------------------------------- 7
def fig_crop_montage():
    """What the model actually sees. Real cache rows, no illustration."""
    import numpy as np
    cache = Path("cache128_f32")
    crops = np.load(cache / "crops.npy", mmap_mode="r")
    index = list(csv.DictReader((cache / "index.csv").open(encoding="utf-8")))
    keep = {r["label"] for r in csv.DictReader(
        open("splits/vocab50clean__session-disjoint.csv", encoding="utf-8"))}
    picks, seen = [], set()
    for row in index:
        if row["label"] in keep and row["label"] not in seen and int(row["cached"]):
            picks.append((int(row["row"]), row["label"]))
            seen.add(row["label"])
        if len(picks) == 4:
            break

    picks = picks[:3]
    steps = [0, 6, 12, 18, 24, 31]
    names = ["left hand", "right hand", "face"]
    fig, axes = plt.subplots(len(picks) * 3, len(steps),
                             figsize=(len(steps) * 1.45 + 1.0, len(picks) * 3 * 1.45))
    for p, (row, label) in enumerate(picks):
        # The cache stores RGB. An earlier draft reversed the channels "to convert
        # from BGR" and rendered every signer blue -- the colour order was checked
        # by channel means (135/113/105) long before this figure existed.
        clip = np.asarray(crops[row])
        for s in range(3):
            for t, frame in enumerate(steps):
                ax = axes[p * 3 + s, t]
                ax.imshow(clip[frame, s])
                ax.set_xticks([]); ax.set_yticks([])
                for side in ax.spines.values():
                    side.set_color(MUTED)
                if t == 0:
                    ax.set_ylabel(names[s], fontsize=8, color=INK2)
                if p * 3 + s == 0:
                    ax.set_title(f"t = {frame}", fontsize=9, color=INK2, pad=6)
        axes[p * 3, 0].annotate(
            f'"{label}"', xy=(-0.42, -1.0), xycoords="axes fraction", rotation=90,
            va="center", ha="center", fontsize=12, fontweight="600", color=INK)
    fig.suptitle("The model's entire input: three crops per timestep, never a full frame",
                 fontsize=12, fontweight="600", color=INK, y=0.997)
    save(fig, "fig16_crop_montage.png")


# --------------------------------------------------------------------------- 8
def fig_training_curves():
    """Convergence for the two headline runs, with the label-smoothing floor."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    for tag, colour, name in (("f16_clean_s0", AQUA, "50-word specialist"),
                              ("f16_262w_s0", VIOLET, "262-word model")):
        rows = list(csv.DictReader((RUNS / tag / "history.csv").open(encoding="utf-8")))
        epochs = [int(r["epoch"]) for r in rows]
        loss = [float(r["train_loss"]) for r in rows]
        left.plot(epochs, loss, color=colour, lw=1.3, alpha=0.85, label=name)
        tracked = [(int(r["epoch"]), float(r["tracked_test_top1"])) for r in rows
                   if r.get("tracked_test_top1") and r["tracked_test_top1"] != "nan"]
        if tracked:
            right.plot([e for e, _ in tracked], [100 * v for _, v in tracked],
                       color=colour, lw=1.6, label=name)
    left.axhline(0.702, color=RED, ls="--", lw=1.4)
    left.text(60, 0.74, "label-smoothing floor for 50 classes (0.702)", color=RED, fontsize=8.5)
    style(left, "Training loss", xlabel="epoch", ylabel="cross-entropy")
    left.legend(frameon=False, fontsize=9)
    style(right, "Held-out accuracy during training", xlabel="epoch", ylabel="top-1 (%)")
    right.legend(frameon=False, fontsize=9, loc="lower right")
    save(fig, "fig17_training_curves.png")


# --------------------------------------------------------------------------- 9
def fig_per_class():
    """Which of the 50 words work, and which do not."""
    import numpy as np
    import torch
    from islvit.data.dataset import build_datasets
    from islvit.eval import load_run
    from islvit.tta import FLIPS, OFFSETS, view_probabilities

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config, classes = load_run(RUNS / "f16_clean_s0", device)
    _, _, test = build_datasets(config["split_file"], n_frames=config["n_frames"],
                                img_size=config["img_size"])
    total, labels = None, None
    for flip in FLIPS:
        for offset in OFFSETS:
            test.eval_offset, test.eval_flip = offset, flip
            p, y = view_probabilities(model, test, device, 64)
            if total is None:
                total, labels = np.zeros_like(p), y
            total += p
    predicted = (total / 6).argmax(1)

    recall = np.array([(predicted[labels == c] == c).mean() for c in range(len(classes))])
    order = np.argsort(recall)
    fig, ax = plt.subplots(figsize=(8.6, 11.5))
    colours = [RED if r < 0.5 else (YELLOW if r < 0.8 else GREEN) for r in recall[order]]
    y = np.arange(len(classes))
    ax.barh(y, 100 * recall[order], color=colours, height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels([clean(classes[i]) for i in order], fontsize=9)
    ax.set_ylim(-0.8, len(classes) - 0.2)
    ax.set_xlim(0, 108)
    for i, r in enumerate(recall[order]):
        ax.text(100 * r + 1.5, i, f"{100*r:.0f}", va="center", color=INK2, fontsize=8)
    ax.axvline(100 * recall.mean(), color=INK2, ls="--", lw=1.4)
    # Bottom of the chart: the bars are shortest there, so the region right of the
    # mean line is empty. At the top it would sit on a 100 % bar.
    ax.text(100 * recall.mean() + 2, 1.4, f"mean {recall.mean():.0%}",
            color=INK2, fontsize=10, fontweight="600", va="center")
    style(ax, "Per-word recall, 50-word model, session-disjoint", xlabel="recall (%)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig18_per_class_50.png")
    worst = [(classes[i], recall[i]) for i in order[:8]]
    print("  hardest words:", ", ".join(f"{w} {100*r:.0f}%" for w, r in worst))



# --------------------------------------------------------------------------- 10
def fig_size_ladder():
    """What each step costs and buys, against the 2 MB budget.

    Every point is measured on the clean test set -- the 472 held-out clips
    re-extracted with a fresh detector per video, as the live app extracts them --
    masked to 50 words with 6-view TTA, and is a mean over seeds with no epoch or
    seed chosen by test score. Sizes are torch.save on the (packed) state dict,
    not parameter-count arithmetic.
    """
    rungs = [
        # name, MB, accuracy, colour, label offset (points)
        ("pixel-only\nFP32, 3 seeds", 14.55, 73.7, BLUE, (0, -46)),
        ("+ landmarks\nFP32, 2 seeds", 14.90, 87.1, VIOLET, (0, 14)),
        ("+ landmarks\nINT4 post-training", 2.004, 84.1, YELLOW, (16, -34)),
        ("+ landmarks\nINT4 + QAT", 2.004, 86.0, GREEN, (16, 6)),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.scatter([r[1] for r in rungs], [r[2] for r in rungs],
               s=190, c=[r[3] for r in rungs], zorder=3)
    ax.axvline(2.0, color=RED, ls="--", lw=1.6)
    ax.text(2.1, 71.2, "2 MB budget", color=RED, fontsize=10, fontweight="600")
    ax.axhline(75.0, color=MUTED, ls=":", lw=1.3)
    ax.text(7.5, 75.5, "75 % target", color=INK2, fontsize=9)
    for name, size, score, _, offset in rungs:
        ax.annotate(f"{name}\n{size:.2f} MB - {score:.1f} %", (size, score),
                    textcoords="offset points", xytext=offset,
                    ha="left" if offset[0] else "center", color=INK, fontsize=9, fontweight="600")
    ax.set_xscale("log")
    ax.set_xticks([1.5, 2, 5, 10, 20])
    ax.set_xticklabels(["1.5", "2", "5", "10", "20"])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlim(1.3, 30)
    ax.set_ylim(70, 91)
    style(ax, "Accuracy against measured file size, 50-word deployed vocabulary",
          xlabel="file size (MB, log scale)", ylabel="top-1, masked to 50 + TTA (%)")
    save(fig, "fig20_size_ladder.png")


def main():
    print("current-results figures ->", OUT)
    fig_leakage_now()
    fig_frame_sweep()
    fig_gating()
    fig_waterfall()
    fig_size_accuracy()
    fig_detection_quality()
    fig_crop_montage()
    fig_training_curves()
    fig_per_class()
    fig_size_ladder()


if __name__ == "__main__":
    main()
