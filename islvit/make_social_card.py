"""Generate the GitHub social preview card (1280x640)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG = "#0d1117"
FG = "#e6edf3"
MUTED = "#8b949e"
ACCENT = "#2ea043"
WARN = "#f78166"
LINE = "#30363d"

fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
fig.patch.set_facecolor(BG)

# ── Left column: identity + edge budget ───────────────────────────────────────
fig.text(0.055, 0.845, "ISL-ViT-Tiny", color=FG, fontsize=46, fontweight="bold",
         va="top", ha="left")
fig.text(0.055, 0.712, "Compact factorised Vision Transformer for",
         color=MUTED, fontsize=17.5, va="top", ha="left")
fig.text(0.055, 0.648, "Indian Sign Language recognition",
         color=MUTED, fontsize=17.5, va="top", ha="left")

fig.text(0.055, 0.545, "S I Z E D   F O R   S M A R T   G L A S S E S", color=ACCENT, fontsize=11.5,
         fontweight="bold", va="top", ha="left")

stats = [("3.80 M", "parameters"), ("0.824", "GMACs / clip"),
         ("12 ms", "CPU latency"), ("3.8 MB", "INT8")]
x0, dx = 0.055, 0.108
for i, (val, lab) in enumerate(stats):
    fig.text(x0 + i * dx, 0.435, val, color=FG, fontsize=21,
             fontweight="bold", va="top", ha="left")
    fig.text(x0 + i * dx, 0.352, lab, color=MUTED, fontsize=11.5,
             va="top", ha="left")

fig.text(0.055, 0.20, "68.9 % top-1  ·  89.4 % top-5  ·  80.7 % at 76 % coverage",
         color=FG, fontsize=15, va="top", ha="left")
fig.text(0.055, 0.125, "INCLUDE-50, session-disjoint — the leakage-free protocol",
         color=MUTED, fontsize=12.5, va="top", ha="left")

# ── Divider ───────────────────────────────────────────────────────────────────
fig.add_artist(plt.Line2D([0.565, 0.565], [0.12, 0.85], color=LINE, lw=1.4))

# ── Right column: the leakage collapse ───────────────────────────────────────
ax = fig.add_axes([0.635, 0.235, 0.315, 0.50])
ax.set_facecolor(BG)

labels = ["random\nsplit", "take-\ngroup", "session-\ndisjoint"]
values = [94.5, 29.3, 22.1]
colors = [WARN, "#d29922", ACCENT]

bars = ax.bar(range(3), values, color=colors, width=0.58)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 3.5, f"{v:.1f}%",
            ha="center", va="bottom", color=FG, fontsize=15, fontweight="bold")

ax.set_ylim(0, 118)
ax.set_xticks(range(3))
ax.set_xticklabels(labels, color=MUTED, fontsize=12)
ax.set_yticks([])
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(LINE)
ax.tick_params(axis="x", length=0, pad=8)

fig.text(0.7925, 0.815, "What the benchmark actually measures",
         color=FG, fontsize=13.5, fontweight="bold", ha="center", va="top")
fig.text(0.7925, 0.153,
         "INCLUDE-263 — 65 pts of the published\nnumber is near-duplicate takes",
         color=MUTED, fontsize=11.5, ha="center", va="top", linespacing=1.5)

out = Path("docs/figures/social_preview.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, facecolor=BG, dpi=100)
print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
