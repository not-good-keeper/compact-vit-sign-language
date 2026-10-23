"""Architecture figures for the paper: the network (Fig. A) and the end-to-end pipeline (Fig. B).

Drawn directly with matplotlib so shapes carry meaning: a row of tall bars is a stack of transformer
blocks, a funnel reduces a token sequence to one embedding, a grid is a token sequence, offset squares
are a stack of frames, crops or same-shape tensors that are added together. Flat, muted colours.

    python report/architecture/make_architecture_diagrams.py
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

OUT = Path(__file__).resolve().parents[2] / "docs" / "architecture"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["hatch.linewidth"] = 0.8

INK = "#1F2933"
MUTE = "#5B6670"
PANEL = "#F5F6F8"
PANEL_EC = "#C9CED4"
BLUE, BLUE_L = "#34659A", "#DCE6F2"
ORNG, ORNG_L = "#B8742B", "#F4E6D4"
GRN, GRN_L = "#3E7D55", "#DDEBE1"
GREY, GREY_L = "#6B7580", "#E7EAEE"
PURP, PURP_L = "#6A4C93", "#E6DFF0"
RED, RED_L = "#A94442", "#F3DCDC"
STREAM = [(BLUE, BLUE_L), (ORNG, ORNG_L), (GRN, GRN_L)]


# ----------------------------------------------------------------------------- drawing helpers
def canvas(w, h):
    fig = plt.figure(figsize=(w * 0.5, h * 0.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def text(ax, x, y, s, size=9, weight="normal", color=INK, ha="center", va="center", style="normal", rot=0):
    ax.text(x, y, s, rotation=rot, fontsize=size, fontweight=weight, color=color, ha=ha, va=va,
            fontstyle=style, linespacing=1.25, zorder=10)


def rect(ax, x, y, w, h, fc="white", ec=INK, lw=1.1, ls="-", r=0.12, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                                lw=lw, ls=ls, zorder=z))


def panel(ax, x, y, w, h, title):
    rect(ax, x, y, w, h, fc=PANEL, ec=PANEL_EC, lw=1.0, r=0.2, z=0)
    text(ax, x + 0.3, y + h - 0.38, title, size=9.5, weight="bold", color=MUTE, ha="left")


def arrow(ax, p, q, color=INK, lw=1.2, ls="-", rad=0.0, head=8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=head, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=6))


def line(ax, pts, color=INK, lw=1.2, ls="-", z=5):
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=lw, ls=ls, zorder=z,
            solid_capstyle="butt")


def plus(ax, x, y, r=0.27):
    ax.add_patch(Circle((x, y), r, fc="white", ec=INK, lw=1.2, zorder=7))
    line(ax, [(x - r * 0.6, y), (x + r * 0.6, y)], lw=1.2, z=8)
    line(ax, [(x, y - r * 0.6), (x, y + r * 0.6)], lw=1.2, z=8)


def funnel(ax, x, yc, w, h_in, h_out, fc, ec, lw=1.3):
    ax.add_patch(Polygon([(x, yc - h_in / 2), (x, yc + h_in / 2), (x + w, yc + h_out / 2),
                          (x + w, yc - h_out / 2)], closed=True, fc=fc, ec=ec, lw=lw, zorder=3,
                         joinstyle="round"))


def block_stack(ax, x, yc, n, bw, gap, h, fc, ec):
    """n tall bars, one per transformer block; returns the x where the stack ends."""
    for i in range(n):
        rect(ax, x + i * (bw + gap), yc - h / 2, bw, h, fc=fc, ec=ec, lw=1.2, r=0.08)
    return x + n * bw + (n - 1) * gap


def stack(ax, x, y, w, h, n, off, fc, ec):
    """n offset rectangles, back to front; (x, y) is the front rectangle's lower-left corner."""
    for i in range(n - 1, -1, -1):
        ax.add_patch(Rectangle((x + off * i, y + off * i), w, h, fc=fc, ec=ec, lw=1.0, zorder=3 + (n - i) * 0.01))


def cells(ax, x, y, n, cw, ch, colors, vertical=False, gap=0.04):
    for i in range(n):
        fc, ec = colors[i] if isinstance(colors, list) else colors
        cx, cy = (x, y - i * (ch + gap)) if vertical else (x + i * (cw + gap), y)
        ax.add_patch(Rectangle((cx, cy), cw, ch, fc=fc, ec=ec, lw=0.8, zorder=4))


def cylinder(ax, x, y, w, h, fc, ec):
    e = h * 0.22
    ax.add_patch(Rectangle((x, y + e / 2), w, h - e, fc=fc, ec="none", zorder=3))
    ax.add_patch(Ellipse((x + w / 2, y + e / 2), w, e, fc=fc, ec=ec, lw=1.2, zorder=3))
    ax.add_patch(Ellipse((x + w / 2, y + h - e / 2), w, e, fc=fc, ec=ec, lw=1.2, zorder=4))
    line(ax, [(x, y + e / 2), (x, y + h - e / 2)], color=ec, lw=1.2, z=4)
    line(ax, [(x + w, y + e / 2), (x + w, y + h - e / 2)], color=ec, lw=1.2, z=4)


def bars(ax, x, y, heights, bw=0.1, gap=0.04, colors=None, width_line=True):
    for i, hh in enumerate(heights):
        fc, ec = colors[i] if colors else ("#D9DDE2", GREY)
        ax.add_patch(Rectangle((x + i * (bw + gap), y), bw, hh, fc=fc, ec=ec, lw=0.7, zorder=4))
    if width_line:
        line(ax, [(x - 0.04, y), (x + len(heights) * (bw + gap), y)], lw=0.9)


def hand(ax, cx, cy, s, color):
    """Stylised hand with 21 landmarks: a wrist plus four joints on each of five fingers."""
    pts = [(cx, cy)]
    fingers = [(-0.26, 0.10, -52, 0.46), (-0.12, 0.36, -12, 0.62), (0.0, 0.40, 0, 0.70),
               (0.12, 0.37, 9, 0.64), (0.23, 0.30, 20, 0.50)]
    for bx, by, ang, ln in fingers:
        a = math.radians(ang)
        base = (cx + bx * s, cy + by * s)
        chain = [(base[0] + math.sin(a) * ln * s * f, base[1] + math.cos(a) * ln * s * f) for f in (0, 0.34, 0.68, 1.0)]
        line(ax, [(cx, cy), base], color=color, lw=0.7, z=5)
        line(ax, chain, color=color, lw=0.9, z=5)
        pts += chain
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    ax.scatter(xs, ys, s=5, color=color, zorder=6, linewidths=0)
    return min(xs), min(ys), max(xs), max(ys)


def hull(ax, box, color, margin):
    x0, y0, x1, y1 = box
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec=color, lw=1.2, zorder=6))
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * margin, (y1 - y0) * margin
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc="none", ec=color, lw=0.9, ls=(0, (3, 2)), zorder=6))


CLS = (GREY, INK)
TOK = ("#C7D6EA", BLUE)
SHEET = {"stream": (PURP_L, PURP), "geom": (ORNG_L, ORNG), "land": (BLUE_L, BLUE), "miss": ("white", RED)}


# ----------------------------------------------------------------------------- Figure A
def model_figure():
    fig, ax = canvas(30.0, 15.0)

    panel(ax, 0.2, 5.4, 12.1, 5.9, "Stage A · spatial ViT")
    panel(ax, 12.5, 5.4, 11.6, 5.9, "Stage B · temporal ViT")
    panel(ax, 24.3, 5.4, 5.5, 5.9, "Head")

    # ---- Stage A: crops -> patches -> tokens -> four blocks -> one embedding
    names = ["left hand", "right hand", "face"]
    for k, yc in enumerate([9.4, 8.0, 6.6]):
        ec, fc = STREAM[k]
        stack(ax, 0.6, yc - 0.475, 0.95, 0.95, 4, 0.12, "white", ec)
        text(ax, 2.15, yc, names[k], size=8, color=ec, ha="left")
        arrow(ax, (3.2, yc), (3.9, 8.1 + (yc - 8.1) * 0.4), color=GREY, lw=1.0, head=7)
    text(ax, 1.6, 5.8, "3 streams × 16 frames\n64 × 64 px", size=8, color=MUTE)

    ax.add_patch(Rectangle((3.95, 7.2), 1.8, 1.8, fc="white", ec=INK, lw=1.1, zorder=3))
    for i in range(1, 4):
        line(ax, [(3.95 + 0.45 * i, 7.2), (3.95 + 0.45 * i, 9.0)], color=MUTE, lw=0.7)
        line(ax, [(3.95, 7.2 + 0.45 * i), (5.75, 7.2 + 0.45 * i)], color=MUTE, lw=0.7)
    text(ax, 4.85, 6.85, "conv 16 × 16, stride 16", size=7.5, color=MUTE)
    arrow(ax, (5.85, 8.1), (6.4, 8.1))
    cells(ax, 6.45, 9.7, 17, 0.5, 0.17, [CLS] + [TOK] * 16, vertical=True, gap=0.04)
    text(ax, 6.7, 5.95, "[CLS] + 16\npatch tokens", size=7.5, color=MUTE)
    arrow(ax, (7.05, 8.1), (7.45, 8.1))
    plus(ax, 7.75, 8.1)
    rect(ax, 7.45, 9.05, 0.6, 0.5, fc=GREY_L, ec=GREY, r=0.06)
    text(ax, 7.75, 9.3, "pos", size=7.5)
    arrow(ax, (7.75, 9.05), (7.75, 8.38), head=6, lw=1.0)
    arrow(ax, (8.05, 8.1), (8.45, 8.1))
    end = block_stack(ax, 8.5, 8.1, 4, 0.5, 0.12, 3.5, BLUE_L, BLUE)
    text(ax, 9.65, 5.85, "Spatial ViT × 4", size=9.5, weight="bold", color=BLUE)
    funnel(ax, end + 0.08, 8.1, 0.6, 3.5, 0.6, BLUE_L, BLUE)
    cells(ax, end + 0.9, 7.85, 1, 0.5, 0.5, [CLS])
    text(ax, end + 1.15, 7.4, "z · 192-d", size=7.5, color=MUTE)
    arrow(ax, (end + 1.45, 8.1), (12.95, 8.1))

    # ---- Stage B: the 3 x 16 token grid, the conditioning terms summed into it, four blocks, readout
    mx = 13.55
    rows_y = [8.49, 7.87, 7.25]
    for k in range(3):
        ec, fc = STREAM[k]
        cells(ax, mx, rows_y[k], 1, 0.28, 0.55, [CLS])
        cells(ax, mx + 0.34, rows_y[k], 16, 0.28, 0.55, (fc, ec))
        text(ax, mx - 0.1, rows_y[k] + 0.27, "LRF"[k], size=7.5, color=ec, ha="right")
    for (r_, c_) in [(0, 4), (1, 6), (1, 7), (2, 11), (0, 13)]:   # crops with no detection
        ax.add_patch(Rectangle((mx + 0.34 + c_ * 0.32, rows_y[r_]), 0.28, 0.55, fc="white", ec=RED, lw=0.8,
                               hatch="//////", zorder=5))
    text(ax, mx + 0.15, 9.3, "[CLS]", size=7.5, color=MUTE)
    arrow(ax, (mx + 0.8, 9.3), (mx + 5.4, 9.3), color=MUTE, lw=0.9, head=6)
    text(ax, mx + 3.1, 9.62, "frame 1 … 16", size=7.5, color=MUTE)
    ax.add_patch(Rectangle((mx + 4.0, 6.62), 0.28, 0.2, fc="white", ec=RED, lw=0.8, hatch="//////", zorder=5))
    text(ax, mx + 4.4, 6.72, "missing crop", size=7, color=MUTE, ha="left")

    sx = mx + 0.34 + 16 * 0.32 / 2
    plus(ax, sx, 6.45)
    arrow(ax, (sx, 6.72), (sx, 7.2), head=7)
    text(ax, sx - 0.4, 6.95, "49 × 192", size=7, color=MUTE, ha="right")

    block_x = mx + 0.34 + 16 * 0.32 + 0.55
    arrow(ax, (mx + 0.34 + 16 * 0.32 + 0.1, 8.15), (block_x - 0.05, 8.15))
    end_b = block_stack(ax, block_x, 8.15, 4, 0.5, 0.12, 3.9, PURP_L, PURP)
    text(ax, block_x + 1.15, 5.8, "Temporal ViT × 4", size=9.5, weight="bold", color=PURP)
    funnel(ax, end_b + 0.08, 8.15, 0.6, 3.9, 0.6, PURP_L, PURP)
    cells(ax, end_b + 0.9, 7.9, 1, 0.5, 0.5, [CLS])
    text(ax, end_b + 1.15, 7.45, "[CLS]", size=7.5, color=MUTE)

    # ---- Head
    arrow(ax, (end_b + 1.45, 8.15), (24.75, 8.15))
    rect(ax, 24.8, 7.55, 1.7, 1.2, fc="white", ec=INK, r=0.1)
    text(ax, 25.65, 8.15, "Linear\n192 → 262", size=8.5)
    arrow(ax, (26.55, 8.15), (27.0, 8.15))
    heights = [0.2, 0.3, 0.15, 1.15, 0.28, 0.12, 0.38, 0.22, 0.1, 0.18]
    cols = [(GRN_L, GRN) if i == 3 else ("#D9DDE2", GREY) for i in range(len(heights))]
    bars(ax, 27.05, 7.6, heights, bw=0.13, gap=0.05, colors=cols)
    text(ax, 28.0, 7.15, "softmax\n50 of 262 words", size=7.5, color=MUTE)
    text(ax, 28.0, 9.15, "p(y | x)", size=9.5, weight="bold", color=GRN)

    # ---- conditioning: same-shape layers added to the token grid
    sheets = [("stream + time", "stream"), ("box geometry", "geom"), ("hand + pose", "land"), ("missing flag", "miss")]
    fx, fy, sw, sh_, off = sx - 2.7 - 0.51, 2.6, 5.4, 0.5, 0.34
    for i in range(len(sheets) - 1, -1, -1):
        fc, ec = SHEET[sheets[i][1]]
        ax.add_patch(Rectangle((fx + off * i, fy + off * i), sw, sh_, fc=fc, ec=ec, lw=1.1, zorder=3 + (4 - i) * 0.01,
                               hatch="//////" if sheets[i][1] == "miss" else None))
    for i, (lab, key) in enumerate(sheets):
        yc = fy + (0.25 if i == 0 else off * i + sh_ - off / 2)
        xr = fx + sw + off * i
        line(ax, [(xr + 0.03, yc), (xr + 0.35, yc)], color=SHEET[key][1], lw=0.9)
        text(ax, xr + 0.45, yc, lab, size=7.5, color=SHEET[key][1], ha="left")
    top_x = fx + off * 3 + sw / 2
    arrow(ax, (sx, fy + off * 3 + sh_), (sx, 6.17), head=7)
    text(ax, sx + 0.15, 4.95, "added to every crop token", size=7.5, color=MUTE, ha="left")

    items = [("Stream, frame", "index", "Emb.", "stream"), ("Crop box", "x, y, size", "MLP", "geom"),
             ("Hand landmarks", "2 × 21 × 3", "MLP", "land"), ("Body keypoints", "7 × 3", "MLP", "land"),
             ("Detected", "0 / 1", "Emb.", "miss")]
    w, gap = 1.9, 0.25
    x0 = sx - (len(items) * w + (len(items) - 1) * gap) / 2
    bus_y = 2.3
    for i, (name, shape, kind, key) in enumerate(items):
        x = x0 + i * (w + gap)
        ec = SHEET[key][1]
        rect(ax, x, 0.3, w, 0.85, fc="white", ec=ec, r=0.08)
        text(ax, x + w / 2, 0.93, name, size=7.5, weight="bold")
        text(ax, x + w / 2, 0.55, shape, size=7.5, color=MUTE)
        rect(ax, x + 0.3, 1.45, w - 0.6, 0.5, fc=SHEET[key][0] if key != "miss" else RED_L, ec=ec, r=0.08)
        text(ax, x + w / 2, 1.7, kind, size=8)
        arrow(ax, (x + w / 2, 1.15), (x + w / 2, 1.45), head=6, color=ec)
        line(ax, [(x + w / 2, 1.95), (x + w / 2, bus_y)], lw=1.0, color=ec)
    line(ax, [(x0 + w / 2, bus_y), (x0 + (len(items) - 1) * (w + gap) + w / 2, bus_y)], lw=1.0)
    arrow(ax, (sx, bus_y), (sx, fy), head=6, lw=1.0)

    # ---- parameter budget
    total = 3894214
    parts = [("spatial ViT", 1930944, BLUE_L, BLUE), ("temporal ViT", 1779456, PURP_L, PURP),
             ("landmark MLPs", 91008, ORNG_L, ORNG), ("geometry MLP", 37824, ORNG_L, ORNG),
             ("head", 50566, GRN_L, GRN), ("embeddings, norms", 4416, GREY_L, GREY)]
    bx, bw_total = 0.5, 10.6
    text(ax, bx, 4.55, "3.89 M parameters", size=9.5, weight="bold", color=MUTE, ha="left")
    cur = bx
    for lab, n, fc, ec in parts:
        wd = bw_total * n / total
        ax.add_patch(Rectangle((cur, 3.45), wd, 0.7, fc=fc, ec=ec, lw=1.0, zorder=3))
        if n > 1e6:
            text(ax, cur + wd / 2, 3.8, f"{lab}  {n / 1e6:.2f} M", size=8, color=ec)
        cur += wd
    text(ax, bx, 2.95, "landmark MLPs 0.09 M · geometry MLP 0.04 M · head 0.05 M · embeddings 4 k", size=7.5,
         color=MUTE, ha="left")
    text(ax, bx, 2.45, "4-bit release: groups of 128 weights, one FP16 scale per group · 1.97 MiB", size=7.5,
         color=MUTE, ha="left")
    text(ax, bx, 1.3, "Output layers of the hand, pose and geometry MLPs start at zero, so training begins\n"
                      "as a crops-only model and learns to use the side inputs.",
         size=7.5, color=MUTE, ha="left", style="italic")

    # ---- inside one transformer block
    ix, iy, iw, ih = 9.0, 11.7, 12.9, 3.15
    rect(ax, ix, iy, iw, ih, fc="white", ec=GREY, ls="--", lw=1.0, r=0.2, z=1)
    text(ax, ix + 0.25, iy + ih - 0.3, "Transformer block (pre-norm)", size=8.5, weight="bold", color=MUTE, ha="left")
    text(ax, ix + iw - 0.25, iy + ih - 0.3, "d = 192 · 3 heads of 64 · MLP 768 · GELU · stochastic depth ≤ 0.1",
         size=7.5, color=MUTE, ha="right")
    yc = iy + ih - 1.05
    parts_ = [("LayerNorm", BLUE_L, BLUE), ("MHSA", PURP_L, PURP), None, ("LayerNorm", BLUE_L, BLUE),
              ("MLP", PURP_L, PURP), None]
    pos, cur = [], ix + 0.6
    for p in parts_:
        if p is None:
            plus(ax, cur + 0.27, yc)
            pos.append((cur, cur + 0.54))
            cur += 0.54 + 0.55
        else:
            rect(ax, cur, yc - 0.32, 1.45, 0.64, fc=p[1], ec=p[2], r=0.08)
            text(ax, cur + 0.725, yc, p[0], size=8.5)
            pos.append((cur, cur + 1.45))
            cur += 1.45 + 0.55
    arrow(ax, (ix + 0.1, yc), (pos[0][0], yc), head=6)
    for a, b in zip(pos[:-1], pos[1:]):
        arrow(ax, (a[1], yc), (b[0], yc), head=6)
    arrow(ax, (pos[-1][1], yc), (pos[-1][1] + 0.5, yc), head=6)
    for a, c0 in ((pos[0][0] - 0.25, pos[2][0] + 0.27), (pos[3][0] - 0.25, pos[5][0] + 0.27)):
        top = yc + 0.62
        line(ax, [(a, yc), (a, top), (c0, top)], lw=1.0)
        arrow(ax, (c0, top), (c0, yc + 0.28), head=6, lw=1.0)
    zy = iy + 0.2
    zh = 1.05
    # attention zoom
    zx = ix + 0.3
    rect(ax, zx, zy, 6.6, zh + 0.35, fc="white", ec=PURP, ls=":", lw=0.9, r=0.1)
    text(ax, zx + 0.2, zy + zh + 0.15, "multi-head self-attention", size=7.5, color=PURP, ha="left")
    cy_ = zy + 0.5
    rect(ax, zx + 0.2, cy_ - 0.28, 1.3, 0.56, fc=PURP_L, ec=PURP, r=0.06)
    for i, ch in enumerate("QKV"):
        ax.add_patch(Rectangle((zx + 0.3 + i * 0.4, cy_ - 0.17), 0.34, 0.34, fc="white", ec=PURP, lw=0.8, zorder=4))
        text(ax, zx + 0.47 + i * 0.4, cy_, ch, size=8)
    arrow(ax, (zx + 1.55, cy_), (zx + 1.9, cy_), head=6, lw=1.0)
    rect(ax, zx + 1.95, cy_ - 0.28, 2.45, 0.56, fc="white", ec=PURP, r=0.06)
    text(ax, zx + 3.175, cy_, "softmax(QKᵀ/√64) V  × 3", size=7.5)
    arrow(ax, (zx + 4.45, cy_), (zx + 4.8, cy_), head=6, lw=1.0)
    rect(ax, zx + 4.85, cy_ - 0.28, 0.75, 0.56, fc="white", ec=PURP, r=0.06)
    text(ax, zx + 5.225, cy_, "concat", size=7.5)
    arrow(ax, (zx + 5.65, cy_), (zx + 5.85, cy_), head=5, lw=1.0)
    rect(ax, zx + 5.85, cy_ - 0.28, 0.65, 0.56, fc="white", ec=PURP, r=0.06)
    text(ax, zx + 6.175, cy_, "W_O", size=7.5)
    line(ax, [((pos[1][0] + pos[1][1]) / 2, yc - 0.32), ((pos[1][0] + pos[1][1]) / 2, zy + zh + 0.35)],
         color=PURP, lw=0.9, ls=":", z=2)
    # MLP zoom
    mx2 = ix + 7.15
    rect(ax, mx2, zy, 5.45, zh + 0.35, fc="white", ec=PURP, ls=":", lw=0.9, r=0.1)
    text(ax, mx2 + 0.2, zy + zh + 0.15, "MLP", size=7.5, color=PURP, ha="left")
    for i, (lab, wd) in enumerate([("Linear 192 → 768", 1.9), ("GELU", 0.85), ("Linear 768 → 192", 1.9)]):
        xx = mx2 + 0.2 + [0, 2.15, 3.25][i]
        rect(ax, xx, cy_ - 0.28, wd, 0.56, fc="white", ec=PURP, r=0.06)
        text(ax, xx + wd / 2, cy_, lab, size=7.5)
        if i < 2:
            arrow(ax, (xx + wd + 0.03, cy_), (xx + wd + 0.22, cy_), head=5, lw=1.0)
    line(ax, [((pos[4][0] + pos[4][1]) / 2, yc - 0.32), ((pos[4][0] + pos[4][1]) / 2, zy + zh + 0.35)],
         color=PURP, lw=0.9, ls=":", z=2)

    # links from the inset to the two block stacks
    line(ax, [(9.65, iy), (9.65, 9.9)], color=GREY, lw=0.9, ls=":", z=2)
    line(ax, [(block_x + 1.1, iy), (block_x + 1.1, 10.15)], color=GREY, lw=0.9, ls=":", z=2)

    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig_model_architecture.{ext}", dpi=220 if ext == "png" else None)
    plt.close(fig)


# ----------------------------------------------------------------------------- Figure B
def pipeline_figure():
    fig, ax = canvas(30.0, 10.8)
    top, ph = 4.7, 5.7

    panel(ax, 0.2, top, 7.2, ph, "1 · Frames")
    panel(ax, 7.6, top, 8.0, ph, "2 · Landmarks")
    panel(ax, 15.8, top, 5.2, ph, "3 · Views")
    panel(ax, 21.2, top, 8.6, ph, "4 · Inference")

    # ---- 1: frames, with the resolution branch
    stack(ax, 0.6, 8.55, 1.5, 0.85, 5, 0.12, "white", GREY)
    text(ax, 1.4, 8.15, "clip · 25 fps", size=8, color=MUTE)
    arrow(ax, (2.75, 9.0), (3.3, 9.0))
    stack(ax, 3.4, 8.7, 1.0, 0.6, 5, 0.1, "white", GREY)
    text(ax, 3.85, 8.15, "32 frames, uniform", size=8, color=MUTE)
    text(ax, 2.55, 7.05, "H < 720 p", size=8, ha="right", color=INK)
    text(ax, 2.55, 5.65, "H ≥ 720 p", size=8, ha="right", color=INK)
    line(ax, [(3.0, 7.55), (3.0, 5.65)], color=GREY, lw=1.0)
    rect(ax, 3.4, 6.35, 1.45, 1.55, fc="none", ec=GREY, ls=(0, (3, 2)), lw=0.9, r=0.1, z=2)
    arrow(ax, (3.0, 7.05), (3.38, 7.05), head=6, lw=1.0)
    arrow(ax, (3.0, 5.65), (3.5, 5.65), head=6, lw=1.0)
    rect(ax, 3.55, 6.55, 1.15, 0.5, fc=BLUE_L, ec=BLUE, r=0.07)
    text(ax, 4.125, 6.8, "bicubic", size=8)
    ax.add_patch(Polygon([(3.55, 7.3), (3.55, 7.55 - 0.0), (4.7, 7.75), (4.7, 7.1)], closed=True, fc=ORNG_L, ec=ORNG,
                         lw=1.2, zorder=3))
    text(ax, 4.2, 7.42, "SR ×4", size=8)
    rect(ax, 3.55, 5.4, 1.15, 0.5, fc=BLUE_L, ec=BLUE, r=0.07)
    text(ax, 4.125, 5.65, "area ↓", size=8)
    # letterbox
    ax.add_patch(Rectangle((5.7, 5.9), 1.35, 1.0, fc=INK, ec=INK, lw=1.0, zorder=3))
    ax.add_patch(Rectangle((5.7, 6.12), 1.35, 0.56, fc="#C7D6EA", ec=BLUE, lw=1.0, zorder=4))
    text(ax, 6.375, 5.6, "letterbox\n1280 × 720", size=8, color=MUTE)
    for yy in (7.05, 6.8, 5.65):
        arrow(ax, (4.8 if yy != 5.65 else 4.8, yy), (5.65, 6.4 + (yy - 6.4) * 0.3), color=GREY, lw=1.0, head=6)

    # ---- 2: landmarks, drawn on a frame
    fx, fy, fw, fh = 7.95, 6.9, 3.3, 2.75
    ax.add_patch(Rectangle((fx, fy), fw, fh, fc="white", ec=GREY, lw=1.1, zorder=3))
    ax.add_patch(Ellipse((9.6, 8.8), 0.7, 0.92, fc="none", ec=GREY, lw=0.9, zorder=4))
    hull(ax, (9.25, 8.34, 9.95, 9.26), GRN, 1.3)
    pose = {"nose": (9.6, 8.7), "ls": (9.0, 7.95), "rs": (10.2, 7.95), "le": (8.25, 7.4), "re": (10.95, 7.4),
            "lw": (8.7, 7.3), "rw": (10.5, 7.3)}
    for a, b in (("ls", "rs"), ("ls", "le"), ("le", "lw"), ("rs", "re"), ("re", "rw")):
        line(ax, [pose[a], pose[b]], color=GREY, lw=0.9, z=5)
    ax.scatter([p[0] for p in pose.values()], [p[1] for p in pose.values()], s=14, color=INK, zorder=7, linewidths=0)
    lb = hand(ax, pose["lw"][0], pose["lw"][1], 0.72, BLUE)
    rb = hand(ax, pose["rw"][0], pose["rw"][1], 0.72, ORNG)
    hull(ax, lb, BLUE, 1.5)
    hull(ax, rb, ORNG, 1.5)
    arrow(ax, (11.35, 8.4), (11.7, 8.4))
    funnel(ax, 11.75, 8.4, 1.35, 1.8, 1.0, PURP_L, PURP)
    text(ax, 12.4, 8.4, "Holistic\ndetector", size=7.5, weight="bold", color=PURP)
    arrow(ax, (13.15, 8.4), (13.45, 8.4))
    for i, (lab, ec, fc) in enumerate([("face box\n× 1.3", GRN, GRN_L), ("hand box × 1.5\n21 joints", BLUE, BLUE_L),
                                        ("pose\n7 keypoints", GREY, GREY_L)]):
        yy = 9.25 - i * 0.95
        rect(ax, 13.5, yy - 0.37, 1.95, 0.74, fc=fc, ec=ec, r=0.08)
        text(ax, 14.475, yy, lab, size=7.5)

    rect(ax, 7.85, 4.8, 7.55, 1.75, fc="white", ec=GREY, ls="--", r=0.1)
    text(ax, 15.2, 6.3, "hand missing", size=8, weight="bold", ha="right")
    ex, wx = (8.05, 5.3), (8.9, 5.75)      # forearm: elbow to wrist
    line(ax, [ex, wx], color=GREY, lw=1.4)
    ax.scatter([ex[0], wx[0]], [ex[1], wx[1]], s=14, color=INK, zorder=7, linewidths=0)
    ax.add_patch(Rectangle((wx[0] - 0.5, wx[1] - 0.5), 1.0, 1.0, fc="none", ec=ORNG, lw=1.2, ls=(0, (3, 2)), zorder=6))
    text(ax, 8.9, 5.0, "2.8 × forearm", size=7, color=ORNG)
    arrow(ax, (9.6, 5.65), (9.95, 5.65), head=6, lw=1.0)
    rect(ax, 10.0, 5.3, 0.45, 0.45, fc=ORNG_L, ec=ORNG, r=0.04)
    arrow(ax, (10.5, 5.52), (10.75, 5.52), head=5, lw=1.0)
    rect(ax, 10.8, 5.2, 0.7, 0.7, fc=ORNG_L, ec=ORNG, r=0.04)
    text(ax, 11.15, 5.0, "256 px", size=7, color=ORNG)
    arrow(ax, (11.6, 5.55), (11.95, 5.55), head=6, lw=1.0)
    funnel(ax, 12.0, 5.55, 1.15, 0.95, 0.5, PURP_L, PURP, lw=1.1)
    text(ax, 12.52, 5.55, "single\nhand", size=7)
    arrow(ax, (13.2, 5.55), (13.55, 5.55), head=6, lw=1.0, color=RED, ls="--")
    rect(ax, 13.6, 5.2, 1.65, 0.7, fc=RED_L, ec=RED, r=0.08)
    text(ax, 14.425, 5.55, "nearest hold,\nflagged", size=7.5, color="#7A2E2C")

    arrow(ax, (7.05, 6.4), (7.9, 6.4), head=0.01, lw=0.01)   # keeps the arrow bounding box tidy
    arrow(ax, (7.4, 7.9), (7.6, 7.9), head=7)
    arrow(ax, (15.6, 8.4), (15.8, 8.4), head=7)

    # ---- 3: views, sampling phases and the flip
    text(ax, 18.4, 9.55, "one frame per segment, 16 segments", size=7.5, color=MUTE)
    seg = 0.285
    for r_, (ph_, yy) in enumerate([(0.25, 9.05), (0.5, 8.75), (0.75, 8.45)]):
        for i in range(16):
            ax.add_patch(Rectangle((16.15 + i * seg, yy), seg, 0.24, fc=GREY_L if i % 2 == 0 else "white",
                                   ec="white", lw=0.5, zorder=3))
            line(ax, [(16.15 + (i + ph_) * seg,) * 2 and (16.15 + (i + ph_) * seg, yy), (16.15 + (i + ph_) * seg, yy + 0.24)],
                 color=INK, lw=1.3, z=5)
    text(ax, 18.4, 8.1, "ticks at 0.25 · 0.5 · 0.75 of each segment", size=7.5, color=MUTE)
    for k, (xx, mirror) in enumerate([(16.7, False), (19.0, True)]):
        ax.add_patch(Rectangle((xx, 6.1), 1.15, 1.15, fc="white", ec=GREY, lw=1.1, zorder=3))
        pts = [(xx + 0.2, 6.3), (xx + 0.95, 6.3), (xx + 0.95, 7.05)] if not mirror else \
              [(xx + 0.95, 6.3), (xx + 0.2, 6.3), (xx + 0.2, 7.05)]
        ax.add_patch(Polygon(pts, closed=True, fc=BLUE_L, ec=BLUE, lw=1.0, zorder=4))
    arrow(ax, (18.0, 6.7), (18.9, 6.7), head=7)
    text(ax, 18.45, 7.0, "flip", size=8)
    text(ax, 18.45, 5.75, "left ↔ right streams swapped", size=7.5, color=MUTE)
    text(ax, 18.4, 5.2, "3 phases × 2 = 6 views", size=9, weight="bold", color=BLUE)
    arrow(ax, (21.0, 8.4), (21.2, 8.4), head=7)

    # ---- 4: inference
    funnel(ax, 21.55, 7.9, 1.5, 2.3, 0.8, PURP_L, PURP)
    text(ax, 22.3, 7.9, "ISL-ViT\nTiny", size=8, weight="bold", color=PURP)
    text(ax, 22.0, 6.45, "4-bit weights", size=7.5, color=MUTE)
    arrow(ax, (23.15, 7.9), (23.5, 7.9), head=6)
    for r_ in range(2):
        for c_ in range(3):
            bars(ax, 23.55 + c_ * 0.62, 8.2 - r_ * 0.62, [0.16, 0.34, 0.12, 0.3, 0.2], bw=0.07, gap=0.03)
    text(ax, 24.4, 6.85, "6 posteriors", size=7.5, color=MUTE)
    arrow(ax, (25.4, 7.9), (25.7, 7.9), head=6)
    plus(ax, 26.0, 7.9, r=0.25)
    text(ax, 26.0, 7.35, "mean", size=7.5, color=MUTE)
    arrow(ax, (26.3, 7.9), (26.6, 7.9), head=6)
    hts = [0.12, 0.2, 0.08, 0.15, 0.1, 0.1, 0.12, 0.7, 0.18, 0.1]
    cols = [(GREY_L, GREY)] * 4 + [(GRN_L, GRN) if i == 7 else ("#D9DDE2", GREY) for i in range(4, 10)]
    bars(ax, 26.65, 7.55, hts, bw=0.08, gap=0.03, colors=cols)
    for i in range(4):
        ax.add_patch(Rectangle((26.65 + i * 0.11, 7.55), 0.08, hts[i], fc="white", ec=GREY, lw=0.7, hatch="//////", zorder=5))
    text(ax, 27.2, 7.1, "262 → 50 words", size=7.5, color=MUTE)
    arrow(ax, (27.9, 7.9), (28.2, 7.9), head=6)
    ax.add_patch(Circle((28.55, 7.9), 0.32, fc="white", ec=INK, lw=1.2, zorder=7))
    text(ax, 28.55, 7.9, "τ", size=11, style="italic")
    arrow(ax, (28.55, 7.58), (28.55, 6.95), head=7)
    rect(ax, 27.75, 6.1, 1.6, 0.8, fc=GRN_L, ec=GRN, r=0.1)
    text(ax, 28.55, 6.5, "word", size=8.5, weight="bold", color=GRN)
    arrow(ax, (28.55, 8.22), (28.55, 8.85), head=7, color=GREY)
    rect(ax, 27.75, 8.9, 1.6, 0.8, fc=RED_L, ec=RED, r=0.1)
    text(ax, 28.55, 9.3, "abstain\ntop-5", size=8, color="#7A2E2C")

    # ---- offline training
    panel(ax, 0.2, 0.3, 29.6, 4.1, "Offline training")
    cylinder(ax, 0.7, 1.45, 1.3, 1.5, GREY_L, GREY)
    text(ax, 1.35, 1.05, "unlabelled\nsign video", size=8, color=MUTE)
    arrow(ax, (2.15, 2.2), (2.6, 2.2))
    ax.add_patch(Rectangle((2.65, 1.7), 1.0, 1.0, fc="white", ec=INK, lw=1.0, zorder=3))
    masked = {(0, 0), (0, 2), (0, 3), (1, 0), (1, 1), (1, 3), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 3)}
    for (r_, c_) in masked:
        ax.add_patch(Rectangle((2.65 + c_ * 0.25, 1.7 + r_ * 0.25), 0.25, 0.25, fc=INK, ec="white", lw=0.5, zorder=4))
    arrow(ax, (3.72, 2.2), (4.0, 2.2), head=6, lw=1.0)
    funnel(ax, 4.05, 2.2, 0.75, 1.0, 0.45, BLUE_L, BLUE, lw=1.1)
    funnel(ax, 4.95, 2.2, 0.75, 0.45, 1.0, GREY_L, GREY, lw=1.1)
    arrow(ax, (5.82, 2.2), (6.1, 2.2), head=6, lw=1.0)
    ax.add_patch(Rectangle((6.15, 1.7), 1.0, 1.0, fc=GREY_L, ec=INK, lw=1.0, zorder=3))
    for i in range(1, 4):
        line(ax, [(6.15 + 0.25 * i, 1.7), (6.15 + 0.25 * i, 2.7)], color="white", lw=0.8)
        line(ax, [(6.15, 1.7 + 0.25 * i), (7.15, 1.7 + 0.25 * i)], color="white", lw=0.8)
    text(ax, 4.9, 3.1, "encoder", size=7.5, color=BLUE)
    text(ax, 5.3, 1.1, "masked image modelling · 75 % of patches hidden", size=8, color=MUTE)
    arrow(ax, (7.3, 2.2), (9.35, 2.2))
    text(ax, 8.3, 2.5, "pre-trained encoder", size=7.5, color=MUTE)
    rect(ax, 9.4, 1.65, 2.7, 1.1, fc=BLUE_L, ec=BLUE, r=0.12)
    text(ax, 10.75, 2.2, "Supervised\nlabel smoothing · EMA", size=8.5)
    cylinder(ax, 10.2, 3.0, 0.9, 0.9, GREY_L, GREY)
    text(ax, 11.25, 3.45, "labelled clips,\nsession-disjoint", size=7.5, color=MUTE, ha="left")
    arrow(ax, (10.65, 3.0), (10.65, 2.78), head=6, lw=1.0)
    text(ax, 10.75, 1.3, "flip + stream swap · mixup · jitter", size=7.5, color=MUTE)
    arrow(ax, (12.2, 2.2), (12.75, 2.2))
    rect(ax, 12.8, 1.65, 2.7, 1.1, fc=BLUE_L, ec=BLUE, r=0.12)
    text(ax, 14.15, 2.2, "Quantisation-aware\ntraining · 4-bit", size=8.5)
    arrow(ax, (15.6, 2.2), (16.2, 2.2))
    rect(ax, 16.25, 1.65, 2.3, 1.1, fc=GRN_L, ec=GRN, r=0.12)
    text(ax, 17.4, 2.2, "Deployed\n4-bit weights", size=8.5, weight="bold", color=GRN)
    line(ax, [(18.6, 2.2), (23.0, 2.2)], color=GRN, lw=1.3, ls="--")
    arrow(ax, (23.0, 2.2), (23.0, 7.3), color=GRN, lw=1.3, ls="--")
    text(ax, 23.15, 4.6, "weights", size=8, color=GRN, ha="left")

    # ---- zoom: group quantisation
    zx, zy, zw, zh = 23.6, 0.55, 6.1, 3.35
    rect(ax, zx, zy, zw, zh, fc="white", ec=GREY, ls="--", lw=1.0, r=0.15, z=1)
    text(ax, zx + 0.25, zy + zh - 0.3, "4-bit group quantisation", size=8.5, weight="bold", color=MUTE, ha="left")
    line(ax, [(14.15, 2.78), (14.15, 3.65), (zx, 3.65)], color=GREY, lw=0.9, ls=":", z=2)
    codes = ["−3", "5", "0", "2", "−6"]
    for g in range(2):
        gx = zx + 0.3 + g * 2.85
        cells(ax, gx, 2.45, 5, 0.4, 0.42, ("#F2F3F5", GREY), gap=0.03)
        cells(ax, gx, 1.2, 5, 0.4, 0.42, (BLUE_L, BLUE), gap=0.03)
        for i, cc in enumerate(codes if g == 0 else codes[::-1]):
            text(ax, gx + 0.2 + i * 0.43, 1.41, cc, size=7)
        rect(ax, gx + 2.25, 2.43, 0.45, 0.46, fc=ORNG_L, ec=ORNG, r=0.05)
        text(ax, gx + 2.475, 2.66, "s", size=8, color=ORNG, style="italic")
        arrow(ax, (gx + 1.1, 2.4), (gx + 1.1, 1.7), head=6, lw=1.0)
        text(ax, gx + 1.3, 2.05, "÷ s, round", size=7, color=MUTE, ha="left")
    text(ax, zx + 3.05, 3.2, "weights in groups of 128 · FP16", size=7.5, color=MUTE)
    text(ax, zx + 3.05, 0.85, "one FP16 scale per group · 4-bit codes", size=7.5, color=MUTE)

    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig_pipeline_architecture.{ext}", dpi=220 if ext == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    model_figure()
    pipeline_figure()
    print("done")
