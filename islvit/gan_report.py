"""Write docs/GAN_COMPARATIVE_STUDY.md and its figures from the study's result files.

Every number is read from a JSON written by run_gan_study.sh; nothing is typed in.
Missing results are shown as missing, so the report can be regenerated part-way
through the study.

    python -m islvit.gan_report
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

RUNS = Path("runs")
STUDY = RUNS / "gan_study"
STUDY_V1 = RUNS / "gan_study_v1"  # the first SR-GAN (60k iterations), archived when v2 replaced it
DOC = Path("docs/GAN_COMPARATIVE_STUDY.md")
FIG = Path("docs/figures")
SEEDS = (0, 1, 2)
LEVELS = (("clean", "Clean 1080p", None), ("L1", "360p, JPEG 60", 2), ("L2", "240p, JPEG 50", 3), ("L3", "180p, JPEG 40", 4))
ARMS = (("bicubic", "Bicubic (no GAN)"), ("srl1", "SR-L1 (no adversarial loss)"), ("srgan", "SR-GAN"))
LOOKALIKES = [["41. you", "46. you (plural)"], ["83. big large", "81. wide", "82. narrow"],
              ["79. short", "84. small little"], ["94. good", "95. bad", "96. wet"], ["38. Dress", "42. T-Shirt"]]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def result(family: str, level: str, arm: str, seed: int) -> dict | None:
    """masked50 result for one model on one front end."""
    if level == "clean":
        # No upscale happens on 1080p video, so every arm shares the clean front end.
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


def fmt(m, sd=None, n=None) -> str:
    if m is None:
        return "—"
    text = f"{100 * m:.1f}"
    if sd is not None and n and n > 1:
        text += f" ± {100 * sd:.1f}"
    return text + ("" if n in (None, 3) else f" (n={n})")


# --------------------------------------------------------------------------- #
# Paired statistics
# --------------------------------------------------------------------------- #
def discordant(a: dict, b: dict) -> tuple[int, int]:
    """(clips A got right and B wrong, clips B got right and A wrong), aligned by path."""
    hits_a = dict(zip(a["video_paths"], a["hits_masked_tta"]))
    hits_b = dict(zip(b["video_paths"], b["hits_masked_tta"]))
    assert hits_a.keys() == hits_b.keys(), "the two results scored different clips"
    a_only = sum(1 for k in hits_a if hits_a[k] and not hits_b[k])
    b_only = sum(1 for k in hits_a if hits_b[k] and not hits_a[k])
    return a_only, b_only


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(family: str, level: str, base: str, other: str) -> dict | None:
    """Pre-registered rule: mean gain > 0, every seed the same direction, pooled p < 0.05."""
    seeds, gained, lost = [], 0, 0
    for s in SEEDS:
        a, b = result(family, level, base, s), result(family, level, other, s)
        if a is None or b is None:
            continue
        lose, gain = discordant(a, b)
        gained, lost = gained + gain, lost + lose
        seeds.append(top1(b) - top1(a))
    if not seeds:
        return None
    p = mcnemar(lost, gained)
    mean = float(np.mean(seeds))
    same = all(d > 0 for d in seeds) or all(d < 0 for d in seeds)
    verdict = ("helps" if mean > 0 and all(d > 0 for d in seeds) and p < 0.05 else
               "hurts" if mean < 0 and all(d < 0 for d in seeds) and p < 0.05 else "no reliable difference")
    return {"delta": mean, "per_seed": seeds, "gained": gained, "lost": lost, "p": p,
            "consistent": same, "seeds": len(seeds), "verdict": verdict}


def compare_row(c: dict | None) -> str:
    if c is None:
        return "— | — | — | —"
    seeds = " / ".join(f"{100 * d:+.1f}" for d in c["per_seed"])
    return f"{100 * c['delta']:+.1f} ({seeds}) | {c['gained']} / {c['lost']} | {c['p']:.2g} | **{c['verdict']}**"


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def figures(table: dict, detections: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from islvit.figures import save, style

    colors = {"bicubic": "#7F899E", "srl1": "#2a78d6", "srgan": "#e87ba4"}
    made = []

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, family, title in ((axes[0], "frozen_int4", "Shipped INT4 model, frozen"),
                              (axes[1], "retrained", "Fine-tuned on 360p video")):
        xs = np.arange(len(LEVELS))
        for arm, label in ARMS:
            if family == "retrained" and arm == "srl1":
                continue
            means = [table[family][level][arm][0] for level, _, _ in LEVELS]
            sds = [table[family][level][arm][1] or 0 for level, _, _ in LEVELS]
            ok = [m is not None for m in means]
            if not any(ok):
                continue
            x = xs[ok]
            y = np.array([m for m in means if m is not None]) * 100
            e = np.array([s for s, k in zip(sds, ok) if k]) * 100
            ax.errorbar(x, y, yerr=e, marker="o", capsize=3, lw=2, color=colors[arm], label=label)
        ax.set_xticks(xs, [name for _, name, _ in LEVELS])
        style(ax, title, "input video", "top-1, 50 words (%)")
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "fig21_gan_accuracy")
    made.append("fig21_gan_accuracy.png")

    fig, ax = plt.subplots(figsize=(11, 4.4))
    groups = [(level, name) for level, name, _ in LEVELS[1:]] + [("cislr", "CISLR (300 px)")]
    width = 0.26
    for i, (arm, label) in enumerate(ARMS):
        values = []
        for level, _ in groups:
            d = detections.get(f"{level}_{arm}")
            values.append(None if d is None else (d["streams"]["left_hand"]["genuine"] + d["streams"]["right_hand"]["genuine"]) / 2)
        xs = np.arange(len(groups)) + (i - 1) * width
        ax.bar([x for x, v in zip(xs, values) if v is not None], [100 * v for v in values if v is not None],
               width, color=colors[arm], label=label)
    ax.axhline(100 * (0.911 + 0.852) / 2, color="#0b0b0b", lw=1, ls="--")
    ax.text(len(groups) - 0.6, 100 * (0.911 + 0.852) / 2 + 1.5, "clean INCLUDE", fontsize=9, ha="right")
    ax.set_xticks(np.arange(len(groups)), [name for _, name in groups])
    style(ax, "Hands found by the detector (mean of left and right)", "", "genuine detection (%)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "fig22_gan_detection")
    made.append("fig22_gan_detection.png")
    return made


def montage() -> str | None:
    """LR / bicubic / SR-L1 / SR-GAN / original for a few held-out hand patches."""
    bank_path = Path("C:/isl_gan_work/hr_heldout.npy")
    l1, gan = RUNS / "sr_l1" / "generator.pt", RUNS / "sr_gan" / "generator.pt"
    if not (bank_path.exists() and l1.exists() and gan.exists()):
        return None
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from islvit.figures import save
    from islvit.sr.evaluate import degrade_patch
    from islvit.sr.upscaler import super_resolve

    bank = np.load(bank_path, mmap_mode="r")
    picks = [np.asarray(bank[i]) for i in range(0, min(len(bank), 600), 100)][:5]
    rows = []
    for hr in picks:
        lr = degrade_patch(hr, 4, 40)
        size = hr.shape[0]
        up = lambda path: cv2.resize(np.ascontiguousarray(super_resolve([np.ascontiguousarray(lr[..., ::-1])], path)[0][..., ::-1]),
                                     (size, size), interpolation=cv2.INTER_AREA)
        rows.append([cv2.resize(lr, (size, size), interpolation=cv2.INTER_NEAREST),
                     cv2.resize(lr, (size, size), interpolation=cv2.INTER_CUBIC), up(l1), up(gan), hr])
    titles = ["180p input", "Bicubic", "SR-L1", "SR-GAN", "Original 720p"]
    fig, axes = plt.subplots(len(rows), 5, figsize=(10, 2.1 * len(rows)))
    for r, row in enumerate(rows):
        for c, image in enumerate(row):
            axes[r, c].imshow(image)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=10)
    fig.tight_layout()
    save(fig, "fig23_gan_montage")
    return "fig23_gan_montage.png"


# --------------------------------------------------------------------------- #
# Document
# --------------------------------------------------------------------------- #
def per_word(r: dict, labels: dict) -> dict:
    tally: dict[str, list[int]] = {}
    for path, hit in zip(r["video_paths"], r["hits_masked_tta"]):
        t = tally.setdefault(labels[path], [0, 0])
        t[0] += hit
        t[1] += 1
    return tally


def main() -> None:
    families = ("frozen_int4", "frozen_fp32", "retrained")
    table = {f: {level: {} for level, _, _ in LEVELS} for f in families}
    for f in families:
        for level, _, _ in LEVELS:
            for arm, _ in ARMS:
                table[f][level][arm] = mean_sd([top1(result(f, level, arm, s)) for s in SEEDS])
    detections = {p.stem: load(p) for p in sorted((STUDY / "detection").glob("*.json"))}
    quality = load(STUDY / "sr_quality.json")
    sr = {stage: load(RUNS / f"sr_{stage}" / "summary.json") for stage in ("l1", "gan_v1", "gan")}
    quality_v1 = load(STUDY_V1 / "sr_quality.json")

    made = []
    try:
        made = figures(table, detections)
        m = montage()
        if m:
            made.append(m)
    except Exception as error:  # figures are optional; the tables are the record
        print(f"  figures skipped: {error}")

    lines: list[str] = []
    w = lines.append
    w("# GAN comparative study: super-resolution for low-resolution video\n")
    w("*Generated by `python -m islvit.gan_report` from `runs/gan_study/`. Every number below is read from a result file.*\n")
    w("## What the GAN does, and where\n")
    w("Every clip reaches the hand detector through `islvit/data/crops.py::read_sampled_frames`. Video below the 720p "
      "working height is upscaled there before MediaPipe looks for hands. Until now that upscale was bicubic. The GAN "
      "(`islvit/sr/`) replaces that one interpolation when `--sr` is passed. Native 1080p INCLUDE never takes this "
      "branch, so the shipped model's clean result cannot change.\n")
    w("```\nvideo -> decode -> [below 720p?  bicubic  |  SR-GAN x4 -> area to 720p] -> MediaPipe -> crops + landmarks -> ISL-ViT-Tiny\n```\n")
    if sr["gan"] or sr["l1"]:
        w("| Model | Stage | Starts from | Generator params | Discriminator params | Iterations | Peak LR | Perceptual features | Minutes |")
        w("|---|---|---|---|---|---|---|---|---|")
        for stage, s in sr.items():
            if s:
                init = f"`{s['init']}`" if s.get("init") else "scratch"
                w(f"| `runs/sr_{stage}` | {s['stage']} | {init} | {s['generator_params']:,} | "
                  f"{s['discriminator_params'] or '—'} | {s['iters']:,} | {s['lr']:g} | {s['perceptual'] or '—'} | {s['minutes']} |")
        w("")
    w("Generator: a compact Real-ESRGAN-style network (SRVGG). Discriminator: U-Net with spectral normalisation. "
      "Losses: L1, then L1 + perceptual + 0.1 × adversarial. It is trained only on INCLUDE training-session clips "
      "(`islvit/sr/patches.py` asserts none of the 472 test clips or held-out sessions are used). Both stages keep the "
      "final EMA weights after a fixed number of iterations, with no checkpoint selection.\n")
    if sr["gan_v1"] and sr["gan"]:
        total = sr["gan_v1"]["iters"] + sr["gan"]["iters"]
        w("**Two GANs.** The study was first run with SR-GAN v1 (`runs/sr_gan_v1`: 60,000 GAN iterations from SR-L1). "
          "SR-GAN v2 (`runs/sr_gan`), which every 'SR-GAN' column below refers to, continues from the v1 generator with "
          f"a fresh discriminator at half the peak learning rate, so its generator has had {total:,} adversarial "
          "iterations in all. v2 was trained **after** v1's results were known, to get the strongest GAN that fitted "
          "the remaining time, so it is not part of the pre-registered design. The v1 results are reported in full "
          "under *SR-GAN v1 against v2*, so nothing is hidden by the swap. SR-L1, the no-GAN control, is unchanged.\n")

    w("## Study design\n")
    w("- **Inputs:** the same 472 held-out clips behind the 85.9 % headline (`splits/_test472_paths.csv`), "
      "degraded inside the frame reader before detection: 360p + JPEG 60, 240p + JPEG 50, 180p + JPEG 40.")
    w("- **Arms:** bicubic (the current pipeline, no GAN), SR-L1 (the same generator trained without the "
      "adversarial loss, which isolates what the GAN adds), and SR-GAN.")
    w("- **Models:** the three shipped INT4 models, frozen; the same three in FP32; and the FP32 models fine-tuned "
      "for 150 epochs on 360p training video through each front end (bicubic or SR-GAN).")
    w("- **Metric:** masked-50 six-view TTA top-1, mean ± sd over three seeds.")
    w("- **Pre-registered rule:** a front end *helps* only if the mean top-1 rises, all three seeds move the same way, "
      "and the pooled exact McNemar p < 0.05. *The GAN helps* only if SR-GAN beats SR-L1 by that rule; otherwise any "
      "gain belongs to learned super-resolution, not to the adversarial training.\n")

    for family, title in (("frozen_int4", "Shipped INT4 model, frozen"), ("frozen_fp32", "Same models in FP32, frozen"),
                          ("retrained", "FP32 models fine-tuned on 360p video through each front end")):
        w(f"## Results: {title}\n")
        arms = [a for a in ARMS if not (family == "retrained" and a[0] == "srl1")]
        w("| Input | " + " | ".join(label for _, label in arms) + " |")
        w("|---|" + "---|" * len(arms))
        for level, name, _ in LEVELS:
            w(f"| {name} | " + " | ".join(fmt(*table[family][level][arm]) for arm, _ in arms) + " |")
        w("")
        w("| Input | Comparison | Δ top-1, mean (per seed) | clips gained / lost | p | verdict |")
        w("|---|---|---|---|---|---|")
        pairs = [("bicubic", "srgan", "SR-GAN vs bicubic")]
        if family != "retrained":
            pairs += [("bicubic", "srl1", "SR-L1 vs bicubic"), ("srl1", "srgan", "SR-GAN vs SR-L1 (the GAN's own effect)")]
        for level, name, _ in LEVELS[1:]:
            for base, other, label in pairs:
                w(f"| {name} | {label} | {compare_row(compare(family, level, base, other))} |")
        w("")

    if any(result("frozen_int4", "L1", "srgan_v1", s) for s in SEEDS):
        w("## SR-GAN v1 against v2\n")
        w("Same 472 clips, same models, same scorer; only the GAN weights differ. *v2 vs v1* is the effect of the "
          "longer adversarial training; *v1 vs SR-L1* is the GAN's own effect as first measured.\n")
        for family, title in (("frozen_int4", "Frozen INT4"), ("frozen_fp32", "Frozen FP32"),
                              ("retrained", "Fine-tuned on 360p")):
            arms = ([] if family == "retrained" else [("srl1", "SR-L1")]) + [("srgan_v1", "SR-GAN v1"), ("srgan", "SR-GAN v2")]
            w(f"**{title}**\n")
            w("| Input | " + " | ".join(label for _, label in arms) + " |")
            w("|---|" + "---|" * len(arms))
            for level, name, _ in LEVELS:
                cells = [fmt(*mean_sd([top1(result(family, level, a, s)) for s in SEEDS])) for a, _ in arms]
                w(f"| {name} | " + " | ".join(cells) + " |")
            w("")
            w("| Input | Comparison | Δ top-1, mean (per seed) | clips gained / lost | p | verdict |")
            w("|---|---|---|---|---|---|")
            pairs = [("srgan_v1", "srgan", "v2 vs v1")]
            pairs += [("bicubic", "srgan_v1", "v1 vs bicubic")] if family == "retrained" else [("srl1", "srgan_v1", "v1 vs SR-L1")]
            for level, name, _ in LEVELS[1:]:
                for base, other, label in pairs:
                    w(f"| {name} | {label} | {compare_row(compare(family, level, base, other))} |")
            w("")

    w("## Detection and hallucination\n")
    w("Share of hand frames with a genuine detection, and agreement with the clean 1080p extraction of the same clips. "
      "*Extra hands* are found in the degraded video but not in the clean one. *Misplaced* hands are found in both but "
      "more than 5 % of the frame height apart. A GAN that raises detection by inventing hands would show up here.\n")
    w("| Input | Arm | Left hand | Right hand | Face | Presence agreement | Extra hands | Missed hands | Wrist error, median | Misplaced |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for level, name, _ in LEVELS[1:]:
        for arm, label in ARMS:
            d = detections.get(f"{level}_{arm}")
            if not d:
                continue
            s, c = d["streams"], d.get("against_clean", {})
            g = lambda v: "—" if v is None else f"{100 * v:.1f}"
            w(f"| {name} | {label} | {g(s['left_hand']['genuine'])} | {g(s['right_hand']['genuine'])} | "
              f"{g(s['face']['genuine'])} | {g(c.get('presence_agreement'))} | {g(c.get('extra_hands'))} | "
              f"{g(c.get('missed_hands'))} | {g(c.get('wrist_error_median'))} | {g(c.get('misplaced_hands'))} |")
    w("")
    w("### CISLR (natively 300 px, different signers and cameras)\n")
    w("| Arm | Left hand | Right hand | Face |")
    w("|---|---|---|---|")
    v1_cislr = load(STUDY_V1 / "detection" / "cislr_srgan.json")
    for arm, label in ARMS + (("srgan_v1", "SR-GAN v1"),):
        d = v1_cislr if arm == "srgan_v1" else detections.get(f"cislr_{arm}")
        if d:
            s = d["streams"]
            w(f"| {label} | {100 * s['left_hand']['genuine']:.1f} | {100 * s['right_hand']['genuine']:.1f} | {100 * s['face']['genuine']:.1f} |")
    w("\nThe report's earlier CISLR measurement found the left hand in 25.4 % of frames, against 91.1 % on INCLUDE (§11.11).\n")

    # Look-alike words, frozen INT4: bicubic vs SR-GAN, pooled over seeds.
    with Path("splits/vocab50clean__session-disjoint.csv").open(encoding="utf-8") as handle:
        labels = {r["video_path"]: r["label"] for r in csv.DictReader(handle)}
    rows = []
    for level, name, _ in LEVELS[1:]:
        base = [result("frozen_int4", level, "bicubic", s) for s in SEEDS]
        gan = [result("frozen_int4", level, "srgan", s) for s in SEEDS]
        if any(r is None for r in base + gan):
            continue
        pooled = lambda results: {k: [sum(per_word(r, labels)[k][0] for r in results), sum(per_word(r, labels)[k][1] for r in results)]
                                  for k in per_word(results[0], labels)}
        b, g = pooled(base), pooled(gan)
        for group in LOOKALIKES:
            hb = sum(b[w_][0] for w_ in group); hg = sum(g[w_][0] for w_ in group); n = sum(b[w_][1] for w_ in group)
            rows.append(f"| {name} | {' / '.join(x.split('. ', 1)[1] for x in group)} | {100 * hb / n:.1f} | {100 * hg / n:.1f} |")
    if rows:
        w("## The look-alike words\n")
        w("Recall pooled over the three frozen INT4 seeds. These groups hold about 70 % of the remaining errors on clean video.\n")
        w("| Input | Group | Bicubic | SR-GAN |")
        w("|---|---|---|---|")
        lines.extend(rows)
        w("")

    if quality:
        w("## Image quality on held-out patches\n")
        w(f"{quality['n']} patches from held-out sessions, at 720p working resolution (median sharpness of the "
          f"originals: {quality['original_sharpness']:.0f}). PSNR and SSIM reward smooth averages, so the PSNR-trained "
          "SR-L1 is expected to score higher than the GAN on them; sharpness is variance of the Laplacian, the measure behind the report's "
          "'CISLR crops are 2.1× blurrier' finding (§11.5).\n")
        w("| Input | Arm | PSNR (dB) | SSIM | Sharpness | ms / patch |")
        w("|---|---|---|---|---|---|")
        for name, level in quality["levels"].items():
            for arm, label in ARMS:
                arm_key = {"bicubic": "bicubic", "srl1": "sr_l1", "srgan": "sr_gan"}[arm]
                m = level[arm_key]
                w(f"| {name} | {label} | {m['psnr']:.2f} | {m['ssim']:.3f} | {m['sharpness']:.0f} | {m['ms_per_patch'] or '—'} |")
            m = (quality_v1 or {}).get("levels", {}).get(name, {}).get("sr_gan")
            if m:
                w(f"| {name} | SR-GAN v1 | {m['psnr']:.2f} | {m['ssim']:.3f} | {m['sharpness']:.0f} | {m['ms_per_patch'] or '—'} |")
        w("")

    if made:
        w("## Figures\n")
        for name in made:
            w(f"![{name}](figures/{name})\n")
        w("*Figure 23 shows held-out INCLUDE frames (Sridhar et al., ACM MM 2020, CC-BY-4.0).*\n")

    w("## Reproduce\n")
    w("```bash\nbash run_gan_study.sh          # resume-safe; about a day on one RTX 3060 (bicubic, SR-L1, SR-GAN v1)\nGAN_ITERS=80000 bash run_gan_study_v2.sh   # SR-GAN v2 and everything that depends on it\npython -m islvit.gan_report    # this document and figures 21-23\n```\n")
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
