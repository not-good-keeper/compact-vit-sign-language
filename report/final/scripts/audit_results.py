"""Recompute every number the final report quotes from the tracked result files.

Nothing here is typed in by hand. Each value is read from a JSON/CSV file under
runs/ or splits/, and the paired statistics are recomputed with an independent
exact McNemar implementation (not imported from islvit.gan_report), so the
report's numbers can be checked against a second piece of code.

Output: report/final/data/audit_results.json, plus a printed summary.

Usage (repository root):
    python report/final/scripts/audit_results.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs"
SPLITS = ROOT / "splits"
OUT = ROOT / "report" / "final" / "data" / "audit_results.json"
SEEDS = (0, 1, 2)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact binomial test on the discordant pairs b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(a: dict, b: dict) -> dict:
    """Clips b gets right and a gets wrong (gained) and the reverse (lost)."""
    ha = dict(zip(a["video_paths"], a["hits_masked_tta"]))
    hb = dict(zip(b["video_paths"], b["hits_masked_tta"]))
    assert ha.keys() == hb.keys(), "different clip sets"
    gained = sum(1 for k in ha if hb[k] and not ha[k])
    lost = sum(1 for k in ha if ha[k] and not hb[k])
    return {"gained": gained, "lost": lost, "p": exact_mcnemar(gained, lost)}


def block(r: dict, key: str) -> dict:
    return {m: r[key][m] for m in ("top1", "top5", "balanced")}


def seed_table(paths: list[Path], key: str = "masked_tta") -> dict:
    rows = [load(p) for p in paths]
    per = [block(r, key) for r in rows]
    agg = {m: mean(x[m] for x in per) for m in ("top1", "top5", "balanced")}
    return {"per_seed": per, "mean": agg, "n": [r["n"] for r in rows],
            "files": [str(p.relative_to(ROOT)).replace("\\", "/") for p in paths]}


def split_counts(name: str) -> dict:
    with (SPLITS / name).open(encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    count = Counter(r["split"] for r in rows)
    words = len({r["label"] for r in rows})
    # Leak check: a take-group must never straddle train and test.
    sides = defaultdict(set)
    for r in rows:
        if r["split"] in ("train", "test"):
            sides[(r["label"], r["take_group"])].add(r["split"])
    straddle = sum(1 for s in sides.values() if len(s) > 1)
    return {"counts": dict(count), "words": words, "take_groups_straddling_train_test": straddle}


def main() -> None:
    res: dict = {}

    # ---- 1. Shipped model and its controlled variants (472 clean test clips) ----
    pix = [RUNS / f"f16_262w_s{s}" / "masked50_clean.json" for s in SEEDS]
    lm = [RUNS / f"f16_262w_lm_s{s}" / "masked50_clean.json" for s in SEEDS]
    ptq = [RUNS / f"f16_262w_lm_s{s}" / "ptq4_clean.json" for s in SEEDS]
    qat = [RUNS / f"f16_262w_lm_s{s}_qat4" / "masked50_clean.json" for s in SEEDS]
    res["pixel_fp32"] = seed_table(pix)
    res["landmark_fp32"] = seed_table(lm)
    res["landmark_ptq_int4"] = seed_table(ptq)
    res["landmark_qat_int4"] = seed_table(qat)
    res["landmark_qat_int4_wide262"] = seed_table(qat, "wide_tta")
    res["landmark_qat_int4_masked_single_view"] = seed_table(qat, "masked_plain")
    res["pixel_fp32_wide262"] = seed_table(pix, "wide_tta")
    res["landmark_fp32_wide262"] = seed_table(lm, "wide_tta")

    # All headline files must score the identical 472 clips, and those must be
    # the test rows of the deployed split and the _test472 list.
    with (SPLITS / "vocab50clean__session-disjoint.csv").open(encoding="utf-8") as h:
        test_rows = {r["video_path"] for r in csv.DictReader(h) if r["split"] == "test"}
    with (SPLITS / "_test472_paths.csv").open(encoding="utf-8") as h:
        reader = csv.DictReader(h)
        col = reader.fieldnames[0]
        list472 = {r[col] for r in reader}
    clip_sets = {str(p.relative_to(ROOT)): frozenset(load(p)["video_paths"]) for p in pix + lm + ptq + qat}
    res["test_set_identity"] = {
        "n_test_rows_vocab50clean": len(test_rows),
        "n_test472_list": len(list472),
        "test472_equals_split_test_rows": list472 == test_rows,
        "all_headline_files_score_same_clips": len(set(clip_sets.values())) == 1,
        "headline_clips_equal_split_test_rows": next(iter(clip_sets.values())) == test_rows,
    }

    # Paired comparisons, per seed and pooled.
    def per_seed_pairs(a_paths, b_paths):
        out = [paired(load(a), load(b)) for a, b in zip(a_paths, b_paths)]
        g, l = sum(x["gained"] for x in out), sum(x["lost"] for x in out)
        return {"per_seed": out, "pooled": {"gained": g, "lost": l, "p": exact_mcnemar(g, l)}}

    res["paired_landmark_vs_pixel"] = per_seed_pairs(pix, lm)
    res["paired_qat_vs_ptq"] = per_seed_pairs(ptq, qat)
    res["paired_qat_vs_fp32"] = per_seed_pairs(lm, qat)

    # ---- 2. Release artifacts ----
    rel = {}
    for s in SEEDS:
        f = ROOT / "release" / f"isl_vit_tiny_lm_int4_s{s}.pt"
        rel[f.name] = {"bytes": f.stat().st_size, "MiB": f.stat().st_size / 2 ** 20,
                       "MB": f.stat().st_size / 1e6,
                       "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    res["release_files"] = rel

    # ---- 3. Leakage: same 50-word pixel-only recipe on two split protocols ----
    def summ(run):
        d = load(RUNS / run / "summary.json")
        return {"run": run, "split_file": d.get("split_file"), "n_frames": d.get("n_frames"),
                "epochs": (d.get("recipe") or {}).get("epochs"), "pretrained": d.get("pretrained"),
                "test": d.get("test"), "test_tta": d.get("test_tta")}
    res["leakage_matched"] = {
        "random_video": summ("baseline_random_s0"),
        "session_disjoint": [summ("f16_clean_s0"), summ("f16_clean_s1")],
    }
    sd = res["leakage_matched"]["session_disjoint"]
    res["leakage_matched"]["session_disjoint_mean_top1"] = mean(x["test"]["top1"] for x in sd)
    res["leakage_matched"]["gap_points"] = 100 * (res["leakage_matched"]["random_video"]["test"]["top1"]
                                                  - res["leakage_matched"]["session_disjoint_mean_top1"])

    # Early 8-frame protocol ladder (one recipe, four split files).
    res["protocol_ladder_8frame"] = {r: summ(r) for r in (
        "full263_v2__random-video", "full263_v2__take-group", "sd_full263",
        "include50_v2__random-video", "include50_v2__official", "abl64__take-group", "frames8__sd")}

    # ---- 4. Ablations on fixed test sets ----
    res["frame_sweep_32cache"] = {r: summ(r) for r in (
        "f8_clean_s0", "f16_clean_s0", "f16_clean_s1", "f24_clean_s0", "f32_clean_s0")}
    res["resolution_ablation"] = {r: summ(r) for r in ("abl64__take-group", "abl112__take-group", "abl128__take-group")}
    noise = [summ(r) for r in ("sd263_ssl_foldval", "sd263_ssl_foldval_s1", "sd263_ssl_foldval_s2",
                               "sd263_ssl_foldval_s3", "sd263_ssl_foldval_s4")]
    t1 = [x["test"]["top1"] for x in noise]
    res["noise_floor_5seeds"] = {"runs": noise, "mean_top1": mean(t1), "sd_top1": stdev(t1),
                                 "min": min(t1), "max": max(t1)}
    res["narrow_finetune"] = {}
    for s in SEEDS:
        d = load(RUNS / f"f16_262w_s{s}_narrow" / "summary.json")
        res["narrow_finetune"][s] = {k: d.get(k) for k in ("masked50_before", "masked50")}
    res["full262_test_qat"] = [load(RUNS / f"f16_262w_lm_s{s}_qat4" / "summary.json")["test"] for s in SEEDS]

    # ---- 5. GAN / super-resolution study (frozen INT4 models, 472 clips) ----
    study, study1 = RUNS / "gan_study", RUNS / "gan_study_v1"
    arms = {"bicubic": study, "srl1": study, "srgan_v1": study1, "srgan": study,
            "sr3_l1_task": study, "sr3_gan_task": study}

    def gan_file(family, level, arm, s):
        base = arms[arm]
        name = "srgan" if arm == "srgan_v1" else arm
        return base / family / f"{level}_{name}_s{s}.json"

    gan = {}
    for family in ("frozen_int4", "frozen_fp32"):
        gan[family] = {}
        for level in ("L1", "L2", "L3"):
            gan[family][level] = {}
            for arm in arms:
                paths = [gan_file(family, level, arm, s) for s in SEEDS]
                if all(p.exists() for p in paths):
                    gan[family][level][arm] = seed_table(paths)
    pairs = [("bicubic", "srl1"), ("bicubic", "srgan"), ("bicubic", "srgan_v1"), ("bicubic", "sr3_gan_task"),
             ("srl1", "srgan_v1"), ("srl1", "srgan"), ("sr3_l1_task", "sr3_gan_task"),
             ("srgan_v1", "srgan"), ("srgan", "sr3_gan_task"), ("srl1", "sr3_gan_task")]
    gan["paired_frozen_int4"] = {}
    for level in ("L1", "L2", "L3"):
        for a, b in pairs:
            pa = [gan_file("frozen_int4", level, a, s) for s in SEEDS]
            pb = [gan_file("frozen_int4", level, b, s) for s in SEEDS]
            if not all(p.exists() for p in pa + pb):
                continue
            ps = [paired(load(x), load(y)) for x, y in zip(pa, pb)]
            deltas = [load(y)["masked_tta"]["top1"] - load(x)["masked_tta"]["top1"] for x, y in zip(pa, pb)]
            g, l = sum(x["gained"] for x in ps), sum(x["lost"] for x in ps)
            p = exact_mcnemar(g, l)
            m = mean(deltas)
            verdict = ("helps" if m > 0 and all(d > 0 for d in deltas) and p < 0.05 else
                       "hurts" if m < 0 and all(d < 0 for d in deltas) and p < 0.05 else "no reliable difference")
            gan["paired_frozen_int4"][f"{level}:{b}_vs_{a}"] = {
                "delta_points": 100 * m, "per_seed_points": [100 * d for d in deltas],
                "gained": g, "lost": l, "p": p, "verdict": verdict}
    # Retrained (fine-tuned at 360p) arms.
    gan["retrained"] = {}
    for level in ("clean", "L1", "L2", "L3"):
        gan["retrained"][level] = {}
        for arm, base in (("bicubic", study), ("srgan", study), ("srgan_v1", study1)):
            name = "srgan" if arm == "srgan_v1" else arm
            paths = [base / "retrained" / f"{level}_{name}_s{s}.json" for s in SEEDS]
            if all(p.exists() for p in paths):
                gan["retrained"][level][arm] = seed_table(paths)
    # Detection (left / right genuine share etc.) and image quality: copied verbatim.
    gan["detection"] = {}
    for base in (study, study1):
        for f in sorted((base / "detection").glob("*.json")):
            key = ("v1:" if base == study1 else "") + f.stem
            gan["detection"][key] = load(f)
    gan["sr_quality"] = {f.name: load(f) for f in sorted(study.glob("sr_quality*.json"))}
    gan["sr_quality"]["v1"] = load(study1 / "sr_quality.json")
    diag = study / "diagnostics"
    gan["diagnostics_files"] = sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in diag.rglob("*.json"))
    # 2x2 decomposition at 180p: clean/degraded pixels x clean/degraded landmarks.
    dec = {}
    for arm in ("bicubic", "srgan"):
        both = mean(load(gan_file("frozen_int4", "L3", arm, s))["masked_tta"]["top1"] for s in SEEDS)
        clean_pix = mean(load(diag / f"res_clean_pix_L3_{arm}_s{s}.json")["masked_tta"]["top1"] for s in SEEDS)
        clean_lm = mean(load(diag / "mirror" / f"degraded_pixels_clean_landmarks_L3_{arm}_s{s}.json")["masked_tta"]["top1"]
                        for s in SEEDS)
        clean = res["landmark_qat_int4"]["mean"]["top1"]
        dec[arm] = {"both_degraded": both, "pixels_clean_landmarks_degraded": clean_pix,
                    "pixels_degraded_landmarks_clean": clean_lm, "clean": clean,
                    "missing_points": 100 * (clean - both),
                    "landmark_side_points": 100 * (clean - clean_pix),
                    "pixel_side_points": 100 * (clean - clean_lm)}
    gan["decomposition_180p"] = dec
    res["gan"] = gan

    # Clean-video detection on the 472 test clips (local cache; skipped if absent).
    test_cache = ROOT / "cache128_f32_test"
    if (test_cache / "sources.npy").exists():
        import numpy as np
        with (test_cache / "index.csv").open(encoding="utf-8") as h:
            idx = {r["video_path"]: int(r["row"]) for r in csv.DictReader(h) if r["cached"] == "1"}
        src = np.load(test_cache / "sources.npy", mmap_mode="r")
        s = np.stack([src[idx[p]] for p in sorted(test_rows)])
        res["clean_test_detection_genuine"] = {n: float(np.isin(s[:, :, k], [1, 2]).mean())
                                               for k, n in enumerate(("left_hand", "right_hand", "face"))}

    # ---- 6. Split catalogue ----
    res["splits"] = {n: split_counts(n) for n in (
        "full263__session-disjoint.csv", "full263__session-disjoint-foldval.csv", "full263__random-video.csv",
        "full263__take-group.csv", "vocab50clean__session-disjoint.csv", "vocab50__random-video.csv",
        "include50__session-disjoint.csv", "crosscorpus__cislr-test.csv")}

    # Independent session check: rebuild the capture-id blocks from the file names
    # of the whole corpus (gap > 5, as islvit/data/splits.py does) and count blocks
    # that put clips on both sides of the train/test boundary.
    import re
    with (SPLITS / "full263__session-disjoint-foldval.csv").open(encoding="utf-8") as h:
        corpus = list(csv.DictReader(h))
    mvi = lambda p: int(m.group(1)) if (m := re.search(r"MVI_(\d+)", p)) else -1
    ids = sorted({mvi(r["video_path"]) for r in corpus if mvi(r["video_path"]) >= 0})
    edges, start, prev = [], ids[0], ids[0]
    for cur in ids[1:]:
        if cur - prev > 5:
            edges.append((start, prev))
            start = cur
        prev = cur
    edges.append((start, prev))
    block_of = lambda i: next(k for k, (lo, hi) in enumerate(edges) if lo <= i <= hi)

    def session_leak(name: str) -> dict:
        with (SPLITS / name).open(encoding="utf-8") as h:
            rows = list(csv.DictReader(h))
        sides = defaultdict(set)
        for r in rows:
            if r["split"] in ("train", "test") and mvi(r["video_path"]) >= 0:
                sides[block_of(mvi(r["video_path"]))].add(r["split"])
        return {"blocks_with_rows": len(sides),
                "blocks_on_both_sides": sum(1 for s in sides.values() if len(s) > 1),
                "test_blocks": sorted(k for k, s in sides.items() if s == {"test"})}
    res["session_blocks"] = {"n_blocks_corpus": len(edges), "edges": edges,
                             **{n: session_leak(n) for n in ("full263__session-disjoint-foldval.csv",
                                                             "vocab50clean__session-disjoint.csv",
                                                             "full263__take-group.csv",
                                                             "vocab50__random-video.csv")}}

    # ---- 7. Confidence gating curve used by the interface ----
    res["gate_mixed"] = load(RUNS / "gate_mixed.json")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")

    # ---- printed summary ----
    pct = lambda x: f"{100 * x:.1f}"
    for k in ("pixel_fp32", "landmark_fp32", "landmark_ptq_int4", "landmark_qat_int4",
              "landmark_qat_int4_wide262", "landmark_qat_int4_masked_single_view"):
        t = res[k]
        print(f"{k:40s} top1 {pct(t['mean']['top1'])} ({' / '.join(pct(x['top1']) for x in t['per_seed'])})"
              f"  top5 {pct(t['mean']['top5'])}  bal {pct(t['mean']['balanced'])}  n={t['n']}")
    for k in ("paired_landmark_vs_pixel", "paired_qat_vs_ptq", "paired_qat_vs_fp32"):
        v = res[k]
        print(k, [(x["gained"], x["lost"], f"{x['p']:.2g}") for x in v["per_seed"]], "pooled", v["pooled"])
    print("test-set identity", res["test_set_identity"])
    print("release", {k: v["bytes"] for k, v in rel.items()})
    print("leakage gap", round(res["leakage_matched"]["gap_points"], 1))
    print("noise floor", pct(res["noise_floor_5seeds"]["mean_top1"]), "+-", pct(res["noise_floor_5seeds"]["sd_top1"]))
    for level in ("L1", "L2", "L3"):
        print(level, {a: pct(v["mean"]["top1"]) for a, v in gan["frozen_int4"][level].items()})
    for k, v in gan["paired_frozen_int4"].items():
        print(f"  {k:32s} {v['delta_points']:+.1f}  {v['gained']}/{v['lost']}  p={v['p']:.2g}  {v['verdict']}")
    print("splits", {k: (v["counts"], v["take_groups_straddling_train_test"]) for k, v in res["splits"].items()})
    print("session blocks", {k: v for k, v in res["session_blocks"].items() if k != "edges"})


if __name__ == "__main__":
    main()
