"""Write report/final/model_comparison.csv from the audit outputs (no typed numbers).

Inputs: report/final/data/audit_results.json, model_measurements.json,
app_integration_*.json, crosscorpus_release_s*.json.

Usage (repository root):
    python report/final/scripts/make_comparison_csv.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "report" / "final" / "data"
PIN = "https://github.com/not-good-keeper/compact-vit-sign-language/blob/3330b87d607eac34615517c092f79fc3e3abf1b6/"
TREE = PIN.replace("/blob/", "/tree/")


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def pct(x):
    return f"{100 * x:.1f}"


def main() -> None:
    a = load("audit_results.json")
    m = load("model_measurements.json")
    app = {t: load(f"app_integration_{t}.json")["summary"] for t in
           ("int4_lm_s0", "fp32_lm_s0", "fp32_pixel_s0", "int4_lm_s0_180p_bicubic", "int4_lm_s0_180p_srl1",
            "int4_lm_s0_180p_srgan_v2")}
    cross = [load(f"crosscorpus_release_s{s}.json") for s in (0, 1, 2)]
    rel_bytes = a["release_files"]["isl_vit_tiny_lm_int4_s0.pt"]["bytes"]
    lm_params = m["landmark_262"]["params"]["total"]
    px_params = m["pixel_only_262"]["params"]["total"]
    common = {"architecture_family": "factorised spatial-temporal ViT (ISL-ViT-Tiny)",
              "dataset": "INCLUDE", "vocabulary": "262 trained; 50 deployed (masked-50)",
              "test_protocol": "session-disjoint, 472 test clips (splits/vocab50clean__session-disjoint.csv)",
              "metric": "masked-50 top-1, 6-view TTA, mean of 3 seeds",
              "training": "iSign SimMIM init; 1,000 epochs; AdamW 5e-4; final weights; train split full263__session-disjoint-foldval (2,845 clips)"}

    def seeds(key):
        return " / ".join(pct(s["top1"]) for s in a[key]["per_seed"])

    rows = []

    def add(**kw):
        rows.append({**common, **kw})

    add(model_id="pixel_fp32", design="3 crops per frame, 16 frames, no landmarks", input_modality="RGB crops",
        input_config="16 x 3 x 64x64 px", parameters=px_params,
        gmacs_per_view=f"{m['pixel_only_262']['gmacs_16f']:.3f}", precision="FP32",
        model_size=f"{4 * px_params} B (params x 4, not a measured file)",
        top1=pct(a["pixel_fp32"]["mean"]["top1"]), top1_per_seed=seeds("pixel_fp32"),
        top5=pct(a["pixel_fp32"]["mean"]["top5"]), balanced=pct(a["pixel_fp32"]["mean"]["balanced"]),
        latency="not measured",
        ui="reachable (--run runs/f16_262w_s0); app test 50 clips: "
           f"{app['fp32_pixel_s0']['correct']}/50 correct, {app['fp32_pixel_s0']['agree_with_offline']}/50 agree with offline",
        evidence=PIN + "runs/f16_262w_s0/masked50_clean.json ; " + PIN + "run_landmarks.sh",
        comparability="directly comparable with the three landmark rows (same test clips, recipe, seeds)")
    add(model_id="landmark_fp32", design="pixel model + 21 hand joints per hand and 7 pose points added to tokens",
        input_modality="RGB crops + MediaPipe landmarks", input_config="16 x 3 x 64x64 px + 2x21x3 joints + 7x3 pose",
        parameters=lm_params, gmacs_per_view=f"{m['landmark_262']['gmacs_16f']:.3f}", precision="FP32",
        model_size=f"{4 * lm_params} B (params x 4, not a measured file)",
        top1=pct(a["landmark_fp32"]["mean"]["top1"]), top1_per_seed=seeds("landmark_fp32"),
        top5=pct(a["landmark_fp32"]["mean"]["top5"]), balanced=pct(a["landmark_fp32"]["mean"]["balanced"]),
        latency="not measured",
        ui="reachable (--run runs/f16_262w_lm_s0); app test 50 clips: "
           f"{app['fp32_lm_s0']['correct']}/50 correct, {app['fp32_lm_s0']['agree_with_offline']}/50 agree",
        evidence=PIN + "runs/f16_262w_lm_s0/masked50_clean.json ; " + PIN + "islvit/models/isl_vit.py#L312-L344",
        comparability="directly comparable")
    add(model_id="landmark_ptq_int4", design="landmark FP32 weights rounded to 4 bits, group 128, no training",
        input_modality="RGB crops + landmarks", input_config="as landmark_fp32", parameters=lm_params,
        gmacs_per_view=f"{m['landmark_262']['gmacs_16f']:.3f}", precision="INT4 storage (group 128, FP16 scales)",
        model_size="same packing as release; no PTQ file written",
        top1=pct(a["landmark_ptq_int4"]["mean"]["top1"]), top1_per_seed=seeds("landmark_ptq_int4"),
        top5=pct(a["landmark_ptq_int4"]["mean"]["top5"]), balanced=pct(a["landmark_ptq_int4"]["mean"]["balanced"]),
        latency="not measured", ui="not reachable (no release file)",
        evidence=PIN + "runs/f16_262w_lm_s0/ptq4_clean.json ; " + PIN + "islvit/int4_eval.py",
        comparability="directly comparable")
    add(model_id="landmark_qat_int4 (released)", design="landmark model + 300 epochs quantisation-aware training",
        input_modality="RGB crops + landmarks", input_config="as landmark_fp32", parameters=lm_params,
        gmacs_per_view=f"{m['landmark_262']['gmacs_16f']:.3f}",
        precision="INT4 storage (group 128, FP16 scales); FP32 arithmetic after load",
        model_size=f"{rel_bytes} B ({rel_bytes / 2**20:.3f} MiB, {rel_bytes / 1e6:.2f} MB)",
        top1=pct(a["landmark_qat_int4"]["mean"]["top1"]), top1_per_seed=seeds("landmark_qat_int4"),
        top5=pct(a["landmark_qat_int4"]["mean"]["top5"]), balanced=pct(a["landmark_qat_int4"]["mean"]["balanced"]),
        latency=f"model only, 1 view, CPU (Ryzen 5 5500, 6 threads): {m['release_s0']['latency_one_view_cpu']['median_ms']:.1f} ms median; "
                f"server route per 1080p clip: CPU {load('app_integration_int4_lm_s0_cpu.json')['summary']['median_seconds']:.1f} s, "
                f"GPU {app['int4_lm_s0']['median_seconds']:.1f} s",
        ui="default model of islvit.demo; app test 50 clips: "
           f"{app['int4_lm_s0']['correct']}/50 correct, {app['int4_lm_s0']['agree_with_offline']}/50 agree",
        evidence=PIN + "runs/f16_262w_lm_s0_qat4/masked50_clean.json ; " + PIN + "release/isl_vit_tiny_lm_int4_s0.pt ; " + PIN + "islvit/qat.py#L50-L84",
        comparability="directly comparable; wide-262 on same clips "
                      f"{pct(a['landmark_qat_int4_wide262']['mean']['top1'])}; CISLR 50-word "
                      f"{sum(100 * c['masked_tta']['top1'] for c in cross) / 3:.1f} (98 clips), 262-word "
                      f"{sum(100 * c['wide_tta']['top1'] for c in cross) / 3:.1f} (609 clips)")
    # Super-resolution front ends, frozen INT4 classifiers, 180p.
    g = a["gan"]["frozen_int4"]["L3"]
    names = {"bicubic": "bicubic (no learning)", "srl1": "SR-L1 generator (no GAN)", "srgan_v1": "SR-GAN v1",
             "srgan": "SR-GAN v2", "sr3_l1_task": "v3 generator without GAN", "sr3_gan_task": "SR-GAN v3"}
    params = {"bicubic": "0", "srl1": "621424", "srgan_v1": "621424", "srgan": "621424",
              "sr3_l1_task": "2039568", "sr3_gan_task": "2039568"}
    ui_sr = {"bicubic": app["int4_lm_s0_180p_bicubic"], "srl1": app["int4_lm_s0_180p_srl1"],
             "srgan": app["int4_lm_s0_180p_srgan_v2"]}
    for arm, label in names.items():
        u = ui_sr.get(arm)
        rows.append({
            "model_id": f"front end: {label}", "architecture_family": "preprocessing (super-resolution)" if arm != "bicubic" else "preprocessing (interpolation)",
            "design": "x4 SRVGG-style generator before hand detection" if arm != "bicubic" else "OpenCV bicubic resize to 720p",
            "input_modality": "180p / JPEG 40 video", "input_config": "frames upscaled to 720p before MediaPipe",
            "parameters": params[arm] + " (generator; recogniser 3,894,214)", "gmacs_per_view": "not measured",
            "precision": "generator FP32 (v3 FP16 at test)", "model_size": "not applicable",
            "training": "see runs/sr_*/summary.json" if arm != "bicubic" else "none",
            "dataset": "INCLUDE", "vocabulary": "50 deployed", "test_protocol": "472 clips degraded to 180p",
            "metric": "masked-50 top-1, 6-view TTA, frozen released INT4 classifiers, mean of 3 seeds",
            "top1": pct(g[arm]["mean"]["top1"]),
            "top1_per_seed": " / ".join(pct(s["top1"]) for s in g[arm]["per_seed"]),
            "top5": pct(g[arm]["mean"]["top5"]), "balanced": pct(g[arm]["mean"]["balanced"]),
            "latency": "not measured on target hardware",
            "ui": (f"{'default path, no --sr' if arm == 'bicubic' else '--sr'}; app test 50 clips at 180p: "
                   f"{u['correct']}/50 correct, {u['agree_with_offline']}/50 agree"
                   if u else ("default path" if arm == "bicubic" else "reachable with --sr; not tested")),
            "evidence": TREE + ("runs/gan_study_v1/frozen_int4" if arm == "srgan_v1" else "runs/gan_study/frozen_int4"),
            "comparability": "comparable across front ends (same clips, same frozen classifiers); not comparable with clean 85.9"})
    fields = ["model_id", "architecture_family", "design", "input_modality", "input_config", "parameters",
              "gmacs_per_view", "precision", "model_size", "training", "dataset", "vocabulary", "test_protocol",
              "metric", "top1", "top1_per_seed", "top5", "balanced", "latency", "ui", "evidence", "comparability"]
    out = ROOT / "report" / "final" / "model_comparison.csv"
    with out.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
