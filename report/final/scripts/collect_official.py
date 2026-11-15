"""Collect the official-INCLUDE-split results into one file for the paper.

Reads runs/official_<bench>_lm_s<seed>[_qat4]/official_eval.json, written by
islvit.mask50 with the official split as both vocabulary and test set, so the
"wide" scores cover every official test clip of that benchmark.

    python report/final/scripts/collect_official.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "report/final/data/official_split.json"
BENCHES = {"include50": "INCLUDE-50", "full263": "INCLUDE, 262 words"}


def main() -> None:
    table: dict = {}
    for bench, name in BENCHES.items():
        for variant, suffix in (("fp32", ""), ("int4_qat", "_qat4")):
            seeds = {}
            for seed in (0, 1, 2):
                path = ROOT / f"runs/official_{bench}_lm_s{seed}{suffix}/official_eval.json"
                if not path.exists():
                    continue
                r = json.loads(path.read_text(encoding="utf-8"))
                seeds[seed] = {"n": r["n_wide"],
                               "plain": r["wide_plain"], "tta": r["wide_tta"]}
            if not seeds:
                continue
            mean = {view: {k: round(100 * sum(s[view][k] for s in seeds.values()) / len(seeds), 1)
                           for k in ("top1", "top5", "balanced")}
                    for view in ("plain", "tta")}
            table.setdefault(bench, {"name": name})[variant] = {
                "seeds": sorted(seeds), "n": next(iter(seeds.values()))["n"], "mean": mean,
                "per_seed_top1": {view: [round(100 * seeds[s][view]["top1"], 1) for s in sorted(seeds)]
                                  for view in ("plain", "tta")},
            }
    OUT.write_text(json.dumps(table, indent=2), encoding="utf-8")
    for bench, rows in table.items():
        for variant, row in rows.items():
            if variant == "name":
                continue
            m = row["mean"]
            print(f"{bench:9s} {variant:8s} seeds {row['seeds']} n={row['n']}  "
                  f"single {m['plain']['top1']:.1f}  TTA {m['tta']['top1']:.1f} "
                  f"(top5 {m['tta']['top5']:.1f}, bal {m['tta']['balanced']:.1f})  "
                  f"per-seed TTA {row['per_seed_top1']['tta']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
