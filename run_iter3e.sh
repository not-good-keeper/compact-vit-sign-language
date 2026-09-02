#!/usr/bin/env bash
# Seeds 3 and 4 of the headline configuration.
#
# Three seeds gave 34.9 / 33.2 / 38.0 -- a 4.8-point range and an sd of 2.4 on a
# sample of three, which is itself barely more reliable than the deltas it is
# meant to adjudicate. Five seeds is still small but halves the standard error on
# that estimate, and the estimate is now load-bearing: every claim in §11.4 under
# ~5 points is judged against it.
#
# Nothing else in the queue depends on these, so they are the cheapest remaining
# way to make the existing results interpretable rather than merely larger.
set -u
PY="/c/Users/ADMIN/anaconda3/envs/islvit/python.exe"
export ISLVIT_CACHE=cache128
FILTER="UserWarning\|warnings.warn\|symlink\|Developer Mode\|huggingface_hub"

echo "waiting for the seed-2 TTA pass to clear the GPU..."
while ps -W 2>/dev/null | grep -q "envs/islvit"; do sleep 30; done

for seed in 3 4; do
  echo ""
  echo "=================== best config, seed $seed ==================="
  "$PY" -m islvit.train --config configs/full263_v2.yaml \
    --split-file splits/full263__session-disjoint-foldval.csv \
    --init-from runs/cislr_mim/pretrained.pt \
    --select last --seed "$seed" --tag "sd263_ssl_foldval_s$seed" 2>&1 | grep -av "$FILTER"

  "$PY" -m islvit.tta --run "runs/sd263_ssl_foldval_s$seed" --write-summary 2>&1 | grep -av "$FILTER"
done

echo ""
echo "=================== SEED SPREAD ==================="
"$PY" - <<'PYEOF'
import json, statistics
from pathlib import Path

tags = ["sd263_ssl_foldval"] + [f"sd263_ssl_foldval_s{s}" for s in (1, 2, 3, 4)]
rows = []
for tag in tags:
    path = Path("runs") / tag / "summary.json"
    if path.exists():
        rows.append((tag, json.loads(path.read_text(encoding="utf-8"))))

print(f"{'run':28s} {'top1':>7s} {'top5':>7s} {'balanced':>9s} {'top1+TTA':>9s}")
for tag, summary in rows:
    tta = summary.get("test_tta", {})
    print(
        f"{tag:28s} {summary['test']['top1']:6.1%} {summary['test']['top5']:6.1%} "
        f"{summary['test']['balanced']:8.1%} {tta.get('top1', float('nan')):8.1%}"
    )

for name, values in (
    ("top1", [s["test"]["top1"] for _, s in rows]),
    ("balanced", [s["test"]["balanced"] for _, s in rows]),
    ("top1+TTA", [s["test_tta"]["top1"] for _, s in rows if "test_tta" in s]),
):
    if len(values) >= 2:
        print(
            f"\n{name}: n={len(values)}  mean {statistics.mean(values):.1%}  "
            f"sd {statistics.stdev(values):.1%}  range {min(values):.1%}-{max(values):.1%}"
        )
PYEOF
