#!/usr/bin/env bash
# Seeds 2 seeds and 3 of pretrain-jitter+finetune-jitter to resolve a wide spread.
#
# Seeds 0 and 1 gave 43.2% and 36.8% -- a 6.4-point range, more than 3x the sd of
# the previous pair (finetune-jitter alone: 39.9/39.8, sd 0.1). Two possibilities:
# this configuration is genuinely noisier because pretraining jitter adds
# stochasticity the pretrained checkpoint itself inherits, or seed 0 was simply a
# lucky draw. Two more seeds is the minimum to tell those apart -- four points
# gives an actual distribution rather than a coin flip between two readings.
set -u
PY="/c/Users/ADMIN/anaconda3/envs/islvit/python.exe"
export ISLVIT_CACHE=cache128
FILTER="UserWarning\|warnings.warn\|symlink\|Developer Mode\|huggingface_hub"
SPLIT=splits/full263__session-disjoint-foldval.csv

echo "waiting for the current TTA pass to clear the GPU..."
while ps -W 2>/dev/null | grep -q "envs/islvit"; do sleep 30; done

for seed in 2 3; do
  echo ""
  echo "=========== rj-pretrain + rj-finetune, seed $seed ==========="
  "$PY" -m islvit.train --config configs/full263_v2.yaml --split-file "$SPLIT" \
    --init-from runs/cislr_mim_rj/pretrained.pt \
    --resolution-jitter 0.5 --select last --seed "$seed" \
    --tag "sd263_ssl_rj2_s$seed" 2>&1 | grep -av "$FILTER"

  "$PY" -m islvit.tta --run "runs/sd263_ssl_rj2_s$seed" --write-summary 2>&1 | grep -av "$FILTER"
done

echo ""
echo "=========== SPREAD ==========="
"$PY" - <<'PYEOF'
import json, statistics
from pathlib import Path

tags = [f"sd263_ssl_rj2_s{s}" for s in (0, 1, 2, 3)]
rows = [(t, json.loads((Path("runs")/t/"summary.json").read_text())) for t in tags
        if (Path("runs")/t/"summary.json").exists()]
print(f"{'run':20s} {'top1':>7s} {'balanced':>9s} {'top1+TTA':>9s}")
for tag, s in rows:
    tta = s.get("test_tta", {})
    print(f"{tag:20s} {s['test']['top1']:6.1%} {s['test']['balanced']:8.1%} {tta.get('top1', float('nan')):8.1%}")
for name, vals in (("top1", [s["test"]["top1"] for _, s in rows]),
                    ("top1+TTA", [s["test_tta"]["top1"] for _, s in rows if "test_tta" in s])):
    if len(vals) >= 2:
        print(f"{name}: n={len(vals)} mean {statistics.mean(vals):.1%} sd {statistics.stdev(vals):.1%} range {min(vals):.1%}-{max(vals):.1%}")
PYEOF

echo ""
echo "=========== SUMMARY ==========="
"$PY" -m islvit.report 2>&1 | grep -av "$FILTER"
