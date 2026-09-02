#!/usr/bin/env bash
# Overnight: make the leakage ladder comparable again, and measure the noise floor.
#
# 1. take-group on the current recipe. The ladder currently reads take-group 29.3%
#    against session-disjoint 34.8% -- a stricter protocol scoring *higher*, which
#    is incoherent as published. The cause is not leakage: the take-group run
#    predates both the folded-val data and last-epoch selection. Rerunning it on
#    the same recipe as the session-disjoint headline restores the comparison.
#
# 2. Two more seeds of the best configuration. Every ablation in the report is
#    quoted without error bars, and §6.3 says outright that the noise floor is
#    guessed rather than measured. Three seeds give a spread, which decides
#    whether small deltas like SSL's +1.9 mean anything at all.
set -u
PY="/c/Users/ADMIN/anaconda3/envs/islvit/python.exe"
export ISLVIT_CACHE=cache128
FILTER="UserWarning\|warnings.warn\|symlink\|Developer Mode\|huggingface_hub"

echo "waiting for the control run to clear the GPU..."
until [ -f runs/sd263_lastsel/summary.json ]; do
  if ! ps -W 2>/dev/null | grep -q "envs/islvit"; then
    echo "control vanished without a summary; continuing anyway"
    break
  fi
  sleep 60
done
while ps -W 2>/dev/null | grep -q "envs/islvit"; do sleep 30; done

echo ""
echo "=================== take-group on the current recipe ==================="
"$PY" -m islvit.train --config configs/full263_v2.yaml \
  --split-file splits/full263__take-group-foldval.csv \
  --init-from runs/cislr_mim/pretrained.pt \
  --select last --tag tg263_ssl_foldval 2>&1 | grep -av "$FILTER"

for seed in 1 2; do
  echo ""
  echo "=================== best config, seed $seed ==================="
  "$PY" -m islvit.train --config configs/full263_v2.yaml \
    --split-file splits/full263__session-disjoint-foldval.csv \
    --init-from runs/cislr_mim/pretrained.pt \
    --select last --seed "$seed" --tag "sd263_ssl_foldval_s$seed" 2>&1 | grep -av "$FILTER"
done

echo ""
echo "=================== TTA over the new runs ==================="
for run in tg263_ssl_foldval sd263_ssl_foldval_s1 sd263_ssl_foldval_s2; do
  [ -f "runs/$run/summary.json" ] && "$PY" -m islvit.tta --run "runs/$run" --write-summary 2>&1 | grep -av "$FILTER"
done

echo ""
echo "=================== SUMMARY ==================="
"$PY" -m islvit.report 2>&1 | grep -av "$FILTER"
