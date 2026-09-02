#!/usr/bin/env bash
# Test-time augmentation over every finished INCLUDE-263 session-disjoint model.
#
# Waits for run_iter3.sh rather than running beside it: TTA is six forward passes
# over 1,010 clips and would otherwise contend with training for the GPU. That
# contention is not hypothetical -- two earlier diagnostic jobs left running
# alongside training slowed it from ~12 s/epoch to ~30 s.
set -u
PY="/c/Users/ADMIN/anaconda3/envs/islvit/python.exe"
export ISLVIT_CACHE=cache128
FILTER="UserWarning\|warnings.warn\|symlink\|Developer Mode\|huggingface_hub"

echo "waiting for run_iter3.sh to finish..."
until [ -f runs/sd263_ssl_foldval/summary.json ]; do
  if ! ps -W 2>/dev/null | grep -q "envs/islvit"; then
    echo "training process vanished without a summary; continuing with what exists"
    break
  fi
  sleep 60
done

echo ""
echo "=================== TEST-TIME AUGMENTATION ==================="
for run in sd263_ssl sd263_ssl_cislr sd263_foldval sd263_ssl_foldval; do
  if [ -f "runs/$run/summary.json" ]; then
    "$PY" -m islvit.tta --run "runs/$run" --write-summary 2>&1 | grep -av "$FILTER"
    echo ""
  else
    echo "[$run] no summary.json, skipped"
  fi
done

echo "=================== SUMMARY ==================="
"$PY" -m islvit.report 2>&1 | grep -av "$FILTER"
