#!/usr/bin/env bash
# Phase 2: INCLUDE-50 baseline across all three split schemes, plus the
# pretrained-init ablation. Same recipe everywhere -- only the split moves --
# so the spread between schemes is attributable to take leakage alone.
set -u
PY="/c/Users/ADMIN/anaconda3/envs/islvit/python.exe"
CFG=configs/include50.yaml

run () {
  echo ""
  echo "=================== $* ==================="
  "$PY" -m islvit.train --config "$CFG" "$@" 2>&1 \
    | grep -av "UserWarning\|warnings.warn\|symlink\|Developer Mode\|huggingface_hub"
}

# Primary metric, then the two leaky schemes for comparison.
run --split-file splits/include50__take-group.csv
run --split-file splits/include50__random-video.csv
run --split-file splits/include50__official.csv

# Ablation: how much of the result is ImageNet transfer?
run --split-file splits/include50__take-group.csv --no-pretrained --tag include50_take-group_scratch

echo ""
echo "=================== SUMMARY ==================="
"$PY" -m islvit.report
