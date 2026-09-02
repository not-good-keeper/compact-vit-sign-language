#!/usr/bin/env bash
# Confirm the temporal ablation on the benchmark that can actually resolve it.
#
# On include50 session-disjoint the four settings came out 34.1 / 31.3 / 26.9 /
# 35.3 balanced -- non-monotonic, with T=8 and T=12 dipping below both ends. A
# U-shape in frame count has no mechanism behind it, so that is split noise, not
# signal: 252 test clips over 45 classes with corr(test_count, train_count) =
# -0.73 cannot separate settings a few points apart.
#
# full263 session-disjoint has 1010 test clips over 154 classes and corr -0.37.
# T=16 is the candidate (best top-5 there by 12 points); T=4 is the cheap end
# worth bracketing against. T=8 is already measured as sd_full263.
set -u
PY="/c/Users/ADMIN/anaconda3/envs/islvit/python.exe"
export ISLVIT_CACHE=cache128

for T in 16 4; do
  CFG="configs/_f263_${T}.yaml"
  sed "s/^n_frames: .*/n_frames: ${T}/" configs/full263_v2.yaml > "$CFG"
  echo ""
  echo "=================== full263 SD  n_frames=${T} ==================="
  "$PY" -m islvit.train --config "$CFG" \
    --split-file splits/full263__session-disjoint.csv \
    --tag "sd263_f${T}" 2>&1 \
    | grep -av "UserWarning\|warnings.warn\|symlink\|Developer Mode\|huggingface_hub"
done

echo ""
echo "=================== SUMMARY ==================="
"$PY" -m islvit.report
