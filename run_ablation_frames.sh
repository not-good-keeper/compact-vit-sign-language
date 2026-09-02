#!/usr/bin/env bash
# Temporal-resolution ablation, evaluated on the strict session-disjoint protocol.
#
# The spatial ablation showed token count is saturated: 64px == 112px, and 128px
# was worse. But the temporal axis was never tested. Sampling 8 frames from a
# 2-5 second clip is only ~2-4 effective fps, and sign language is defined by
# movement, so this is the more likely bottleneck of the two.
#
# The cache stores 16 frames per clip, so every setting up to 16 is free of
# re-extraction. T scales the spatial-encoder pass count linearly, so T=16 costs
# roughly 2x T=8.
set -u
PY="/c/Users/ADMIN/anaconda3/envs/islvit/python.exe"
export ISLVIT_CACHE=cache128

for T in 4 8 12 16; do
  CFG="configs/_frames_${T}.yaml"
  sed "s/^n_frames: .*/n_frames: ${T}/" configs/include50_v2.yaml > "$CFG"
  echo ""
  echo "=================== n_frames=${T} ==================="
  "$PY" -m islvit.train --config "$CFG" \
    --split-file splits/include50__session-disjoint.csv \
    --tag "frames${T}__sd" 2>&1 \
    | grep -av "UserWarning\|warnings.warn\|symlink\|Developer Mode\|huggingface_hub"
done

echo ""
echo "=================== SUMMARY ==================="
"$PY" -m islvit.report
