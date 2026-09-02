#!/usr/bin/env bash
# Does closing the sharpness gap make CISLR pretraining pay off?
#
# CISLR crops are 2.1x blurrier than INCLUDE's, measured by variance of
# Laplacian: its hands reach 128 px by upscaling from ~78 source px, INCLUDE's by
# downscaling from ~173. So a model pretrained on CISLR and finetuned on INCLUDE
# has to unlearn the blur rather than reuse the features -- a plausible reason
# pretraining measured +1.9 (p=0.36) instead of the gain its 12x extra data
# should have bought.
#
# resolution_jitter degrades INCLUDE clips to a random lower resolution and back,
# putting both corpora in the same sharpness distribution.
#
# Two arms, two seeds each, because the noise floor is +-1.9 points and a single
# run cannot resolve anything smaller:
#   rj + ImageNet  -- does the augmentation help on its own?
#   rj + CISLR SSL -- does it rescue the pretraining specifically?
# The difference between arms is the answer; either alone is uninterpretable.
set -u
PY="/c/Users/ADMIN/anaconda3/envs/islvit/python.exe"
export ISLVIT_CACHE=cache128
FILTER="UserWarning\|warnings.warn\|symlink\|Developer Mode\|huggingface_hub"
SPLIT=splits/full263__session-disjoint-foldval.csv

for seed in 0 1; do
  echo ""
  echo "=========== rj + ImageNet init, seed $seed ==========="
  "$PY" -m islvit.train --config configs/full263_v2.yaml --split-file "$SPLIT" \
    --resolution-jitter 0.5 --select last --seed "$seed" \
    --tag "sd263_rj_s$seed" 2>&1 | grep -av "$FILTER"

  echo ""
  echo "=========== rj + CISLR SSL init, seed $seed ==========="
  "$PY" -m islvit.train --config configs/full263_v2.yaml --split-file "$SPLIT" \
    --init-from runs/cislr_mim/pretrained.pt \
    --resolution-jitter 0.5 --select last --seed "$seed" \
    --tag "sd263_ssl_rj_s$seed" 2>&1 | grep -av "$FILTER"
done

echo ""
echo "=========== TTA ==========="
for run in sd263_rj_s0 sd263_rj_s1 sd263_ssl_rj_s0 sd263_ssl_rj_s1; do
  [ -f "runs/$run/summary.json" ] && "$PY" -m islvit.tta --run "runs/$run" --write-summary 2>&1 | grep -av "$FILTER"
done

echo ""
echo "=========== SUMMARY ==========="
"$PY" -m islvit.report 2>&1 | grep -av "$FILTER"
