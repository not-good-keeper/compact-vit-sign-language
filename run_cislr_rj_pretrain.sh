#!/usr/bin/env bash
# Does re-pretraining WITH resolution_jitter beat finetune-time jitter alone?
#
# The 39.9% result applied jitter only at finetuning time; the CISLR pretraining
# checkpoint (cislr_mim) was trained on CISLR's fixed native blur and never saw a
# *range* of sharpness. This retrains that checkpoint with jitter applied during
# pretraining too, so the encoder has to be blur-invariant from the start rather
# than adapt to it only during the much-shorter finetuning stage.
#
# If this doesn't beat 39.9%, finetune-time jitter was already sufficient and
# this checkpoint isn't worth carrying into the iSign pretraining.
set -u
PY="/c/Users/ADMIN/anaconda3/envs/islvit/python.exe"
export ISLVIT_CACHE=cache128
FILTER="UserWarning\|warnings.warn\|symlink\|Developer Mode\|huggingface_hub"
SPLIT=splits/full263__session-disjoint-foldval.csv

echo "=========== PRETRAIN: CISLR with resolution jitter ==========="
"$PY" -m islvit.pretrain --cache cache_cislr --tag cislr_mim_rj \
  --epochs 100 --batch-size 64 --min-genuine 0.25 --resolution-jitter 0.5 2>&1 | grep -av "$FILTER"

if [ ! -f runs/cislr_mim_rj/pretrained.pt ]; then
  echo "pretraining produced no checkpoint; stopping"
  exit 1
fi

for seed in 0 1; do
  echo ""
  echo "=========== finetune: rj-pretrain + rj-finetune, seed $seed ==========="
  "$PY" -m islvit.train --config configs/full263_v2.yaml --split-file "$SPLIT" \
    --init-from runs/cislr_mim_rj/pretrained.pt \
    --resolution-jitter 0.5 --select last --seed "$seed" \
    --tag "sd263_ssl_rj2_s$seed" 2>&1 | grep -av "$FILTER"

  "$PY" -m islvit.tta --run "runs/sd263_ssl_rj2_s$seed" --write-summary 2>&1 | grep -av "$FILTER"
done

echo ""
echo "=========== SUMMARY ==========="
"$PY" -m islvit.report 2>&1 | grep -av "$FILTER"
