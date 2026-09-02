#!/usr/bin/env bash
# Is finetuning destroying the corpus-general features pretraining installed?
#
# Three interventions have now failed to move cross-corpus accuracy off chance:
#   CISLR labelled merge (p=0.45), iSign pretraining on 18k diverse clips
#   (1.3%), and supervised CISLR exposure (1.3%, and -2 pts on INCLUDE).
#
# Common factor: all three keep finetuning the spatial encoder at 0.1x LR for
# 250 epochs on 2,845 INCLUDE clips -- ample to overwrite general features with
# INCLUDE-specific ones. If that is the mechanism, freezing the encoder should
# RAISE cross-corpus accuracy while LOWERING INCLUDE accuracy. That trade is the
# signature to look for; a fall on both axes falsifies the hypothesis.
set -u
PY="/c/Users/ADMIN/anaconda3/envs/islvit/python.exe"
export ISLVIT_CACHE=cache128
FILTER="UserWarning\|warnings.warn\|symlink\|Developer Mode\|huggingface_hub"
TA=splits/full263__session-disjoint-foldval.csv
TB=splits/crosscorpus__cislr-test.csv

echo "waiting for the GPU to clear..."
while ps -W 2>/dev/null | grep -q "envs/islvit"; do sleep 30; done

# 0.0 = frozen encoder, 0.01 = 10x gentler than the current 0.1 default.
for scale in 0.0 0.01; do
  tag="frz${scale/./}_isign"
  echo ""
  echo "=========== backbone_lr_scale=$scale ==========="
  "$PY" -m islvit.train --config configs/full263_v2.yaml --split-file "$TA" \
    --init-from runs/isign_mim/pretrained.pt \
    --backbone-lr-scale "$scale" --resolution-jitter 0.5 --select last --seed 0 \
    --tag "$tag" 2>&1 | grep -av "$FILTER" | tail -4

  echo "-- T-B cross-corpus --"
  "$PY" -m islvit.eval --run "runs/$tag" --split-file "$TB" 2>&1 | grep -av "$FILTER" | grep -a TEST
done

echo ""
echo "reference: backbone_lr_scale=0.1 (current default) gave T-A 41.8% / T-B 1.3%"
echo "=========== DONE ==========="
