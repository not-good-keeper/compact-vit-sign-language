#!/usr/bin/env bash
# Lever 2: self-supervised pretraining on ALL of CISLR, then finetune.
#
# Lever 1 (run_lever2's predecessor, tag sd263_cislr) can only use the 493 CISLR
# clips whose glosses match an INCLUDE class. This uses all ~6,100 that pass the
# detection filter -- including the ~5,600 whose words INCLUDE does not contain --
# because masked reconstruction needs no labels. That is 12x more data than the
# supervised merge sees, which is why this could outrank lever 1.
#
# Waits for lever 1 rather than running beside it: both are GPU-bound and would
# only slow each other down.
set -u
PY="/c/Users/ADMIN/anaconda3/envs/islvit/python.exe"
export ISLVIT_CACHE=cache128

echo "waiting for sd263_cislr to finish..."
until [ -f runs/sd263_cislr/summary.json ]; do
  if ! ps -W 2>/dev/null | grep -q python; then
    echo "lever 1 process vanished without a summary; continuing anyway"
    break
  fi
  sleep 60
done

echo ""
echo "=================== PRETRAIN (masked crops, all CISLR) ==================="
"$PY" -m islvit.pretrain \
  --cache cache_cislr --tag cislr_mim \
  --epochs 100 --batch-size 64 --min-genuine 0.25 2>&1 \
  | grep -av "UserWarning\|warnings.warn\|symlink\|Developer Mode\|huggingface_hub"

if [ ! -f runs/cislr_mim/pretrained.pt ]; then
  echo "pretraining produced no checkpoint; stopping"
  exit 1
fi

# Two finetunes, because they answer different questions:
#   ssl        -- does self-supervised pretraining beat ImageNet init on its own?
#   ssl+cislr  -- does it stack with lever 1's extra labelled clips?
echo ""
echo "=================== FINETUNE A: ssl init, INCLUDE-263 only ==================="
"$PY" -m islvit.train --config configs/full263_v2.yaml \
  --split-file splits/full263__session-disjoint.csv \
  --init-from runs/cislr_mim/pretrained.pt \
  --tag sd263_ssl 2>&1 \
  | grep -av "UserWarning\|warnings.warn\|symlink\|Developer Mode\|huggingface_hub"

echo ""
echo "=================== FINETUNE B: ssl init + merged CISLR clips ==================="
"$PY" -m islvit.train --config configs/full263_v2.yaml \
  --split-file splits/full263__session-disjoint+cislr.csv \
  --init-from runs/cislr_mim/pretrained.pt \
  --tag sd263_ssl_cislr 2>&1 \
  | grep -av "UserWarning\|warnings.warn\|symlink\|Developer Mode\|huggingface_hub"

echo ""
echo "=================== SUMMARY ==================="
"$PY" -m islvit.report
