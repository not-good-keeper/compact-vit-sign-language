#!/usr/bin/env bash
# Confirm landmark augmentation: seed-1 pair (base vs aug 1.0), and strength 2.0 at seed 0.
export PYTHONPATH=/d/nn_dl_prac ISLVIT_CACHE=cache128_f32 ISLVIT_CACHE_LM=cache_lm_f32
PY=/c/Users/ADMIN/anaconda3/envs/islvit/python.exe
BASE="--config configs/full263_v2.yaml --split-file splits/full263__session-disjoint.csv
  --init-from runs/isign_mim/pretrained.pt --resolution-jitter 0.5 --select last
  --n-frames 16 --landmarks --val-every 50 --resume --epochs 500"
run() {  # tag seed extra-flags...
  local tag=$1 seed=$2; shift 2
  $PY -m islvit.train $BASE --seed $seed --tag $tag "$@" || return 1
  $PY -m islvit.mask50 --run runs/$tag --eval-file splits/full263__session-disjoint.csv \
    --eval-split val --out runs/$tag/val50.json
}
run dev_lm_base_s1  1
run dev_lm_aug_s1   1 --lm-aug 1.0
run dev_lm_aug2_s0  0 --lm-aug 2.0
