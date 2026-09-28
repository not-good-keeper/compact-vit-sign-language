#!/usr/bin/env bash
# Screening on the ORIGINAL split: train 1,957 clips, choose on the 888-clip
# validation set (154 of them deployed words). Test is not touched here.
# Each run differs from dev_lm_base_s0 by exactly one change.
export PYTHONPATH=/d/nn_dl_prac ISLVIT_CACHE=cache128_f32 ISLVIT_CACHE_LM=cache_lm_f32
PY=/c/Users/ADMIN/anaconda3/envs/islvit/python.exe
BASE="--config configs/full263_v2.yaml --split-file splits/full263__session-disjoint.csv
  --init-from runs/isign_mim/pretrained.pt --resolution-jitter 0.5 --select last
  --n-frames 16 --landmarks --seed 0 --val-every 50 --resume"
run() {  # tag epochs extra-flags...
  local tag=$1 epochs=$2; shift 2
  $PY -m islvit.train $BASE --epochs $epochs --tag $tag "$@" || return 1
  $PY -m islvit.mask50 --run runs/$tag --eval-file splits/full263__session-disjoint.csv \
    --eval-split val --out runs/$tag/val50.json
}
run dev_lm_base_s0     500
run dev_lm_aug_s0      500 --lm-aug 1.0
run dev_lm_interp_s0   500 --lm-interp
run dev_lm_vel_s0      500 --lm-velocity
run dev_lm_base1000_s0 1000
