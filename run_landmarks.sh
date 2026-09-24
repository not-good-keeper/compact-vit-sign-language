#!/usr/bin/env bash
# Landmark stream: the f16_262w recipe exactly, plus --landmarks. Two seeds,
# paired against f16_262w_s0/s1. Waits for the narrow chain to free the GPU.
export PYTHONPATH=/d/nn_dl_prac ISLVIT_CACHE=cache128_f32
PY=/c/Users/ADMIN/anaconda3/envs/islvit/python.exe
until [ -f runs/f16_262w_s2_narrow/summary.json ]; do sleep 60; done
sleep 30
COMMON="--config configs/full263_v2.yaml --split-file splits/full263__session-disjoint-foldval.csv
  --init-from runs/isign_mim/pretrained.pt --resolution-jitter 0.5 --select last --epochs 1000
  --n-frames 16 --landmarks"
$PY -m islvit.train $COMMON --seed 0 --tag _lm_overfit --overfit-batch || exit 1
rm -rf runs/_lm_overfit
for s in 0 1; do
  $PY -m islvit.train $COMMON --seed $s --track-test --track-every 20 --resume --tag f16_262w_lm_s$s
  $PY -m islvit.mask50 --run runs/f16_262w_lm_s$s --out runs/f16_262w_lm_s$s/masked50.json
done
