#!/usr/bin/env bash
# Honest QAT on both landmark seeds: 300 epochs fixed in advance, final epoch.
export PYTHONPATH=/d/nn_dl_prac ISLVIT_CACHE=cache128_f32
PY=/c/Users/ADMIN/anaconda3/envs/islvit/python.exe
for s in 0 1; do
  $PY -m islvit.qat --run runs/f16_262w_lm_s$s --epochs 300 --group 128 --seed $s --tag f16_262w_lm_s${s}_qat4
done
