#!/usr/bin/env bash
# Deployed-objective fine-tune, all three wide seeds, fixed 150 epochs, final epoch.
export PYTHONPATH=/d/nn_dl_prac ISLVIT_CACHE=cache128_f32
PY=/c/Users/ADMIN/anaconda3/envs/islvit/python.exe
for s in 0 1 2; do
  $PY -m islvit.narrow --run runs/f16_262w_s$s --epochs 150 --seed $s 2>&1 | grep -v findfont
done
