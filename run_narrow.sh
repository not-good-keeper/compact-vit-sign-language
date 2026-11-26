#!/usr/bin/env bash
# Deployed-objective fine-tune, all three wide seeds, fixed 150 epochs, final epoch.
cd "$(dirname "$0")" || exit 1
export PYTHONPATH=$PWD ISLVIT_CACHE=cache128_f32
PY=${PY:-python}
for s in 0 1 2; do
  $PY -m islvit.narrow --run runs/f16_262w_s$s --epochs 150 --seed $s 2>&1 | grep -v findfont
done
