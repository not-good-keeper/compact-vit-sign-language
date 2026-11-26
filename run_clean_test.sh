#!/usr/bin/env bash
# Deployment-consistent test set: the 472 held-out clips re-extracted with a fresh
# MediaPipe detector per video (crops.reset_detectors), crops then landmarks.
cd "$(dirname "$0")" || exit 1
export PYTHONPATH=$PWD
PY=${PY:-python}
$PY -m islvit.data.crops --cache-dir cache128_f32_test --crop-size 128 --frames 32 \
  --restrict-to splits/_test472_paths.csv --workers 5 || exit 1
$PY -m islvit.data.landmarks --cache cache128_f32_test --out cache_lm_f32_test --frames 32 --workers 5 || exit 1
