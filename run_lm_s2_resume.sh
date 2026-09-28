#!/usr/bin/env bash
# Resume the third landmark seed (crashed at epoch ~235), then give it the same
# downstream treatment as seeds 0 and 1: clean-test scoring, fixed-epoch QAT,
# clean-test scoring of the INT4 model. Training and QAT read the training
# cache; only scoring uses the fresh-detector test caches.
export PYTHONPATH=/d/nn_dl_prac
PY=/c/Users/ADMIN/anaconda3/envs/islvit/python.exe
TRAIN="ISLVIT_CACHE=cache128_f32"
CLEAN="ISLVIT_CACHE=cache128_f32_test ISLVIT_CACHE_LM=cache_lm_f32_test"
env $TRAIN $PY -m islvit.train --config configs/full263_v2.yaml \
  --split-file splits/full263__session-disjoint-foldval.csv \
  --init-from runs/isign_mim/pretrained.pt --resolution-jitter 0.5 --select last --epochs 1000 \
  --n-frames 16 --landmarks --seed 2 --track-test --track-every 20 --resume --tag f16_262w_lm_s2 || exit 1
env $CLEAN $PY -m islvit.mask50 --run runs/f16_262w_lm_s2 --out runs/f16_262w_lm_s2/masked50_clean.json
env $TRAIN $PY -m islvit.qat --run runs/f16_262w_lm_s2 --epochs 300 --group 128 --seed 2 --tag f16_262w_lm_s2_qat4 || exit 1
env $CLEAN $PY -m islvit.mask50 --run runs/f16_262w_lm_s2_qat4 --out runs/f16_262w_lm_s2_qat4/masked50_clean.json
