#!/usr/bin/env bash
# The released recipe (landmarks, 16 frames, iSign init, 1,000 epochs, final
# weights, then 300 epochs of INT4 QAT) trained on INCLUDE's own shipped split,
# so the result sits on the same test clips as the published INCLUDE numbers.
# Training uses the official train rows only; the official val rows are scored
# every 50 epochs for the record and never select a checkpoint.
#
#   bash run_official.sh full263 "0 1 2"    # 262 words, 816 official test clips
#   bash run_official.sh include50 "0 1 2"  # INCLUDE-50, 191 official test clips
#
# One run at a time: two concurrent runs exhaust the Windows commit limit
# (WinError 1455). Resume-safe: rerun the same command after an interruption.
set -u
cd "$(dirname "$0")" || exit 1
export PYTHONPATH=$PWD ISLVIT_CACHE=cache128_f32
PY=${PY:-python}
BENCH=$1
SEEDS=${2:-0 1 2}
SPLIT=splits/${BENCH}__official.csv
COMMON="--config configs/full263_v2.yaml --split-file $SPLIT
  --init-from runs/isign_mim/pretrained.pt --resolution-jitter 0.5 --select last --epochs 1000
  --n-frames 16 --landmarks --val-every 50"
for s in $SEEDS; do
  TAG=official_${BENCH}_lm_s$s
  [ -f runs/$TAG/summary.json ] || $PY -m islvit.train $COMMON --seed $s --resume --tag $TAG || exit 1
  [ -f runs/$TAG/official_eval.json ] || \
    $PY -m islvit.mask50 --run runs/$TAG --split-file $SPLIT --out runs/$TAG/official_eval.json || exit 1
  [ -f runs/${TAG}_qat4/summary.json ] || \
    $PY -m islvit.qat --run runs/$TAG --epochs 300 --group 128 --seed $s --tag ${TAG}_qat4 \
      --deployed-split $SPLIT || exit 1
  [ -f runs/${TAG}_qat4/official_eval.json ] || \
    $PY -m islvit.mask50 --run runs/${TAG}_qat4 --split-file $SPLIT \
      --out runs/${TAG}_qat4/official_eval.json || exit 1
done
echo "official $BENCH done"
