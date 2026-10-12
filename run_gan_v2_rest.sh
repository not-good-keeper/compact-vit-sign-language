#!/usr/bin/env bash
# Recovery for run_gan_study_v2.sh after lane B crashed (CUDA out of memory: five SR workers
# plus lane A's six, each holding a CUDA context). Lane A is still running in the original
# script, so this one only does lane B, waits for lane A, then finishes the study.
#
# Lane B starts once lane A has moved from landmark extraction (many CUDA workers) to
# fine-tuning (one process), and uses fewer workers. Resume-safe.
export PYTHONPATH=/d/nn_dl_prac PYTHONDONTWRITEBYTECODE=1
PY=/c/Users/ADMIN/anaconda3/envs/islvit/python.exe
WORK=${GAN_WORK:-C:/isl_gan_work}
OUT=runs/gan_study
SRL1=runs/sr_l1/generator.pt
SRGAN=runs/sr_gan/generator.pt
LEVELS="L1:360:60 L2:240:50 L3:180:40"
WORKERS_B=${WORKERS_B:-4}
stamp() { echo "[$(date +%H:%M:%S)] $*"; }

extract() {  # cache-dir restrict-csv degrade workers [corpus]
  local cache=$1 paths=$2 degrade=$3 workers=$4 corpus=${5:-include}
  local deg=""; [ -n "$degrade" ] && deg="--degrade $degrade"
  $PY -m islvit.data.crops --corpus "$corpus" --cache-dir "$cache" --crop-size 128 --frames 32 \
      --restrict-to "$paths" --workers "$workers" $deg --sr "$SRGAN" || exit 1
  $PY -m islvit.data.landmarks --corpus "$corpus" --cache "$cache" --out "${cache}_lm" --frames 32 \
      --workers "$workers" $deg --sr "$SRGAN" || exit 1
}
score() {
  [ -f "$3" ] && return
  mkdir -p "$(dirname "$3")"
  ISLVIT_CACHE=$1 ISLVIT_CACHE_LM=${4:-${1}_lm} $PY -m islvit.mask50 --run "$2" --out "$3" || exit 1
}

stamp "waiting for lane A to start fine-tuning"
until grep -aq "A: fine-tune seed 0" gan_v2_laneA.log; do sleep 30; done
stamp "lane A is fine-tuning; starting lane B with $WORKERS_B workers"

for spec in $LEVELS; do
  level=${spec%%:*}; degrade=${spec#*:}
  cache=$WORK/test_${level}_srgan
  if [ ! -f "$OUT/detection/${level}_srgan.json" ]; then
    [ -f "$cache/crops.npy" ] && [ -f "${cache}_lm/done.npy" ] || { rm -rf "$cache" "${cache}_lm"; extract "$cache" splits/_test472_paths.csv "$degrade" "$WORKERS_B"; }
  fi
  for s in 0 1 2; do
    score "$cache" "release/isl_vit_tiny_lm_int4_s$s.pt" "$OUT/frozen_int4/${level}_srgan_s$s.json"
    score "$cache" "runs/f16_262w_lm_s$s" "$OUT/frozen_fp32/${level}_srgan_s$s.json"
  done
  [ -f "$OUT/detection/${level}_srgan.json" ] || $PY -m islvit.sr.study detection "$cache" "${cache}_lm" \
      "$OUT/detection/${level}_srgan.json" --reference cache128_f32_test cache_lm_f32_test || exit 1
  stamp "B: $level done"
done
if [ ! -f "$OUT/detection/cislr_srgan.json" ]; then
  cache=$WORK/cislr_srgan
  rm -rf "$cache" "${cache}_lm"
  extract "$cache" "$WORK/cislr_paths.csv" "" "$WORKERS_B" cislr
  $PY -m islvit.sr.study detection "$cache" "${cache}_lm" "$OUT/detection/cislr_srgan.json" || exit 1
  $PY -m islvit.sr.study drop "$cache"
fi
stamp "B: done; waiting for lane A"

until grep -aq "A: done" gan_v2_laneA.log; do
  grep -aq "Traceback\|Error" gan_v2_laneA.log && ! pgrep -f "islvit.train" >/dev/null 2>&1 && { stamp "lane A seems to have failed"; }
  sleep 60
done

for spec in $LEVELS; do
  level=${spec%%:*}
  cache=$WORK/test_${level}_srgan
  for s in 0 1 2; do score "$cache" "runs/ft360_srgan_s$s" "$OUT/retrained/${level}_srgan_s$s.json"; done
  [ -f "$cache/crops.npy" ] && $PY -m islvit.sr.study drop "$cache"
done

stamp "image quality + report"
[ -f "$OUT/sr_quality.json" ] || $PY -m islvit.sr.evaluate --bank "$WORK/hr_heldout.npy" \
    --sr-l1 "$SRL1" --sr-gan "$SRGAN" --out "$OUT/sr_quality.json" || exit 1
$PY -m islvit.gan_report || exit 1
stamp "=========== GAN v2 STUDY DONE ==========="
