#!/usr/bin/env bash
# GAN study, second GAN (v2): re-run only what depends on the SR-GAN weights.
#
# v1 (60k iterations from SR-L1) is archived under runs/sr_gan_v1, runs/ft360_srgan_v1_s*
# and runs/gan_study_v1/. Bicubic and SR-L1 results are unchanged and reused.
# v2 continues from the v1 generator with a fresh discriminator. It was chosen
# AFTER seeing v1's results, so it is not part of the pre-registered design; the
# report says so.
#
# After training, two lanes run at once to finish inside the time budget:
#   A  360p SR-GAN training cache -> three fine-tunes -> clean scoring
#   B  degraded test front ends (frozen INT4/FP32 scoring, detection) -> CISLR detection
# then the retrained models are scored on lane B's test caches, and the report is rebuilt.
# Resume-safe like run_gan_study.sh: rerun after any interruption.
export PYTHONPATH=/d/nn_dl_prac PYTHONDONTWRITEBYTECODE=1
PY=/c/Users/ADMIN/anaconda3/envs/islvit/python.exe
export GAN_WORK=${GAN_WORK:-C:/isl_gan_work}
WORK=$GAN_WORK
OUT=runs/gan_study
SRL1=runs/sr_l1/generator.pt
SRGAN=runs/sr_gan/generator.pt
GAN_ITERS=${GAN_ITERS:?set GAN_ITERS}
LEVELS="L1:360:60 L2:240:50 L3:180:40"
mkdir -p "$OUT"
stamp() { echo "[$(date +%H:%M:%S)] $*"; }

need_space() {
  local free
  free=$($PY -c "import shutil; print(shutil.disk_usage('$WORK').free // 2**30)")
  if [ "$free" -lt "$1" ]; then echo "only ${free} GiB free on $WORK, need $1 -- stopping"; exit 1; fi
}
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

# ------------------------------------------------------------------ 1  SR-GAN v2
stamp "train SR-GAN v2 ($GAN_ITERS iterations)"
[ -f "$SRGAN" ] || $PY -m islvit.sr.train --stage gan --bank "$WORK/hr_train.npy" --iters "$GAN_ITERS" --batch 16 \
    --lr 5e-5 --channels-last --init runs/sr_gan_v1/generator.pt --out runs/sr_gan --resume || exit 1

lane_a() {
  local cache=$WORK/train360_srgan pending=0
  for s in 0 1 2; do [ -f "runs/ft360_srgan_s$s/summary.json" ] || pending=1; done
  if [ $pending = 1 ]; then
    [ -f "$cache/crops.npy" ] || need_space 17
    stamp "A: 360p training cache"
    extract "$cache" "$WORK/train_paths.csv" 360:60 6
    for s in 0 1 2; do
      [ -f "runs/ft360_srgan_s$s/summary.json" ] && continue
      stamp "A: fine-tune seed $s"
      ISLVIT_CACHE=$cache ISLVIT_CACHE_LM=${cache}_lm $PY -m islvit.train --config configs/full263_v2.yaml \
        --split-file splits/full263__session-disjoint-foldval.csv --finetune-from "runs/f16_262w_lm_s$s/best.pt" \
        --lr 1e-4 --resolution-jitter 0.5 --select last --epochs 150 --n-frames 16 --landmarks \
        --seed $s --resume --tag "ft360_srgan_s$s" || exit 1
    done
  fi
  [ -f "$cache/crops.npy" ] && $PY -m islvit.sr.study drop "$cache"
  for s in 0 1 2; do
    score cache128_f32_test "runs/ft360_srgan_s$s" "$OUT/retrained/clean_srgan_s$s.json" cache_lm_f32_test
  done
  stamp "A: done"
}

lane_b() {
  for spec in $LEVELS; do
    level=${spec%%:*}; degrade=${spec#*:}
    cache=$WORK/test_${level}_srgan
    if [ ! -f "$OUT/detection/${level}_srgan.json" ]; then
      [ -f "$cache/crops.npy" ] || need_space 4
      stamp "B: test front end $level"
      extract "$cache" splits/_test472_paths.csv "$degrade" 5
    fi
    for s in 0 1 2; do
      score "$cache" "release/isl_vit_tiny_lm_int4_s$s.pt" "$OUT/frozen_int4/${level}_srgan_s$s.json"
      score "$cache" "runs/f16_262w_lm_s$s" "$OUT/frozen_fp32/${level}_srgan_s$s.json"
    done
    [ -f "$OUT/detection/${level}_srgan.json" ] || $PY -m islvit.sr.study detection "$cache" "${cache}_lm" \
        "$OUT/detection/${level}_srgan.json" --reference cache128_f32_test cache_lm_f32_test || exit 1
    # crops are kept until the retrained models have been scored on them (below)
  done
  if [ ! -f "$OUT/detection/cislr_srgan.json" ]; then
    cache=$WORK/cislr_srgan
    [ -f "$cache/crops.npy" ] || need_space 4
    stamp "B: CISLR"
    extract "$cache" "$WORK/cislr_paths.csv" "" 5 cislr
    $PY -m islvit.sr.study detection "$cache" "${cache}_lm" "$OUT/detection/cislr_srgan.json" || exit 1
    $PY -m islvit.sr.study drop "$cache"
  fi
  stamp "B: done"
}

if [ "${LANES:-AB}" = "A" ]; then
  # lane A alone; run_gan_v2_rest.sh runs lane B (gated on A's fine-tuning), then the rest
  lane_a > gan_v2_laneA.log 2>&1 || { echo "lane A failed (gan_v2_laneA.log)"; exit 1; }
  exit 0
fi
( lane_a ) > gan_v2_laneA.log 2>&1 &
pid_a=$!
( lane_b ) > gan_v2_laneB.log 2>&1 &
pid_b=$!
wait $pid_a; rc_a=$?
wait $pid_b; rc_b=$?
[ $rc_a = 0 ] || { echo "lane A failed (gan_v2_laneA.log)"; exit 1; }
[ $rc_b = 0 ] || { echo "lane B failed (gan_v2_laneB.log)"; exit 1; }

# ------------------------------------------------------------------ retrained models on the degraded test sets
for spec in $LEVELS; do
  level=${spec%%:*}
  cache=$WORK/test_${level}_srgan
  for s in 0 1 2; do score "$cache" "runs/ft360_srgan_s$s" "$OUT/retrained/${level}_srgan_s$s.json"; done
  [ -f "$cache/crops.npy" ] && $PY -m islvit.sr.study drop "$cache"
done

# ------------------------------------------------------------------ image quality + report
stamp "image quality + report"
[ -f "$OUT/sr_quality.json" ] || $PY -m islvit.sr.evaluate --bank "$WORK/hr_heldout.npy" \
    --sr-l1 "$SRL1" --sr-gan "$SRGAN" --out "$OUT/sr_quality.json" || exit 1
$PY -m islvit.gan_report || exit 1
stamp "=========== GAN v2 STUDY DONE ==========="
