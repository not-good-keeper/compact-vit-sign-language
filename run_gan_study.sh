#!/usr/bin/env bash
# GAN comparative study (docs/GAN_COMPARATIVE_STUDY.md): a super-resolution GAN in
# place of bicubic upscaling for low-resolution video, against bicubic and against
# the same generator trained without the adversarial loss.
#
# Resume-safe: every step is skipped when its output exists, and training resumes
# from its checkpoint, so after a power cut just run this script again.
#
#   1  train the SR GAN: L1 stage (the no-GAN control), then the GAN stage
#   2  retrained arm: 360p training caches (bicubic, GAN) and three fine-tunes each
#   3  degraded test front ends (3 levels x 3 arms): score frozen INT4, frozen FP32
#      and the retrained models; detection and landmark agreement against clean
#   4  CISLR (natively 300 px): detection rates per arm
#   5  image quality on held-out patches, then the report
export PYTHONPATH=/d/nn_dl_prac PYTHONDONTWRITEBYTECODE=1
PY=/c/Users/ADMIN/anaconda3/envs/islvit/python.exe
export GAN_WORK=${GAN_WORK:-C:/isl_gan_work}
WORK=$GAN_WORK
OUT=runs/gan_study
SRL1=runs/sr_l1/generator.pt
SRGAN=runs/sr_gan/generator.pt
LEVELS="L1:360:60 L2:240:50 L3:180:40"
WORKERS=5
mkdir -p "$OUT"

need_space() {  # GiB that must be free on the work drive before a large build
  local free
  free=$($PY -c "import shutil; print(shutil.disk_usage('$WORK').free // 2**30)")
  if [ "$free" -lt "$1" ]; then echo "only ${free} GiB free on $WORK, need $1 -- stopping"; exit 1; fi
}
sr_flag() { case $1 in srl1) echo "--sr $SRL1";; srgan) echo "--sr $SRGAN";; *) echo "";; esac; }
extract() {  # cache-dir restrict-csv degrade arm [corpus]
  local cache=$1 paths=$2 degrade=$3 arm=$4 corpus=${5:-include}
  local deg=""; [ -n "$degrade" ] && deg="--degrade $degrade"
  $PY -m islvit.data.crops --corpus "$corpus" --cache-dir "$cache" --crop-size 128 --frames 32 \
      --restrict-to "$paths" --workers $WORKERS $deg $(sr_flag "$arm") || exit 1
  $PY -m islvit.data.landmarks --corpus "$corpus" --cache "$cache" --out "${cache}_lm" --frames 32 \
      --workers $WORKERS $deg $(sr_flag "$arm") || exit 1
}
score() {  # cache run out [landmark-cache, default <cache>_lm]
  [ -f "$3" ] && return
  mkdir -p "$(dirname "$3")"
  ISLVIT_CACHE=$1 ISLVIT_CACHE_LM=${4:-${1}_lm} $PY -m islvit.mask50 --run "$2" --out "$3" || exit 1
}

# ------------------------------------------------------------------ 1  SR GAN
[ -f "$WORK/hr_train.npy" ] || $PY -m islvit.sr.patches --split train --out "$WORK/hr_train.npy" || exit 1
[ -f "$SRL1" ] || $PY -m islvit.sr.train --stage l1 --bank "$WORK/hr_train.npy" --iters 100000 \
    --out runs/sr_l1 --resume || exit 1
# GAN stage: batch 16, 60k iterations (fixed before running; a GAN step costs ~25x an L1 step here).
[ -f "$SRGAN" ] || $PY -m islvit.sr.train --stage gan --bank "$WORK/hr_train.npy" --iters 60000 --batch 16 \
    --init "$SRL1" --out runs/sr_gan --resume || exit 1

# ------------------------------------------------------------------ 2  retrained arm
$PY -m islvit.sr.study train-paths "$WORK/train_paths.csv" || exit 1
for arm in bicubic srgan; do
  cache=$WORK/train360_$arm
  pending=0
  for s in 0 1 2; do [ -f "runs/ft360_${arm}_s$s/summary.json" ] || pending=1; done
  [ $pending = 0 ] && continue
  [ -f "$cache/crops.npy" ] || need_space 17
  extract "$cache" "$WORK/train_paths.csv" 360:60 $arm
  for s in 0 1 2; do
    [ -f "runs/ft360_${arm}_s$s/summary.json" ] && continue
    ISLVIT_CACHE=$cache ISLVIT_CACHE_LM=${cache}_lm $PY -m islvit.train --config configs/full263_v2.yaml \
      --split-file splits/full263__session-disjoint-foldval.csv --finetune-from "runs/f16_262w_lm_s$s/best.pt" \
      --lr 1e-4 --resolution-jitter 0.5 --select last --epochs 150 --n-frames 16 --landmarks \
      --seed $s --resume --tag "ft360_${arm}_s$s" || exit 1
  done
  $PY -m islvit.sr.study drop "$cache"
done
# The retrained models on clean video (no upscale happens at 1080p, so one front end serves all arms).
for arm in bicubic srgan; do
  for s in 0 1 2; do
    score cache128_f32_test "runs/ft360_${arm}_s$s" "$OUT/retrained/clean_${arm}_s$s.json" cache_lm_f32_test
  done
done

# ------------------------------------------------------------------ 3  degraded test front ends
for spec in $LEVELS; do
  level=${spec%%:*}; degrade=${spec#*:}
  for arm in bicubic srl1 srgan; do
    cache=$WORK/test_${level}_$arm
    if [ ! -f "$OUT/detection/${level}_$arm.json" ]; then
      [ -f "$cache/crops.npy" ] || need_space 4
      extract "$cache" splits/_test472_paths.csv "$degrade" $arm
    fi
    for s in 0 1 2; do
      score "$cache" "release/isl_vit_tiny_lm_int4_s$s.pt" "$OUT/frozen_int4/${level}_${arm}_s$s.json"
      score "$cache" "runs/f16_262w_lm_s$s" "$OUT/frozen_fp32/${level}_${arm}_s$s.json"
      case $arm in bicubic|srgan) score "$cache" "runs/ft360_${arm}_s$s" "$OUT/retrained/${level}_${arm}_s$s.json";; esac
    done
    [ -f "$OUT/detection/${level}_$arm.json" ] || $PY -m islvit.sr.study detection "$cache" "${cache}_lm" \
        "$OUT/detection/${level}_$arm.json" --reference cache128_f32_test cache_lm_f32_test || exit 1
    [ -f "$cache/crops.npy" ] && $PY -m islvit.sr.study drop "$cache"
  done
done

# ------------------------------------------------------------------ 4  CISLR detection
$PY -m islvit.sr.study cislr-paths "$WORK/cislr_paths.csv" || exit 1
for arm in bicubic srl1 srgan; do
  [ -f "$OUT/detection/cislr_$arm.json" ] && continue
  cache=$WORK/cislr_$arm
  [ -f "$cache/crops.npy" ] || need_space 4
  extract "$cache" "$WORK/cislr_paths.csv" "" $arm cislr
  $PY -m islvit.sr.study detection "$cache" "${cache}_lm" "$OUT/detection/cislr_$arm.json" || exit 1
  $PY -m islvit.sr.study drop "$cache"
done

# ------------------------------------------------------------------ 5  image quality + report
[ -f "$WORK/hr_heldout.npy" ] || $PY -m islvit.sr.patches --split heldout --crops 600 --frames-videos 75 \
    --out "$WORK/hr_heldout.npy" || exit 1
[ -f "$OUT/sr_quality.json" ] || $PY -m islvit.sr.evaluate --bank "$WORK/hr_heldout.npy" \
    --sr-l1 "$SRL1" --sr-gan "$SRGAN" --out "$OUT/sr_quality.json" || exit 1
$PY -m islvit.gan_report || exit 1
echo "=========== GAN STUDY DONE ==========="
