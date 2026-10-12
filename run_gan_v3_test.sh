#!/usr/bin/env bash
# Test-set study for the two v3 arms that bracket the trade-off seen on the validation sessions:
#   sr3_gan_task (GAN + hand losses) and sr3_l1_task (hand losses, no GAN).
# Frozen INT4 and FP32 models, 472 test clips, 180p / 240p / 360p, same front ends and scorer as
# run_gan_study.sh (bicubic, SR-L1 and the earlier GAN are reused from runs/gan_study/).
# Front ends run in parallel lanes (LANES x WORKERS MediaPipe workers; every worker also holds a CUDA
# context, so LANES*WORKERS is bounded by GPU memory). Resume-safe: finished steps are skipped.
export PYTHONPATH=/d/nn_dl_prac PYTHONDONTWRITEBYTECODE=1
PY=/c/Users/ADMIN/anaconda3/envs/islvit/python.exe
W=C:/isl_gan_work; OUT=runs/gan_study
LANES=${LANES:-3}; WORKERS=${WORKERS:-3}
export ISLVIT_SR_FP16=1   # half-precision upscaling: 1.8x faster, <= 1/255 pixel difference (see islvit/sr/upscaler.py)
LEVELS=${LEVELS:-"L3 L2"}
job() {  # level degrade arm
  level=$1; deg=$2; arm=$3
  cache=$W/test_${level}_$arm; gen=runs/$arm/generator.pt
  stamp() { echo "[$(date +%H:%M:%S)] ${level}_$arm: $*"; }
  if [ ! -f $OUT/detection/${level}_$arm.json ]; then
    stamp extract
    $PY -m islvit.data.crops --corpus include --cache-dir $cache --crop-size 128 --frames 32 \
        --restrict-to splits/_test472_paths.csv --workers $WORKERS --degrade $deg --sr $gen > $cache.crops.log 2>&1 || { stamp "crops FAILED"; return 1; }
    $PY -m islvit.data.landmarks --corpus include --cache $cache --out ${cache}_lm --frames 32 \
        --workers $WORKERS --degrade $deg --sr $gen > $cache.lm.log 2>&1 || { stamp "landmarks FAILED"; return 1; }
  fi
  for s in 0 1 2; do
    for fam in int4 fp32; do
      if [ $fam = int4 ]; then run=release/isl_vit_tiny_lm_int4_s$s.pt; else run=runs/f16_262w_lm_s$s; fi
      o=$OUT/frozen_$fam/${level}_${arm}_s$s.json
      [ -f $o ] && continue
      stamp "score $fam s$s"
      ISLVIT_CACHE=$cache ISLVIT_CACHE_LM=${cache}_lm $PY -m islvit.mask50 --run $run --out $o > /dev/null 2>&1 || { stamp "score FAILED"; return 1; }
    done
  done
  [ -f $OUT/detection/${level}_$arm.json ] || $PY -m islvit.sr.study detection $cache ${cache}_lm $OUT/detection/${level}_$arm.json \
      --reference cache128_f32_test cache_lm_f32_test > /dev/null 2>&1
  [ -f $cache/crops.npy ] && $PY -m islvit.sr.study drop $cache > /dev/null 2>&1
  stamp done
}
export -f job; export PY W OUT WORKERS PYTHONPATH PYTHONDONTWRITEBYTECODE
mkdir -p $OUT/frozen_int4 $OUT/frozen_fp32 $OUT/detection
printf '%s\n' "L3 180:40 sr3_gan_task" "L3 180:40 sr3_l1_task" "L2 240:50 sr3_gan_task" "L2 240:50 sr3_l1_task" \
              "L1 360:60 sr3_gan_task" "L1 360:60 sr3_l1_task" | xargs -P $LANES -L 1 bash -c 'job $0 $1 $2'
echo "=========== v3 TEST STUDY DONE ==========="
