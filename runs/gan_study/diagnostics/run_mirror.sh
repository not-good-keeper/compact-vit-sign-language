cd "$(dirname "$0")/../../.." || exit 1
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1
PYX=${PY:-python}
D=${GAN_WORK:-gan_work}/diag
echo "[$(date +%H:%M:%S)] crops L3 srgan"
$PYX -m islvit.data.crops --corpus include --cache-dir $D/crops_L3_srgan --crop-size 128 --frames 32 --restrict-to splits/_test472_paths.csv --workers 6 --degrade 180:40 --sr runs/sr_gan/generator.pt 2>&1 | grep -E "done|videos ->" | tail -n 2
echo "[$(date +%H:%M:%S)] crops L3 bicubic"
$PYX -m islvit.data.crops --corpus include --cache-dir $D/crops_L3_bicubic --crop-size 128 --frames 32 --restrict-to splits/_test472_paths.csv --workers 8 --degrade 180:40 2>&1 | grep -E "done|videos ->" | tail -n 2
for arm in srgan bicubic; do
  m=$D/mirror_L3_$arm; mkdir -p $m
  for f in hands hand_src pose done; do cp cache_lm_f32_test/$f.npy $m/; done
  echo "$D/crops_L3_$arm 32" > $m/SOURCE_CACHE
  cp $D/crops_L3_$arm/FRONTEND.json $m/FRONTEND.json
  for s in 0 1 2; do
    out=$m/score_s$s.json
    ISLVIT_CACHE=$D/crops_L3_$arm ISLVIT_CACHE_LM=$m $PYX -m islvit.mask50 --run release/isl_vit_tiny_lm_int4_s$s.pt --out $out 2>&1 | tail -n 1
  done
done
echo MIRRORDONE
