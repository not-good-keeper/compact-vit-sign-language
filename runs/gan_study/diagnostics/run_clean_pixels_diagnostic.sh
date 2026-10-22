# Intermediate landmark caches (lm_<level>_<arm>) live under $GAN_WORK/diag/mix; the scored results are the res_clean_pix_*.json files beside this script.
cd /d/nn_dl_prac
for lvl in L2 L3; do for arm in bicubic srl1 srgan; do for s in 0 1 2; do
  out="${GAN_WORK:-C:/isl_gan_work}/diag/mix/res_clean_pix_${lvl}_${arm}_s$s.json"; [ -f "$out" ] && continue
  ISLVIT_CACHE=cache128_f32_test ISLVIT_CACHE_LM="${GAN_WORK:-C:/isl_gan_work}/diag/mix/lm_${lvl}_${arm}" /c/Users/ADMIN/anaconda3/envs/islvit/python.exe -m islvit.mask50 --run release/isl_vit_tiny_lm_int4_s$s.pt --out "$out" 2>&1 | tail -n 1
done; done; done
echo MIXDONE
