cd /d/nn_dl_prac
export PYTHONPATH=/d/nn_dl_prac PYTHONDONTWRITEBYTECODE=1
PYX=/c/Users/ADMIN/anaconda3/envs/islvit/python.exe
D=C:/isl_gan_work/diag
for v in lowconf bigroi multiroi all; do
  [ -f $D/L3srgan_$v/FRONTEND.json ] && continue
  echo "[$(date +%H:%M:%S)] extract $v"
  $PYX -m islvit.sr.diag_detector extract --variant $v --out $D/L3srgan_$v --sr runs/sr_gan/generator.pt --degrade 180:40 --workers 6 2>&1 | grep -E "clips/s|wrote|failed" | tail -n 2
done
echo LEVERS_EXTRACTED
