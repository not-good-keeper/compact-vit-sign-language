cd "$(dirname "$0")/../../.." || exit 1
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1
PYX=${PY:-python}
D=${GAN_WORK:-gan_work}/diag
for v in lowconf bigroi multiroi all; do
  [ -f $D/L3srgan_$v/FRONTEND.json ] && continue
  echo "[$(date +%H:%M:%S)] extract $v"
  $PYX -m islvit.sr.diag_detector extract --variant $v --out $D/L3srgan_$v --sr runs/sr_gan/generator.pt --degrade 180:40 --workers 6 2>&1 | grep -E "clips/s|wrote|failed" | tail -n 2
done
echo LEVERS_EXTRACTED
