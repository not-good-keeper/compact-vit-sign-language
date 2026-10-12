#!/usr/bin/env bash
# Validation-session checks for the v3 arms, run on the spare CPU while the next arm trains.
# run_gan_v3.sh does the same checks one after another, which stalls the GPU queue; this watcher
# does them as soon as each generator exists. It uses 4 MediaPipe workers (the machine has 12
# cores and training takes ~40 %), and the SR inference it needs is a few ms per frame.
# Nothing here feeds back into training, so quality is unaffected; only wall-clock changes.
export PYTHONPATH=/d/nn_dl_prac PYTHONDONTWRITEBYTECODE=1
PY=/c/Users/ADMIN/anaconda3/envs/islvit/python.exe
W=C:/isl_gan_work
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
ext() {  # outdir degrade [generator]
  $PY -m islvit.sr.diag_detector extract --variant base --split val --n 200 --ref-cache cache128_f32 \
      --ref-lm cache_lm_f32 --workers 4 --out "$1" --degrade "$2" ${3:+--sr "$3"} 2>&1 | grep -E "wrote|failed" || exit 1
}
# 1. the 240p baselines for the comparison table
for arm in bicubic: srl1:runs/sr_l1/generator.pt srgan_v2:runs/sr_gan/generator.pt; do
  name=${arm%%:*}; g=${arm#*:}
  [ -f $W/dev/L2_$name/done.npy ] && continue
  stamp "L2_$name"; ext $W/dev/L2_$name 240:50 "$g"
done
# 2. each v3 arm, as soon as its generator exists
for arm in sr3_gan_task sr3_gan sr3_l1_task; do
  until [ -f runs/$arm/generator.pt ]; do sleep 60; done
  for spec in L3:180:40 L2:240:50; do
    level=${spec%%:*}; deg=${spec#*:}
    [ -f $W/dev/${level}_$arm/done.npy ] && continue
    stamp "${level}_$arm"; ext $W/dev/${level}_$arm $deg runs/$arm/generator.pt
  done
done
stamp "=========== DEV CHECKS DONE ==========="
