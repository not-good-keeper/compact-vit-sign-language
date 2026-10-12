#!/usr/bin/env bash
# SR-GAN v3: a generator trained for what the classifier needs (docs/GAN_IMPROVEMENT_PLAN.md s7).
#
# Four arms, all from one 2.0 M-parameter generator, all with the hand-crop batches and
# H.264/JPEG degradations, all fixed in advance (iterations, weights) -- no selection on results:
#   sr3_l1        pixel L1 only                     (the no-GAN control; also the init of the rest)
#   sr3_gan_task  + VGG + RaGAN + hand-proxy losses (the main model)
#   sr3_gan       + VGG + RaGAN, no hand losses     (ablation: what the hand losses add)
#   sr3_l1_task   + hand-proxy losses, no GAN       (ablation: what the adversarial part adds)
# After each arm: development metrics with real MediaPipe on 200 clips from the held-out
# *validation* sessions at 180p and 240p. The 472 test clips are not touched here.
# Resume-safe: rerun after any interruption.
export PYTHONPATH=/d/nn_dl_prac PYTHONDONTWRITEBYTECODE=1
PY=/c/Users/ADMIN/anaconda3/envs/islvit/python.exe
W=C:/isl_gan_work
COMMON="--bank $W/hr_train_v3.npy --hands $W/hands_train_v3.npy --codec-prob 0.5 --channels-last --workers 10
        --hand-proxy runs/hand_proxy/proxy.pt --val-hands $W/hp_val.npz --resume"
stamp() { echo "[$(date +%H:%M:%S)] $*"; }

dev() {  # arm generator
  for spec in L3:180:40 L2:240:50; do
    level=${spec%%:*}; deg=${spec#*:}
    out=$W/dev/${level}_$1
    [ -f $out/FRONTEND.json ] && continue
    $PY -m islvit.sr.diag_detector extract --variant base --split val --n 200 --ref-cache cache128_f32 \
        --ref-lm cache_lm_f32 --workers 6 --out $out --degrade $deg --sr $2 2>&1 | grep -E "wrote|failed" || exit 1
  done
}

stamp "sr3_l1"
[ -f runs/sr3_l1/generator.pt ] || $PY -m islvit.sr.train --stage l1 $COMMON --g-features 96 --g-convs 24 \
    --iters 100000 --batch 32 --out runs/sr3_l1 || exit 1
dev sr3_l1 runs/sr3_l1/generator.pt

stamp "sr3_gan_task"
[ -f runs/sr3_gan_task/generator.pt ] || $PY -m islvit.sr.train --stage gan $COMMON --init runs/sr3_l1/generator.pt \
    --gan-loss ragan --w-hand 2 --w-hand-feat 0.2 --iters 60000 --batch 16 --out runs/sr3_gan_task || exit 1
dev sr3_gan_task runs/sr3_gan_task/generator.pt

stamp "sr3_gan"
[ -f runs/sr3_gan/generator.pt ] || $PY -m islvit.sr.train --stage gan $COMMON --init runs/sr3_l1/generator.pt \
    --gan-loss ragan --iters 60000 --batch 16 --out runs/sr3_gan || exit 1
dev sr3_gan runs/sr3_gan/generator.pt

stamp "sr3_l1_task"
[ -f runs/sr3_l1_task/generator.pt ] || $PY -m islvit.sr.train --stage l1 $COMMON --init runs/sr3_l1/generator.pt \
    --lr 1e-4 --w-hand 2 --w-hand-feat 0.2 --iters 60000 --batch 16 --out runs/sr3_l1_task || exit 1
dev sr3_l1_task runs/sr3_l1_task/generator.pt

# L2 baselines for the comparison table (L3 ones exist from the baseline run)
for arm in bicubic:"" srl1:runs/sr_l1/generator.pt srgan_v2:runs/sr_gan/generator.pt; do
  name=${arm%%:*}; g=${arm#*:}
  [ -f $W/dev/L2_$name/FRONTEND.json ] && continue
  $PY -m islvit.sr.diag_detector extract --variant base --split val --n 200 --ref-cache cache128_f32 --ref-lm cache_lm_f32 \
      --workers 6 --out $W/dev/L2_$name --degrade 240:50 ${g:+--sr $g} 2>&1 | grep -E "wrote|failed" || exit 1
done
$PY -m islvit.sr.diag_detector stats $W/dev/L3_* $W/dev/L2_* --ref-cache cache128_f32 --ref-lm cache_lm_f32 \
    --json runs/gan_study/diagnostics/dev_v3.json
stamp "=========== v3 TRAINING + DEV DONE ==========="
