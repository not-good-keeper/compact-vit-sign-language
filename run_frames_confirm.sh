#!/usr/bin/env bash
# Confirm the frame-count result and test whether it stacks with augmentation.
#
# cap_16f64 scored 66.7 % / 67.8 % TTA against grok_wd005's 61.9 % / 64.6 % on the
# SAME seed -- the largest single move measured in this project. But the 50-word
# tier has a seed spread of ~4.5 pts at this configuration (61.9 vs 57.4 across
# the two baseline seeds), so a one-seed +4.8 is exactly the size of result that
# has been overturned three times here already. It gets a second seed before it
# goes in the report.
#
# The augmentation bundle was a top-1 null (60.3 vs 59.7 plain, and -2.0 on TTA)
# but it moved top-5 by +4.0 with the two seeds agreeing to 0.0, and it halved the
# seed spread. That profile -- better ranking, tighter runs, flat top-1 -- is worth
# retesting at 16 frames rather than discarding, because it was measured at the
# frame count we now know was the binding constraint.
set -u
PY="/c/Users/ADMIN/anaconda3/envs/islvit/python.exe"
export ISLVIT_CACHE=cache128
FILTER="UserWarning\|warnings.warn\|symlink\|Developer Mode\|huggingface_hub\|TensorFlow\|oneDNN\|absl"
SPLIT=splits/vocab50+cislr__session-disjoint.csv
COMMON="--config configs/full263_v2.yaml --split-file $SPLIT --init-from runs/isign_mim/pretrained.pt
        --resolution-jitter 0.5 --select last --epochs 2000 --track-test --track-every 20 --n-frames 16"
STRONG="--mixup 0.8 --mixup-prob 1.0 --cutmix 1.0 --speed-jitter 0.3 --random-erasing 0.25"

run () {
  local tag="$1"; shift
  if [ -f "runs/$tag/summary.json" ]; then echo "=== $tag already done ==="; return; fi
  echo ""; echo "=========== $tag ==========="; date "+start %H:%M:%S"
  "$PY" -m islvit.train $COMMON --tag "$tag" "$@" 2>&1 | grep -av "$FILTER" | tail -4
  "$PY" -m islvit.tta --run "runs/$tag" --write-summary 2>&1 | grep -av "$FILTER" | grep -aE "TTA|delta"
  date "+done  %H:%M:%S"
}

run cap_16f64_s1 --seed 1                      # does the +4.8 survive a second seed?
run aug16_s0     --seed 0 $STRONG              # do the two levers stack?
run aug16_s1     --seed 1 $STRONG
