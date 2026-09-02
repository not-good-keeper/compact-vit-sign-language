#!/usr/bin/env bash
# Two questions, one batch, everything at the 2,000-epoch schedule that §11.8
# showed is the honest budget.
#
# Q1 -- AUGMENTATION. The model reaches ~100 % training accuracy, and the recipe
# was missing three of the standard defences: cutmix, random erasing, and any
# variation in signing tempo. mixup and EMA were already in. The "strong" arm is
# a bundle (mixup 0.8 applied always, cutmix, erasing, speed jitter) rather than
# one knob, because five 2.5 h runs to decompose a bundle that might do nothing
# is the wrong order to spend GPU time in. If the bundle wins, decompose it then.
#
# Q2 -- CAPACITY, REOPENED. "Data-limited, not capacity-limited" was concluded
# from 250-epoch runs, and §11.8 established those were undertrained by ~8 pts.
# The frames ablation is the tell: 4/8/12/16 frames scored 21.8/14.3/12.3/22.2 %,
# which is not a flat effect, it is noise. The cache holds 16 frames at 128 px
# while training uses 8 at 64 px, so this costs no re-extraction.
#
# The SSL checkpoint was pretrained at 8 frames x 64 px, so its position and time
# embeddings are interpolated on load (train.py::resize_position_embeddings).
# Without that, every capacity cell would silently lose the iSign initialisation
# and the ablation would measure pretraining, not capacity.
#
# BASELINE, already measured, not re-run here:
#   grok_wd005   8f/64px, baseline aug, seed 0 -> 61.9 % / 64.6 % TTA
#   long2000_s1  8f/64px, baseline aug, seed 1 -> 57.4 % / 62.1 % TTA
# Every cell below differs from those in exactly one respect.
set -u
PY="/c/Users/ADMIN/anaconda3/envs/islvit/python.exe"
export ISLVIT_CACHE=cache128
FILTER="UserWarning\|warnings.warn\|symlink\|Developer Mode\|huggingface_hub\|TensorFlow\|oneDNN\|absl"
SPLIT=splits/vocab50+cislr__session-disjoint.csv
INIT=runs/isign_mim/pretrained.pt
COMMON="--config configs/full263_v2.yaml --split-file $SPLIT --init-from $INIT
        --resolution-jitter 0.5 --select last --epochs 2000 --track-test --track-every 20"
STRONG="--mixup 0.8 --mixup-prob 1.0 --cutmix 1.0 --speed-jitter 0.3 --random-erasing 0.25"

run () {  # run <tag> <extra flags...>
  local tag="$1"; shift
  if [ -f "runs/$tag/summary.json" ]; then
    echo "=== $tag already done, skipping ==="
    return
  fi
  echo ""
  echo "=========== $tag ==========="
  date "+start %H:%M:%S"
  "$PY" -m islvit.train $COMMON --tag "$tag" "$@" 2>&1 | grep -av "$FILTER" | tail -4
  "$PY" -m islvit.tta --run "runs/$tag" --write-summary 2>&1 | grep -av "$FILTER" | grep -aE "TTA|delta"
  date "+done  %H:%M:%S"
}

# Cheapest and most directly comparable first, so the augmentation answer lands
# in ~5 h rather than after the whole batch. Two seeds, because the 50-word tier
# has sd 3.3 and a one-seed move of a few points would mean nothing.
run aug50_s0  --seed 0 $STRONG
run aug50_s1  --seed 1 $STRONG

# Capacity cells, baseline augmentation, seed 0 -- matched to grok_wd005.
run cap_16f64 --seed 0 --n-frames 16 --img-size 64
run cap_8f96  --seed 0 --n-frames 8  --img-size 96
run cap_16f96 --seed 0 --n-frames 16 --img-size 96

echo ""
echo "=========== SUMMARY ==========="
"$PY" - <<'PYEOF'
import json
from pathlib import Path
rows = [("grok_wd005", "8f/64  base  s0"), ("long2000_s1", "8f/64  base  s1"),
        ("aug50_s0", "8f/64  STRONG s0"), ("aug50_s1", "8f/64  STRONG s1"),
        ("cap_16f64", "16f/64 base  s0"), ("cap_8f96", "8f/96  base  s0"),
        ("cap_16f96", "16f/96 base  s0")]
print(f"  {'run':<14s} {'config':<17s} {'top1':>7s} {'top5':>7s} {'TTA':>7s}")
for tag, label in rows:
    path = Path("runs") / tag / "summary.json"
    if not path.exists():
        print(f"  {tag:<14s} {label:<17s} {'--':>7s}")
        continue
    d = json.loads(path.read_text(encoding="utf-8"))
    tta = d.get("test_tta", {}).get("top1")
    print(f"  {tag:<14s} {label:<17s} {d['test']['top1']:6.1%} {d['test']['top5']:6.1%} "
          f"{(f'{tta:6.1%}' if tta else '     --')}")
PYEOF
