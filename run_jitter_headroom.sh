#!/usr/bin/env bash
# Is it frame count, or is it temporal jitter headroom?
#
# Reading the sweep so far by frames fed to the model gives an incoherent curve:
# 8 -> 59.7 %, 16 -> 69.9 %, 24 -> 69.7 %. Reading it by the RATIO of cached frames
# to sampled frames orders it perfectly, and orders the TTA gain with it:
#
#   16 of 16 (1.00x)  66.7 %  TTA +1.1
#   24 of 32 (1.33x)  69.7 %  TTA +0.6
#   16 of 32 (2.00x)  69.9 %  TTA +3.4
#
# The mechanism is that sampling k frames from k leaves the sampler no choice, so
# train-time jitter and test-time phase offsets both degenerate to no-ops. That
# predicts f32 (32 of 32, 1.00x) lands near the 16-of-16 case despite seeing twice
# the frames -- which the cell now running will settle.
#
# These two runs separate the hypotheses properly:
#
#   f16_clean_s1  a second seed on the headline. 73.3 % TTA is one run against one
#                 run, and the difference-of-runs noise floor here is 2.7 pts.
#   f8_clean_s0   8 of 32 = 4.0x headroom, the MOST headroom tested, on the FEWEST
#                 frames. Frame count predicts this is the worst cell (8 frames
#                 scored 59.7 % off the 16-frame cache); headroom predicts it is
#                 competitive. The two hypotheses disagree sharply, which is the
#                 point -- and it is also the cheapest cell to run.
set -u
PY="/c/Users/ADMIN/anaconda3/envs/islvit/python.exe"
export ISLVIT_CACHE=cache128_f32
FILTER="UserWarning\|warnings.warn\|symlink\|Developer Mode\|huggingface_hub\|TensorFlow\|oneDNN\|absl"
SPLIT=splits/vocab50clean__session-disjoint.csv
COMMON="--config configs/full263_v2.yaml --split-file $SPLIT --init-from runs/isign_mim/pretrained.pt
        --resolution-jitter 0.5 --select last --epochs 2000 --track-test --track-every 20 --resume"

echo "waiting for the frame sweep to finish..."
until [ -f runs/f32_clean_s0/summary.json ]; do
  if ! ps -W 2>/dev/null | grep -q "envs/islvit"; then
    echo "no training process alive; f32 stalled -- run run_frames32.sh to resume it first"
    exit 1
  fi
  sleep 120
done

run () {
  local tag="$1"; shift
  if [ -f "runs/$tag/summary.json" ]; then echo "=== $tag already done ==="; return; fi
  echo ""; echo "=========== $tag ==========="; date "+start %H:%M:%S"
  "$PY" -m islvit.train $COMMON --tag "$tag" "$@" 2>&1 | grep -av "$FILTER" | tail -4
  "$PY" -m islvit.tta --run "runs/$tag" --write-summary 2>&1 | grep -av "$FILTER" | grep -aE "TTA|delta"
  date "+done  %H:%M:%S"
}

run f16_clean_s1 --seed 1 --n-frames 16
run f8_clean_s0  --seed 0 --n-frames 8

echo ""
echo "=========== HEADROOM TABLE ==========="
"$PY" - <<'PYEOF'
import json
from pathlib import Path
cells = [("f8_clean_s0", 8), ("f16_clean_s0", 16), ("f16_clean_s1", 16),
         ("f24_clean_s0", 24), ("f32_clean_s0", 32)]
print(f"  {'run':<14s} {'frames':>6s} {'headroom':>9s} {'top1':>7s} {'TTA':>7s} {'gain':>6s}")
for tag, frames in cells:
    p = Path("runs") / tag / "summary.json"
    if not p.exists():
        print(f"  {tag:<14s} {frames:>6d}      --"); continue
    d = json.loads(p.read_text(encoding="utf-8"))
    top1, tta = d["test"]["top1"], d.get("test_tta", {}).get("top1")
    gain = f"{100*(tta-top1):+5.1f}" if tta else "   --"
    print(f"  {tag:<14s} {frames:>6d} {32/frames:8.2f}x {top1:6.1%} "
          f"{(f'{tta:6.1%}' if tta else '     --')} {gain}")
PYEOF
