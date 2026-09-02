#!/usr/bin/env bash
# Is the frame curve still climbing past 16?
#
# 8 -> 16 frames was worth +6.4 pts over two seeds (§11.10), the largest single
# move in the project, and 16 was the cache ceiling rather than a measured
# optimum. cache128_f32 now holds 32 frames at matched detection quality
# (left-hand genuine 91.1 % vs 91.3 %, right 85.2 % vs 85.4 %), so the ceiling is
# gone and the question is answerable.
#
# All three cells read the SAME 32-frame cache and differ only in n_frames. Using
# the old 16-frame cache as the control would have confounded frame count with
# how much temporal jitter the sampler has to choose from -- a cell reading 16 of
# 32 sees a different augmentation distribution than one reading 16 of 16, and
# that difference is part of what more cached frames buys.
#
# The split is the LEAK-FREE 50-word tier: 571 INCLUDE clips, no CISLR. cache_cislr
# holds only 16 frames, so the 76 CISLR clips in the old split would have been
# temporally upsampled in the 24f and 32f cells and not in the control. The test
# set is byte-identical to the one behind every other 50-word number here.
set -u
PY="/c/Users/ADMIN/anaconda3/envs/islvit/python.exe"
export ISLVIT_CACHE=cache128_f32
FILTER="UserWarning\|warnings.warn\|symlink\|Developer Mode\|huggingface_hub\|TensorFlow\|oneDNN\|absl"
SPLIT=splits/vocab50clean__session-disjoint.csv
COMMON="--config configs/full263_v2.yaml --split-file $SPLIT --init-from runs/isign_mim/pretrained.pt
        --resolution-jitter 0.5 --select last --epochs 2000 --track-test --track-every 20 --seed 0 --resume"

run () {
  local tag="$1"; shift
  if [ -f "runs/$tag/summary.json" ]; then echo "=== $tag already done ==="; return; fi
  echo ""; echo "=========== $tag ==========="; date "+start %H:%M:%S"
  "$PY" -m islvit.train $COMMON --tag "$tag" "$@" 2>&1 | grep -av "$FILTER" | tail -4
  "$PY" -m islvit.tta --run "runs/$tag" --write-summary 2>&1 | grep -av "$FILTER" | grep -aE "TTA|delta"
  date "+done  %H:%M:%S"
}

run f16_clean_s0 --n-frames 16    # control, matched cache
run f24_clean_s0 --n-frames 24
run f32_clean_s0 --n-frames 32

echo ""
echo "=========== FRAME CURVE ==========="
"$PY" - <<'PYEOF'
import json
from pathlib import Path
print(f"  {'frames':>7s} {'top1':>7s} {'top5':>7s} {'TTA':>7s}")
for tag, frames in (("f16_clean_s0", 16), ("f24_clean_s0", 24), ("f32_clean_s0", 32)):
    p = Path("runs") / tag / "summary.json"
    if not p.exists():
        print(f"  {frames:>7d}      --"); continue
    d = json.loads(p.read_text(encoding="utf-8"))
    tta = d.get("test_tta", {}).get("top1")
    print(f"  {frames:>7d} {d['test']['top1']:6.1%} {d['test']['top5']:6.1%} "
          f"{(f'{tta:6.1%}' if tta else '     --')}")
PYEOF
