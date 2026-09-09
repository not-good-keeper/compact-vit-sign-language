#!/usr/bin/env bash
# Three levers past 75 %, each isolated against the same control.
#
# CONTROL: f16_clean_s0 / f16_clean_s1 -- 16 frames from the 32-frame cache, the
# leak-free 571-clip split, 2,000 epochs, 8-frame SSL init interpolated to 16.
# Two seeds, 69.9 % and 70.3 % plain (spread 0.4), 73.3 % / 72.0 % TTA.
#
#   f16_mim16_s0  SSL pretrained NATIVELY at 16 frames instead of stretched from 8.
#                 Every run to date has used time embeddings interpolated 8 -> 16;
#                 pretraining was worth ~+10 pts originally, so an approximation
#                 sitting on top of it is the best-odds thing left.
#   f16_full_s0   647 training clips instead of 571, recovering the 76 CISLR clips
#                 that were reserved for T-B. Costs the cross-corpus test set, which
#                 has never measured above ~1.6 %. Free in compute: +13 % data.
#   f16_long4k_s0 4,000 epochs instead of 2,000. The 2,000 figure was tuned at 8
#                 frames on the old cache; 250 -> 2,000 was worth +7.9 and was still
#                 climbing at the end.
#
# One variable each, all against the same control and the same 472-clip test set,
# because three of this project's conclusions have been overturned by follow-ups
# that turned out to have moved two things at once.
set -u
PY="/c/Users/ADMIN/anaconda3/envs/islvit/python.exe"
export ISLVIT_CACHE=cache128_f32
export ISLVIT_CACHE_CISLR=cache_cislr_f32   # 32-frame CISLR: matches INCLUDE's headroom
FILTER="UserWarning\|warnings.warn\|symlink\|Developer Mode\|huggingface_hub\|TensorFlow\|oneDNN\|absl"
CLEAN=splits/vocab50clean__session-disjoint.csv
FULL=splits/vocab50full__session-disjoint.csv
BASE="--config configs/full263_v2.yaml --resolution-jitter 0.5 --select last
      --track-test --track-every 20 --resume --seed 0 --n-frames 16"

echo "waiting for 16-frame SSL pretraining to finish..."
until [ -f runs/isign_mim16/summary.json ] || [ -f runs/isign_mim16/.done ]; do
  if ! ps -W 2>/dev/null | grep -q "envs/islvit"; then
    echo "no process alive; pretraining stopped early -- check runs/isign_mim16/history.csv"
    break
  fi
  sleep 300
done

run () {
  local tag="$1"; shift
  if [ -f "runs/$tag/summary.json" ]; then echo "=== $tag already done ==="; return; fi
  echo ""; echo "=========== $tag ==========="; date "+start %H:%M:%S"
  "$PY" -m islvit.train $BASE --tag "$tag" "$@" 2>&1 | grep -av "$FILTER" | tail -4
  "$PY" -m islvit.tta --run "runs/$tag" --write-summary 2>&1 | grep -av "$FILTER" | grep -aE "TTA|delta"
  date "+done  %H:%M:%S"
}

run f16_mim16_s0  --split-file "$CLEAN" --init-from runs/isign_mim16/pretrained.pt --epochs 2000
run f16_full_s0   --split-file "$FULL"  --init-from runs/isign_mim/pretrained.pt   --epochs 2000
run f16_long4k_s0 --split-file "$CLEAN" --init-from runs/isign_mim/pretrained.pt   --epochs 4000

echo ""
echo "=========== AGAINST THE CONTROL ==========="
"$PY" - <<'PYEOF'
import json
from pathlib import Path
rows = [("f16_clean_s0", "control (s0)"), ("f16_clean_s1", "control (s1)"),
        ("f16_mim16_s0", "native 16f SSL"), ("f16_full_s0", "+76 CISLR clips"),
        ("f16_long4k_s0", "4000 epochs")]
base = []
print(f"  {'run':<15s} {'what':<17s} {'top1':>7s} {'top5':>7s} {'TTA':>7s} {'vs ctrl':>8s}")
for tag, what in rows:
    p = Path("runs") / tag / "summary.json"
    if not p.exists():
        print(f"  {tag:<15s} {what:<17s}      --"); continue
    d = json.loads(p.read_text(encoding="utf-8"))
    top1 = d["test"]["top1"]
    tta = d.get("test_tta", {}).get("top1")
    if tag.startswith("f16_clean"):
        base.append(top1)
    delta = f"{100*(top1 - sum(base)/len(base)):+7.1f}" if base and not tag.startswith("f16_clean") else "       -"
    print(f"  {tag:<15s} {what:<17s} {top1:6.1%} {d['test']['top5']:6.1%} "
          f"{(f'{tta:6.1%}' if tta else '     --')} {delta}")
print("\n  noise floor for a difference of two runs: 2.7 pts")
PYEOF
