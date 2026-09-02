#!/usr/bin/env bash
# Confirm the +9.9 pts from 8x longer training, and separate it from its confound.
#
# One 2000-epoch run at 50 words reached 61.6% (last-10 mean) against 51.7% at
# 250 epochs. If real this shifts every number in the project, so it needs a
# second seed before anything is rebuilt on it.
#
# It also has a confound: the cosine schedule stretches over 2000 epochs, so the
# long run is not "the same run, continued" -- it anneals more slowly throughout.
# Arm 3 separates duration from schedule by running 250 epochs' worth of
# annealing... which is not expressible with the current scheduler, so instead it
# tests the cheaper question: does 750 epochs already capture most of the gain?
# If 750 ~= 2000, the useful knob is modest and cheap; if 750 << 2000, duration
# matters in its own right.
#
# Arm 4 is the weight-decay arm the previous probe failed to run: --weight-decay
# now exists and was verified to reach the optimiser groups.
set -u
PY="/c/Users/ADMIN/anaconda3/envs/islvit/python.exe"
export ISLVIT_CACHE=cache128
FILTER="UserWarning\|warnings.warn\|symlink\|Developer Mode\|huggingface_hub"
SPLIT=splits/vocab50+cislr__session-disjoint.csv

echo "waiting for the GPU to clear..."
while ps -W 2>/dev/null | grep -q "envs/islvit"; do sleep 30; done

run () {  # tag, epochs, seed, weight_decay
  echo ""
  echo "=========== $1  (epochs=$2 seed=$3 wd=$4) ==========="
  "$PY" -m islvit.train --config configs/full263_v2.yaml --split-file "$SPLIT" \
    --init-from runs/isign_mim/pretrained.pt \
    --resolution-jitter 0.5 --select last --epochs "$2" --seed "$3" --weight-decay "$4" \
    --track-test --track-every 25 --tag "$1" 2>&1 | grep -av "$FILTER" | tail -3
  "$PY" -m islvit.tta --run "runs/$1" --write-summary 2>&1 | grep -av "$FILTER" | grep -aE "TTA"
}

run long2000_s1  2000 1 0.05     # does the +9.9 replicate on a second seed?
run long750_s0    750 0 0.05     # is most of the gain already there at 750?
run long2000_wd5 2000 0 0.5      # the weight-decay arm that never ran

echo ""
echo "=========== SUMMARY ==========="
"$PY" - <<'PYEOF'
import csv, json, math, statistics
from pathlib import Path
print(f"{'run':>16} {'epochs':>7} {'wd':>5} {'final(last-10)':>15} {'peak':>7} {'TTA':>7}")
for tag, ep, wd in (("grok_wd005",2000,0.05), ("long2000_s1",2000,0.05),
                    ("long750_s0",750,0.05), ("long2000_wd5",2000,0.5)):
    h = Path("runs")/tag/"history.csv"
    if not h.exists():
        continue
    v = [float(r["tracked_test_top1"]) for r in csv.DictReader(h.open(encoding="utf-8"))
         if r.get("tracked_test_top1") not in (None,"","nan")
         and not math.isnan(float(r["tracked_test_top1"]))]
    s = Path("runs")/tag/"summary.json"
    tta = json.loads(s.read_text(encoding="utf-8")).get("test_tta",{}).get("top1") if s.exists() else None
    print(f"{tag:>16} {ep:>7} {wd:>5} {statistics.mean(v[-10:]):>14.1%} {max(v):>7.1%} "
          f"{(f'{tta:.1%}' if tta else 'n/a'):>7}")
print("\n250-epoch reference at 50 words: 51.7% plain (2 seeds, sd 3.3), 56.2% TTA")
PYEOF
echo "=========== DONE ==========="
