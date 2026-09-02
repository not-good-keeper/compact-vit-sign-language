#!/usr/bin/env bash
# Does extended training produce delayed generalisation (grokking)?
#
# The precondition is met and was verified, not assumed: training accuracy is
# 100.0% on both the 262-word and 50-word models, with test at 41.8% and 54.0%.
# The ~0.9 final train loss is not underfitting -- it sits at the label-smoothing
# floor (0.879 for 262 classes), which is what perfect fit looks like under
# smoothing plus mixup. There is no optimisation headroom left; the entire
# remaining gap is generalisation.
#
# Grokking typically appears at 10-100x the memorisation time under strong weight
# decay. Memorisation here completes well inside 250 epochs, so 2,000 epochs (8x)
# is a reasonable probe. Two arms:
#
#   grok_wd05   weight_decay 0.05 (current)  -- does simply training longer help?
#   grok_wd50   weight_decay 0.5  (10x)      -- grokking is usually reported to
#                                               REQUIRE strong regularisation;
#                                               without this arm a null result
#                                               would be uninformative
#
# --track-test logs the test curve every 20 epochs for shape only. Selection
# stays on --select last and never reads it.
#
# Honest prior: grokking is mostly documented on small algorithmic tasks
# (modular arithmetic), rarely on real vision data. Expect a null. The value is
# that a null is then MEASURED rather than assumed, and the curve shape tells us
# whether longer training is worth any further GPU time at all.
set -u
PY="/c/Users/ADMIN/anaconda3/envs/islvit/python.exe"
export ISLVIT_CACHE=cache128
FILTER="UserWarning\|warnings.warn\|symlink\|Developer Mode\|huggingface_hub"
SPLIT=splits/vocab50+cislr__session-disjoint.csv

echo "waiting for the vocabulary sweep to finish..."
while ps -W 2>/dev/null | grep -q "envs/islvit"; do sleep 30; done

# The 50-word tier: 647 training clips, ~5 s/epoch, so 2,000 epochs is ~3 h per
# arm rather than ~11 h on the full vocabulary. Memorisation is just as complete.
for wd in 0.05 0.5; do
  tag="grok_wd${wd/./}"
  echo ""
  echo "=========== 2000 epochs, weight_decay=$wd ==========="
  "$PY" -m islvit.train --config configs/full263_v2.yaml --split-file "$SPLIT" \
    --init-from runs/isign_mim/pretrained.pt \
    --resolution-jitter 0.5 --select last --seed 0 --epochs 2000 \
    --track-test --track-every 20 \
    --tag "$tag" 2>&1 | grep -av "$FILTER" | tail -3
done

echo ""
echo "=========== TEST CURVE OVER TRAINING ==========="
"$PY" - <<'PYEOF'
import csv, math
from pathlib import Path
for tag in ("grok_wd005", "grok_wd05"):
    path = Path("runs") / tag / "history.csv"
    if not path.exists():
        continue
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8"))
            if r.get("tracked_test_top1") and not math.isnan(float(r["tracked_test_top1"]))]
    if not rows:
        continue
    print(f"\n{tag}: {len(rows)} tracked points")
    print(f"{'epoch':>6} {'train_loss':>11} {'test_top1':>10}")
    step = max(1, len(rows) // 12)
    for r in rows[::step]:
        print(f"{int(r['epoch']):>6} {float(r['train_loss']):>11.4f} {float(r['tracked_test_top1']):>9.1%}")
    best = max(float(r["tracked_test_top1"]) for r in rows)
    last = float(rows[-1]["tracked_test_top1"])
    early = max(float(r["tracked_test_top1"]) for r in rows[: len(rows)//4])
    print(f"  best {best:.1%}   final {last:.1%}   best in first quarter {early:.1%}")
    print(f"  -> {'LATE GAIN (investigate)' if best > early + 0.03 else 'no delayed transition'}")
PYEOF
echo ""
echo "=========== DONE ==========="
