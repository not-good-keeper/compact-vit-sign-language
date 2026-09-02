#!/usr/bin/env bash
# Accuracy vs vocabulary size -- the product-scope decision, made from data.
#
# 262 words is INCLUDE's choice, not a requirement. A small vocabulary recognised
# reliably beats a large one recognised badly, and the effect is known to be
# large: on an older recipe 262 words scored 29.3% against 50 words at 46.5%.
#
# Per-word support is held constant across tiers (each kept word keeps all its
# clips), so the variable that moves is the number of classes to separate.
# All tiers include the overlapping CISLR clips, which §11.6 measured as free.
#
# Two seeds per tier: this project's noise floor is sd ~1.9 pts and single seeds
# have produced misleading readings four times.
set -u
PY="/c/Users/ADMIN/anaconda3/envs/islvit/python.exe"
export ISLVIT_CACHE=cache128
FILTER="UserWarning\|warnings.warn\|symlink\|Developer Mode\|huggingface_hub"

echo "waiting for the GPU to clear..."
while ps -W 2>/dev/null | grep -q "envs/islvit"; do sleep 30; done

for words in 137 100 50 30; do
  SPLIT="splits/vocab${words}+cislr__session-disjoint.csv"
  for seed in 0 1; do
    tag="vocab${words}_s${seed}"
    echo ""
    echo "=========== ${words} words, seed ${seed} ==========="
    "$PY" -m islvit.train --config configs/full263_v2.yaml --split-file "$SPLIT" \
      --init-from runs/isign_mim/pretrained.pt \
      --resolution-jitter 0.5 --select last --seed "$seed" \
      --tag "$tag" 2>&1 | grep -av "$FILTER" | tail -3
    "$PY" -m islvit.tta --run "runs/$tag" --write-summary 2>&1 | grep -av "$FILTER" | grep -aE "TTA|delta"
  done
done

echo ""
echo "=========== ACCURACY vs VOCABULARY ==========="
"$PY" - <<'PYEOF'
import json, statistics
from pathlib import Path
print(f"{'words':>6} {'seeds':>5} {'top1':>16} {'top1+TTA':>16} {'balanced':>9}")
for words in (137, 100, 50, 30):
    got = []
    for seed in (0, 1):
        path = Path(f"runs/vocab{words}_s{seed}/summary.json")
        if path.exists():
            got.append(json.loads(path.read_text(encoding="utf-8")))
    if not got:
        continue
    t1 = [s["test"]["top1"] for s in got]
    bal = [s["test"]["balanced"] for s in got]
    tta = [s["test_tta"]["top1"] for s in got if "test_tta" in s]
    spread = f" +-{statistics.stdev(t1)*100:.1f}" if len(t1) > 1 else ""
    tta_s = f"{statistics.mean(tta):.1%}" if tta else "n/a"
    print(f"{words:>6} {len(got):>5} {statistics.mean(t1):>10.1%}{spread:>6} {tta_s:>16} {statistics.mean(bal):>9.1%}")
print("\nreference: 262 words = 42.1% top-1, ~44% with TTA")
PYEOF
echo "=========== DONE ==========="
