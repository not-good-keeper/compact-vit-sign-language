#!/usr/bin/env bash
# Score every leak-free checkpoint on BOTH test sets.
#
# T-A  INCLUDE session-disjoint (n=1010) -- comparable to all prior work
# T-B  CISLR cross-corpus     (n= 609) -- the real-world proxy
#
# The key question: an ImageNet-init model scored 31% on T-A and 1.5% on T-B.
# Does pretraining on 18k diverse iSign clips -- which is exactly what should
# teach corpus-invariant features -- close that gap?
#
# Only iSign-pretrained and ImageNet-init checkpoints appear here. Anything
# pretrained on CISLR saw T-B's clips during pretraining and cannot be scored on
# it honestly.
set -u
PY="/c/Users/ADMIN/anaconda3/envs/islvit/python.exe"
export ISLVIT_CACHE=cache128
FILTER="UserWarning\|warnings.warn\|symlink\|Developer Mode\|huggingface_hub"
TA=splits/full263__session-disjoint-foldval.csv
TB=splits/crosscorpus__cislr-test.csv

echo "waiting for the iSign finetunes to finish..."
until [ -f runs/sd263_isign_rj_s1/summary.json ]; do
  if ! ps -W 2>/dev/null | grep -q "envs/islvit"; then
    echo "training stopped; scoring whatever exists"
    break
  fi
  sleep 60
done

for run in sd263_isign_rj_s0 sd263_isign_rj_s1 sd263_rj_s0 sd263_rj_s1 sd263_foldval; do
  [ -f "runs/$run/best.pt" ] || continue
  echo ""
  echo "=========== $run ==========="
  echo "-- T-A (INCLUDE session-disjoint) --"
  "$PY" -m islvit.eval --run "runs/$run" --split-file "$TA" 2>&1 | grep -av "$FILTER" | grep -a "TEST"
  echo "-- T-B (CISLR cross-corpus) --"
  "$PY" -m islvit.eval --run "runs/$run" --split-file "$TB" 2>&1 | grep -av "$FILTER" | grep -a "TEST"
done
echo ""
echo "=========== DONE ==========="
