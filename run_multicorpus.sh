#!/usr/bin/env bash
# Does ANY exposure to a second corpus fix the cross-corpus collapse?
#
# Established: INCLUDE-trained models score 31-42% on INCLUDE's held-out sessions
# and 0.3-1.5% on CISLR -- at or below chance. Pretraining on 18k diverse iSign
# clips did not help (1.3%), so unlabelled diversity alone is not the answer.
#
# This tests supervised exposure. Half the overlapping words contribute their
# CISLR clips to training (304 clips); the other half stay held out (305 clips).
# Split by WORD, so the test words are never seen in CISLR -- the model must
# transfer the domain, not memorise the signs.
#
#   baseline  INCLUDE only          -> ~1% on CISLR
#   this run  INCLUDE + 304 CISLR   -> ?
#
# 304 clips is only +8% data, so a large gain would mean domain exposure matters
# far more than data volume -- which is exactly the question.
set -u
PY="/c/Users/ADMIN/anaconda3/envs/islvit/python.exe"
export ISLVIT_CACHE=cache128
FILTER="UserWarning\|warnings.warn\|symlink\|Developer Mode\|huggingface_hub"
MIX=splits/crosscorpus__cislr-test-mix50.csv

echo "waiting for the GPU to clear..."
while ps -W 2>/dev/null | grep -q "envs/islvit"; do sleep 30; done

for seed in 0 1; do
  echo ""
  echo "=========== INCLUDE + CISLR(half the words), seed $seed ==========="
  "$PY" -m islvit.train --config configs/full263_v2.yaml --split-file "$MIX" \
    --init-from runs/isign_mim/pretrained.pt \
    --resolution-jitter 0.5 --select last --seed "$seed" \
    --tag "mix50_isign_s$seed" 2>&1 | grep -av "$FILTER"

  echo "-- held-out CISLR words (the transfer question) --"
  "$PY" -m islvit.eval --run "runs/mix50_isign_s$seed" --split-file "$MIX" 2>&1 | grep -av "$FILTER" | grep -a TEST
  echo "-- INCLUDE session-disjoint (did adding CISLR cost us anything?) --"
  "$PY" -m islvit.eval --run "runs/mix50_isign_s$seed" \
    --split-file splits/full263__session-disjoint-foldval.csv 2>&1 | grep -av "$FILTER" | grep -a TEST
done
echo ""
echo "=========== DONE ==========="
