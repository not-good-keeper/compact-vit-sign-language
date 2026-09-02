#!/usr/bin/env bash
# Control: last-epoch selection on the ORIGINAL split (1,957 clips).
#
# sd263_foldval changed two things at once -- +45% training clips AND a switch
# from best-val to final-EMA checkpoint selection -- and scored 33.0% against the
# baseline's 22.1%. That +10.9 cannot be attributed without holding one variable
# still. Best-val was selecting on in-session accuracy while the report measures
# cross-session accuracy, so a badly-chosen checkpoint is a live explanation for
# a large share of the gap.
#
# This run isolates it: same 1,957 clips as the baseline, same ImageNet init,
# only the selection rule differs.
#
#   sd_full263      1957 clips, best-val   -> 22.1%
#   sd263_lastsel   1957 clips, last-EMA   -> this run
#   sd263_foldval   2845 clips, last-EMA   -> 33.0%
#
# The first gap is selection; the second is data.
set -u
PY="/c/Users/ADMIN/anaconda3/envs/islvit/python.exe"
export ISLVIT_CACHE=cache128
FILTER="UserWarning\|warnings.warn\|symlink\|Developer Mode\|huggingface_hub"

echo "waiting for training and TTA to clear the GPU..."
until [ -f runs/sd263_ssl_foldval/summary.json ]; do sleep 60; done
# TTA runs immediately after B; starting now would contend with it.
while ps -W 2>/dev/null | grep -q "envs/islvit"; do sleep 30; done

echo ""
echo "=================== CONTROL: last-epoch selection, original split ==================="
"$PY" -m islvit.train --config configs/full263_v2.yaml \
  --split-file splits/full263__session-disjoint.csv \
  --select last --tag sd263_lastsel 2>&1 | grep -av "$FILTER"

echo ""
echo "=================== SUMMARY ==================="
"$PY" -m islvit.report 2>&1 | grep -av "$FILTER"
