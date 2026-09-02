#!/usr/bin/env bash
# Iteration 3: spend the validation set as training data.
#
# Val shares every recording session with train, so it measures in-session
# recognition -- not the cross-session generalisation the test set measures. It
# cannot steer checkpoint selection usefully (SSL pretraining moved val -0.8 and
# test +7.6), so its 888 clips are worth more as training data: +45% on 1,957.
# Selection becomes the final EMA checkpoint under a cosine schedule ending at
# zero LR.
#
# Two runs, because "more data" and "SSL pretraining" must not be confounded --
# the same mistake that made the labelled-merge result unreadable until McNemar
# separated it out.
set -u
PY="/c/Users/ADMIN/anaconda3/envs/islvit/python.exe"
export ISLVIT_CACHE=cache128
FILTER="UserWarning\|warnings.warn\|symlink\|Developer Mode\|huggingface_hub"

echo "=================== A: ImageNet init + folded val (isolates the data) ==================="
"$PY" -m islvit.train --config configs/full263_v2.yaml \
  --split-file splits/full263__session-disjoint-foldval.csv \
  --select last --tag sd263_foldval 2>&1 | grep -av "$FILTER"

echo ""
echo "=================== B: SSL init + folded val (headline candidate) ==================="
"$PY" -m islvit.train --config configs/full263_v2.yaml \
  --split-file splits/full263__session-disjoint-foldval.csv \
  --init-from runs/cislr_mim/pretrained.pt \
  --select last --tag sd263_ssl_foldval 2>&1 | grep -av "$FILTER"

echo ""
echo "=================== SUMMARY ==================="
"$PY" -m islvit.report 2>&1 | grep -av "$FILTER"
