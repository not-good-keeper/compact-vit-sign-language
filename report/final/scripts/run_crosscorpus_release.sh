#!/usr/bin/env bash
# Score the three released INT4 landmark models on the CISLR cross-corpus test
# (609 clips, 218 shared words). The repository's earlier cross-corpus numbers
# (crosscorpus_eval.log) were measured on pixel-only 8-frame checkpoints; the
# shipped landmark model had not been scored on another corpus.
#
# Needs the local CISLR crop cache (cache_cislr_f32/) and CISLR videos. The
# landmark cache is written to cache_lm_cislr_f32/ (local, not committed, like
# the other caches). Results: report/final/data/crosscorpus_release_s{0,1,2}.json
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-/c/Users/ADMIN/anaconda3/envs/islvit/python.exe}
export PYTHONPATH=.

if [ ! -f cache_lm_cislr_f32/done.npy ]; then
  "$PY" -m islvit.data.landmarks --cache cache_cislr_f32 --out cache_lm_cislr_f32 --frames 32 \
    --corpus cislr --split-file splits/crosscorpus__cislr-test.csv --split test --workers 4
fi

# islvit.data.dataset only loads landmarks for rows marked corpus=include, so the
# 609 CISLR test rows are copied with that column relabelled and the CISLR crop
# cache is passed as the "include" cache. Nothing else about the rows changes.
SPLIT=report/final/data/crosscorpus_cislr_test_as_include.csv
"$PY" report/final/scripts/relabel_cislr_split.py "$SPLIT"

for s in 0 1 2; do
  ISLVIT_CACHE=cache_cislr_f32 ISLVIT_CACHE_LM=cache_lm_cislr_f32 \
    "$PY" -m islvit.mask50 --run release/isl_vit_tiny_lm_int4_s$s.pt \
      --eval-file "$SPLIT" --eval-split test \
      --out report/final/data/crosscorpus_release_s$s.json
done
