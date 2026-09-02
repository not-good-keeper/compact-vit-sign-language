#!/usr/bin/env bash
# Full iSign pipeline: wait for download -> extract a diverse subset -> crop ->
# pretrain (resolution-matched, per the proven §11.5 recipe) -> finetune -> TTA.
#
# Only the finetune-time jitter recipe is used (§11.5 confirmed sd 0.1; the
# double-jitter pretraining variant was tested and REJECTED, sd 4.2 with a
# lower mean -- do not repeat that mistake here).
set -u
PY="/c/Users/ADMIN/anaconda3/envs/islvit/python.exe"
export ISLVIT_CACHE=cache128
FILTER="UserWarning\|warnings.warn\|symlink\|Developer Mode\|huggingface_hub"
SPLIT=splits/full263__session-disjoint-foldval.csv

echo "waiting for iSign video download to finish..."
until grep -q "ALL PARTS DOWNLOADED" /d/nn_dl_prac/isign_download.log 2>/dev/null; do
  sleep 120
done
echo "download complete."

echo ""
echo "=========== EXTRACT: diverse subset from the split zip ==========="
"$PY" -m islvit.data.isign_extract --target-clips 18000 --per-video-cap 4 2>&1 | tee /tmp/isign_extract_status.log | grep -av "$FILTER"

# Do not trust the extraction step's exit code alone -- it is piped through grep,
# which changes $? to grep's status, not python's. Check the actual output
# directory directly before doing anything irreversible to the source zip.
extracted_count=$(find iSign_videos -name "*.mp4" 2>/dev/null | wc -l)
if [ "$extracted_count" -eq 0 ]; then
  echo "extraction produced 0 clips -- NOT deleting the source zip. Fix the bug and rerun."
  exit 1
fi
echo "verified $extracted_count clips on disk before touching the zip"

# The 54 GB zip is fully consumed by extraction -- crops.py never touches it,
# only iSign_videos/. Freeing it BEFORE cropping matters: with the zip parts
# (54 GB) plus a growing ~40 GB crop cache both live at once, this machine's
# free space (32 GB at time of writing) is not enough -- crops.py would run out
# of disk partway through and leave a corrupt cache. Deleting the zip the moment
# it is no longer needed, rather than after cropping "for tidiness", is load-
# bearing here, not cosmetic.
echo "extraction done, freeing the 54 GB zip before cropping needs the headroom"
rm -rf iSign_raw
df -h /d | tail -1

echo ""
echo "=========== CROP: smoke test (300 clips) before committing hours to the full run ==========="
# The previous attempt died on an uncaught MediaPipe RuntimeError partway through
# these same 18,000 clips. The exception is now caught per-video (islvit/data/
# crops.py), but that fix is unverified against this specific real-world footage
# until it actually runs -- a 300-clip smoke test costs under a minute and would
# have caught the original bug immediately instead of after a full extraction.
"$PY" -m islvit.data.crops --corpus isign --cache-dir cache_isign --crop-size 128 --limit 300 2>&1 | grep -av "$FILTER"

# A non-zero count is not enough -- it already passed once at 50/300 (17%), a
# near-total failure that a bare ">0" check waved through. Require a coverage
# rate the finetuning recipe can actually use.
smoke_count=$(python -c "
import csv
try:
    with open('cache_isign/index.csv', encoding='utf-8') as f:
        print(sum(1 for r in csv.DictReader(f) if r['cached']=='1'))
except FileNotFoundError:
    print(0)
" 2>/dev/null || echo 0)
if [ "$smoke_count" -lt 270 ]; then
  echo "smoke test cached only $smoke_count/300 (need >=90%) -- stopping before the full run"
  exit 1
fi
echo "smoke test passed ($smoke_count/300 cached), proceeding to the full 18,000-clip extraction"
rm -rf cache_isign  # the smoke test's 300-row cache is the wrong shape for the full run; start clean

echo ""
echo "=========== CROP: MediaPipe hand+face extraction ==========="
"$PY" -m islvit.data.crops --corpus isign --cache-dir cache_isign --crop-size 128 2>&1 | grep -av "$FILTER"

# Same discipline as the zip deletion above, and for the same reason: a crash
# already destroyed 18,000 extracted clips once by deleting them unconditionally
# after crops.py died mid-run. Count real cached rows, not exit status -- the
# grep in the pipe already made exit status meaningless.
cached_count=$(python -c "
import csv
try:
    with open('cache_isign/index.csv', encoding='utf-8') as f:
        print(sum(1 for r in csv.DictReader(f) if r['cached']=='1'))
except FileNotFoundError:
    print(0)
" 2>/dev/null || echo 0)
# Same standard as the smoke test: a low but non-zero count already slipped
# through once (50/18000 = 0.3%) because this only checked for zero. Require a
# coverage rate a pretraining run can actually use, not just "something worked".
if [ "$cached_count" -lt 9000 ]; then
  echo "cropping produced only $cached_count/18000 cached clips (need >=50%) -- NOT deleting iSign_videos/. Fix the bug and rerun."
  exit 1
fi
echo "verified $cached_count cached clips before deleting the raw mp4s"
echo "cropping done, freeing the extracted mp4 subset (crops are cached, raw is redundant)"
rm -rf iSign_videos
df -h /d | tail -1

echo ""
echo "=========== PRETRAIN: masked reconstruction on iSign ==========="
"$PY" -m islvit.pretrain --cache cache_isign --tag isign_mim \
  --epochs 100 --batch-size 64 --min-genuine 0.25 2>&1 | grep -av "$FILTER"

if [ ! -f runs/isign_mim/pretrained.pt ]; then
  echo "iSign pretraining produced no checkpoint; stopping"
  exit 1
fi

for seed in 0 1; do
  echo ""
  echo "=========== finetune: iSign SSL init + resolution jitter, seed $seed ==========="
  "$PY" -m islvit.train --config configs/full263_v2.yaml --split-file "$SPLIT" \
    --init-from runs/isign_mim/pretrained.pt \
    --resolution-jitter 0.5 --select last --seed "$seed" \
    --tag "sd263_isign_rj_s$seed" 2>&1 | grep -av "$FILTER"

  "$PY" -m islvit.tta --run "runs/sd263_isign_rj_s$seed" --write-summary 2>&1 | grep -av "$FILTER"
done

echo ""
echo "=========== SUMMARY ==========="
"$PY" -m islvit.report 2>&1 | grep -av "$FILTER"
