#!/usr/bin/env bash
# Input-resolution / token-count ablation, run against the 128px cache.
#
# At 64px with patch16 each crop is only 4x4 = 16 tokens, which is very coarse for
# telling handshapes apart. These runs raise the token count while holding the
# recipe, the data and the split fixed, so any change is attributable to spatial
# resolution alone.
#
#   64px  -> 16 tokens, 0.82 GMACs
#   112px -> 49 tokens, 2.34 GMACs
#   128px -> 64 tokens, ~3.0 GMACs
set -u
PY="/c/Users/ADMIN/anaconda3/envs/islvit/python.exe"
export ISLVIT_CACHE=cache128

for SIZE in 64 112 128; do
  CFG="configs/_abl_${SIZE}.yaml"
  sed "s/^img_size: .*/img_size: ${SIZE}/" configs/include50_v2.yaml > "$CFG"
  echo ""
  echo "=================== img_size=${SIZE} ==================="
  "$PY" -m islvit.train --config "$CFG" --tag "abl${SIZE}__take-group" 2>&1 \
    | grep -av "UserWarning\|warnings.warn\|symlink\|Developer Mode\|huggingface_hub"
done

echo ""
echo "=================== SUMMARY ==================="
"$PY" -m islvit.report
