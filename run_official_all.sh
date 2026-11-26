#!/usr/bin/env bash
# Both official-split benchmarks, seed by seed, so that seed 0 of each is ready
# first. Waits until no other islvit training process is alive, because two
# runs at once exceed the Windows commit limit.
cd "$(dirname "$0")" || exit 1
while powershell.exe -NoProfile -Command \
  "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'islvit\.(train|qat)' }) { exit 0 } else { exit 1 }"; do
  sleep 60
done
for s in 0 1 2; do
  bash run_official.sh include50 "$s" || exit 1
  bash run_official.sh full263 "$s" || exit 1
done
echo "official all done"
