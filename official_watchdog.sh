#!/usr/bin/env bash
# Restart run_official_all.sh if it dies before finishing (for example on
# WinError 1455 when the commit limit runs out). Every stage is resume-safe, so a
# restart continues from the last 25-epoch checkpoint. Gives up after 6 restarts
# so a deterministic failure is reported instead of looping.
cd /d/nn_dl_prac || exit 1
restarts=0
while true; do
  if grep -q "official all done" official_all.log 2>/dev/null; then
    echo "WATCHDOG: all runs finished after $restarts restarts"; exit 0
  fi
  if ! ps -ef | grep -v grep | grep -q "run_official_all.sh"; then
    if [ $restarts -ge 6 ]; then
      echo "WATCHDOG: runner failed $restarts times; last log lines:"; tail -15 official_all.log; exit 2
    fi
    restarts=$((restarts + 1))
    echo "$(date '+%F %T') WATCHDOG restart $restarts" | tee -a official_watchdog.log
    tail -5 official_all.log >> official_watchdog.log
    sleep 90   # let crashed workers release their memory first
    nohup bash run_official_all.sh >> official_all.log 2>&1 &
  fi
  sleep 120
done
