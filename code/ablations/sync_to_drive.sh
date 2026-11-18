#!/bin/bash
# Mirror ablation checkpoints (and finished tables) to Google Drive every 5 minutes,
# so a run can be resumed after the Colab VM is reset. Skips a round while any
# last.pt was modified in the last 60 s (i.e. may still be being written).
#   usage: nohup code/ablations/sync_to_drive.sh SRC DST [RESULTS_DIR] > sync.log 2>&1 &
SRC=${1:-/content/checkpoints/ablations_9155}
DST=${2:-/content/drive/MyDrive/inkrepair/tmp/ablations_9155}
RES=${3:-/content/inkcloser/paper/v1/results_ablations_9155}
mkdir -p "$DST"
while true; do
  busy=0
  for f in "$SRC"/*/last.pt; do
    [ -f "$f" ] && [ $(( $(date +%s) - $(stat -c %Y "$f") )) -lt 60 ] && busy=1
  done
  if [ $busy = 0 ]; then
    rsync -a "$SRC/" "$DST/"
    [ -d "$RES" ] && rsync -a "$RES" "$(dirname "$DST")/"
  fi
  sleep 300
done
