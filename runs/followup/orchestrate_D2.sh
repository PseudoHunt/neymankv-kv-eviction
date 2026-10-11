#!/bin/bash
# Phase D2: reserves on the recall-cue probe (20) and 20 MATH500 (accuracy check), one job at a time.
cd /home/neymankv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for m in random_reserve v2_reserve random_oracle v2_oracle; do
  for set in forced math; do
    echo "$(date -u) D2 launch $m $set" >> runs/followup/queue.log
    python -m poc.run --method $m --budget 256 --sets $set --n 20 --slots 32 --out followup --clock followup --tag _D2 \
      > runs/followup/logs/D2_${m}_${set}.log 2>&1
    echo "$(date -u) D2 $m $set exit $?" >> runs/followup/queue.log
  done
done
echo "$(date -u) D2 done" >> runs/followup/queue.log
