#!/bin/bash
# Phase G queue: launch each job when >= 12 GB GPU memory is free; up to 3 G jobs at once.
cd /home/neymankv
mkdir -p runs/followup/G/logs
JOBS=()
for d in ruler4k ruler8k loogle; do
  for m in full random v1_prop v1_prop_noweight random_rw expected_attention; do JOBS+=("$m 0.75 $d 20"); done
done
for d in ruler4k ruler8k loogle; do JOBS+=("kvzip 0.75 $d 20"); done
for d in ruler4k ruler8k loogle; do
  for m in random v1_prop v1_prop_noweight random_rw expected_attention; do JOBS+=("$m 0.9 $d 10"); done
done
for j in "${JOBS[@]}"; do
  set -- $j
  out="runs/followup/G/${3}_${1}_r${2}.stats.json"
  [ -f "$out" ] && continue
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    ng=$(pgrep -fc "longdoc/run_g.py")
    if [ "$free" -ge 12000 ] && [ "$ng" -lt 3 ]; then break; fi
    sleep 30
  done
  echo "$(date -u) G launch $j (free ${free} MiB)" >> runs/followup/queue.log
  nohup /home/venv_kvpress/bin/python longdoc/run_g.py $1 $2 $3 $4 > "runs/followup/G/logs/${3}_${1}_r${2}.log" 2>&1 &
  sleep 90   # let the model load before measuring free memory again
done
wait
echo "$(date -u) G queue done" >> runs/followup/queue.log
