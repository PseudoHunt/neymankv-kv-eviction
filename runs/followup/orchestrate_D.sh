#!/bin/bash
# Wait for full seed-1 to finish, run D1, then the last Phase A job (neymankv_v2 seed 2).
cd /home/neymankv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "method full --budget 256 --sets math,math_x,gsm,gsm_x --seed 1 " >/dev/null; do sleep 20; done
echo "$(date -u) launching D1" >> runs/followup/queue.log
python -m poc.d1_retention > runs/followup/logs/D1.log 2>&1
echo "$(date -u) D1 exit $?" >> runs/followup/queue.log
echo "$(date -u) launching A neymankv_v2 seed 2" >> runs/followup/queue.log
python -m poc.run --method neymankv_v2 --budget 256 --sets math,math_x,gsm,gsm_x --seed 2 --slots 32 --out followup --clock followup --tag _A > runs/followup/logs/A_neymankv_v2_B256_math-math_x-gsm-gsm_x_s2.log 2>&1
echo "$(date -u) A v2 s2 exit $?" >> runs/followup/queue.log
