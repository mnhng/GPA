#!/bin/bash

cat /dev/null > out_hps.txt

# echo "ERM, color in:" | tee -a out_hps.txt
# python -u hps_neo.py \
#   --color_input \
#   --penalty_anneal_iters=0 \
#   --penalty_weight=0.0 | tee -a out_hps.txt

echo "ERM, color regress:" | tee -a out_hps.txt
python -u hps_neo.py \
  --color_regress \
  --penalty_anneal_iters=0 \
  --penalty_weight=0.0 | tee -a out_hps.txt
