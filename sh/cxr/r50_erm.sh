#!/bin/bash
#SBATCH --array=0-4
#SBATCH --mem=20G
source activate pt

./run_cxr.py \
    --zlabels "Age" "AP/PA" "Sex" \
    --conf "conf/cxr.json" \
    --method erm \
    --batch_size 64 \
    --hidden_dim 256 \
    --seed $SLURM_ARRAY_TASK_ID \
    --lr 3e-5 \
    --steps 12000 \
    --out out/cxr_r50
