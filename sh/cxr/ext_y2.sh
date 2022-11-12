#!/bin/bash
#SBATCH --array=0-4
source activate pt
I=$SLURM_ARRAY_TASK_ID

./run_cxr.py \
    --zlabels "Age" "AP/PA" "Sex" \
    --conf "conf/cxr_ext.json" \
    --methods r50_y2 \
    --batch_size 64 \
    --hidden_dim 256 \
    --seed $I \
    --lr 3e-5 \
    --steps 12000 \
    --n_reps 1 \
    --coral_penalty_weight 0.01 \
    --dann_penalty_weight 0.01 \
    --group_dro_step_size 0.001 \
    --irm_lambda 0.01 \
    --out out/ecxr_r50_y2/seed${I}
