#!/bin/bash
#SBATCH --array=0-5
source activate pt2
coral_penalty_weight=(0.01 0.1 1 10 100 1000)

./run_syn.py \
    --methods mlp_coral \
    --batch_size 512 \
    --hidden_dim 256 \
    --lr 1e-3 \
    --steps 20000 \
    --n_reps=1 \
    --coral_penalty_weight ${coral_penalty_weight[$SLURM_ARRAY_TASK_ID]} \
    --out out/syn/coral_$SLURM_ARRAY_TASK_ID
