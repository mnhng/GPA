#!/bin/bash
#SBATCH --array=0-5
source activate pt2
irm_lambda=(0.01 0.1 1 10 100 1000)

./run_syn.py \
    --methods mlp_irm \
    --batch_size 512 \
    --hidden_dim 256 \
    --lr 1e-3 \
    --steps 20000 \
    --n_reps=1 \
    --irm_lambda ${irm_lambda[$SLURM_ARRAY_TASK_ID]} \
    --out out/syn/irm_$SLURM_ARRAY_TASK_ID
