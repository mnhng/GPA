#!/bin/bash
#SBATCH --array=0-5
source activate pt2
dann_penalty_weight=(0.01 0.1 1 10 100 1000)

./run_syn.py \
    --methods mlp_dann \
    --batch_size 512 \
    --hidden_dim 256 \
    --lr 1e-3 \
    --steps 20000 \
    --n_reps=1 \
    --dann_penalty_weight ${dann_penalty_weight[$SLURM_ARRAY_TASK_ID]} \
    --out out/syn/dann_$SLURM_ARRAY_TASK_ID
