#!/bin/bash
#SBATCH --array=0-5
source activate pt2
group_dro_step_size=(0.001 0.01 0.1 1 10 100)

./run_syn.py \
    --methods mlp_dro \
    --batch_size 512 \
    --hidden_dim 256 \
    --lr 1e-3 \
    --steps 20000 \
    --n_reps=1 \
    --group_dro_step_size ${group_dro_step_size[$SLURM_ARRAY_TASK_ID]} \
    --out out/syn/dro_$SLURM_ARRAY_TASK_ID
