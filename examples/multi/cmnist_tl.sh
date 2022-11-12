#!/bin/bash
#SBATCH --array=0-29
#SBATCH --mem=5G
source activate gpa
METHODS=(gpa erm2 copa erm irm ora)
I=$(( $SLURM_ARRAY_TASK_ID % 5))
M=$(( $SLURM_ARRAY_TASK_ID / 5))

./run_cmnist.py --config conf/sim_multi_training.json \
    --env y_cause \
    --method ${METHODS[M]} \
    --batch_size 512 \
    --hidden_dim 256 \
    --lr 1e-4 \
    --steps 20000 \
    --seed $I \
    --irm_lambda 0.01 \
    --out out/cmnist/tl
