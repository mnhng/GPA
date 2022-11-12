#!/bin/bash
#SBATCH --array=0-4
source activate pt
I=$SLURM_ARRAY_TASK_ID

./run_cmnist.py --config conf/syn_weak_strong.json \
    --env c_cause \
    --methods cnn_y4 cnn_erm2 cnn_y2 cnn_erm cnn_irm cnn_dann cnn_coral2 cnn_dro ora_erm \
    --batch_size 512 \
    --hidden_dim 256 \
    --lr 1e-4 \
    --steps 20000 \
    --seed $I \
    --coral_penalty_weight 10 \
    --dann_penalty_weight 0.01 \
    --group_dro_step_size 0.1 \
    --irm_lambda 0.01 \
    --out out/cmnist_weak/di/seed${I}
