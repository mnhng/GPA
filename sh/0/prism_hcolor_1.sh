#!/bin/bash
#SBATCH --array=0-7
source activate pt2
I=$SLURM_ARRAY_TASK_ID

./run_mnist_prism.py --config conf/syn1.json \
    --env hid_color \
    --morph frac \
    --methods cnn_erm2 cnn_y2 cnn_erm cnn_irm cnn_dann2 cnn_coral2 cnn_dro \
    --batch_size 512 \
    --hidden_dim 256 \
    --lr 1e-4 \
    --steps 20000 \
    --seed $I \
    --n_reps=1 \
    --coral_penalty_weight 10 \
    --dann_penalty_weight 0.01 \
    --group_dro_step_size 0.1 \
    --irm_lambda 0.01 \
    --out out/prism_hcolor1/di/seed${I}
