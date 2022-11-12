#!/bin/bash
#SBATCH --array=0-39
#SBATCH --mem=5G
source activate pt
METHODS=(mlp_y4 mlp_erm2 mlp_y2 mlp_erm mlp_irm mlp_dann mlp_coral2 mlp_dro)
I=$(( $SLURM_ARRAY_TASK_ID % 5))
M=$(( $SLURM_ARRAY_TASK_ID / 5))

./run_syn.py --config conf/sim_multi_training.json \
    --env c_cause \
    --method ${METHODS[M]} \
    --batch_size 512 \
    --hidden_dim 10 \
    --lr 1e-4 \
    --steps 20000 \
    --seed $I \
    --coral_penalty_weight 1 \
    --dann_penalty_weight 10 \
    --group_dro_step_size 0.1 \
    --irm_lambda 0.01 \
    --out out/syn/di
