#!/bin/bash
#SBATCH --array=0-4
#SBATCH --mem=40G
source activate gpa

./run_isic.py \
    --zlabels anatom_site_general age_approx sex \
    --conf "conf/isic.json" \
    --method erm2 \
    --batch_size 64 \
    --hidden_dim 256 \
    --seed $SLURM_ARRAY_TASK_ID \
    --lr 3e-5 \
    --steps 12000 \
    --out out/isic_r50
