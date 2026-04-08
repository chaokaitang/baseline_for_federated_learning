#!/bin/bash
set -e

DATASET="emnist_balanced_0_dirichlet_t3_a0p3_niid"
MODEL="2nn"

ROUNDS=40
CLIENTS=50
EPOCH=1
LR=0.01
SEED=0

MU=0.000

# 把这里改成第一阶段跑出来最好的 lambda_l
LAMBDA_L=0.001

BASE="tune_lambdaS_${MODEL}_r${ROUNDS}_c${CLIENTS}_lr${LR}_sd${SEED}_mu${MU}_ll${LAMBDA_L}"

for LAMBDA_S in 0.0001 0.001 0.01 0.1 
do
  echo "===== Tune lambda_s: $LAMBDA_S (lambda_l fixed at $LAMBDA_L) ====="
  python main.py \
    --algo stp_fedcl \
    --dataset $DATASET \
    --model $MODEL \
    --sequential_cl \
    --num_tasks 3 \
    --num_round $ROUNDS \
    --clients_per_round $CLIENTS \
    --num_epoch $EPOCH \
    --batch_size 32 \
    --lr $LR \
    --mu $MU \
    --lambda_old 0 \
    --lambda_s $LAMBDA_S \
    --lambda_l $LAMBDA_L \
    --alpha 0.9 \
    --beta_mode fixed \
    --beta_fixed 0.5 \
    --seed $SEED \
    --gpu \
    --task_aware \
    --log_reg_terms \
    --reg_log_every 10 \
    --run_name "stp_ls${LAMBDA_S}_ll${LAMBDA_L}_${BASE}"
done