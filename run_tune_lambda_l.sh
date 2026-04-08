#!/bin/bash
set -e

DATASET="emnist_balanced_0_dirichlet_t3_a0p3_niid"
MODEL="2nn"

ROUNDS=40
CLIENTS=50
EPOCH=1
LR=0.01
SEED=0

# 先固定为前一轮更合适的 mu
MU=0.000

# 第一阶段：固定 lambda_s，只调 lambda_l
LAMBDA_S=0.0

BASE="tune_lambdaL_${MODEL}_r${ROUNDS}_c${CLIENTS}_lr${LR}_sd${SEED}_mu${MU}_ls${LAMBDA_S}"

for LAMBDA_L in 0.1 0.01 0.001 0.0001
do
  echo "===== Tune lambda_l: $LAMBDA_L (lambda_s fixed at $LAMBDA_S) ====="
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