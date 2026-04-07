#!/bin/bash
set -e

DATASET="emnist_balanced_0_dirichlet_t3_a0p3_niid"
MODEL="2nn"

ROUNDS=40
CLIENTS=50
EPOCH=1
LR=0.01
SEED=0

# 按调参结果改这里
MU=0.001
LS=0.01
LL=0.01

BASE="beta_${MODEL}_r${ROUNDS}_c${CLIENTS}_lr${LR}_sd${SEED}"

run_beta () {
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
    --lambda_s $LS \
    --lambda_l $LL \
    --alpha 0.9 \
    --beta_mode fixed \
    --beta_fixed "$1" \
    --seed $SEED \
    --gpu \
    --task_aware \
    --run_name "$2"
}

echo "===== Beta 0.2 ====="
run_beta 0.2 "stp_b02_${BASE}"

echo "===== Beta 0.5 ====="
run_beta 0.5 "stp_b05_${BASE}"

echo "===== Beta 0.8 ====="
run_beta 0.8 "stp_b08_${BASE}"
