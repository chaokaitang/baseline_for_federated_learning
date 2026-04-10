#!/bin/bash
set -e

DATASET="emnist_balanced_0_dirichlet_t3_a0p3_niid"
MODEL="2nn"

ROUNDS=40
CLIENTS=50
EPOCH=1
LR=0.01

SEED=0

MU=0
LO=1.0
LS=0.001
LL=0.001

BETAS=(0.2 0.5 0.8)

for BETA in "${BETAS[@]}"
do
  echo "===== Beta $BETA ====="

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
    --lambda_old $LO \
    --lambda_s $LS \
    --lambda_l $LL \
    --alpha 0.9 \
    --beta_mode fixed \
    --beta_fixed $BETA \
    --seed $SEED \
    --gpu \
    --task_aware \
    --run_name "stp_beta${BETA}"
done