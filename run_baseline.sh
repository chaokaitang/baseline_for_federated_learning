#!/bin/bash
set -e

DATASET="emnist_balanced_0_dirichlet_t3_a0p3_niid"
MODEL="2nn"

ROUNDS=40
CLIENTS=50
EPOCH=1
LR=0.01

# 按调参结果改这三个值
STP_MU=0.001
STP_LS=0.01
STP_LL=0.01

SEEDS=(0 1 2)

for SEED in "${SEEDS[@]}"
do
  BASE="dir_t3_a0p3_${MODEL}_r${ROUNDS}_c${CLIENTS}_lr${LR}_sd${SEED}"

  echo "===== Seed $SEED ====="

  python main.py \
    --algo fedavg \
    --dataset $DATASET \
    --model $MODEL \
    --sequential_cl \
    --num_tasks 3 \
    --num_round $ROUNDS \
    --clients_per_round $CLIENTS \
    --num_epoch $EPOCH \
    --batch_size 32 \
    --lr $LR \
    --seed $SEED \
    --gpu \
    --task_aware \
    --run_name "fedavg_${BASE}"

  python main.py \
    --algo fedprox \
    --dataset $DATASET \
    --model $MODEL \
    --mu 0.001 \
    --sequential_cl \
    --num_tasks 3 \
    --num_round $ROUNDS \
    --clients_per_round $CLIENTS \
    --num_epoch $EPOCH \
    --batch_size 32 \
    --lr $LR \
    --seed $SEED \
    --gpu \
    --task_aware \
    --run_name "fedprox_mu1e3_${BASE}"

  python main.py \
    --algo ditto \
    --dataset $DATASET \
    --model $MODEL \
    --lambda_p 1 \
    --personal_num_epoch 1 \
    --sequential_cl \
    --num_tasks 3 \
    --num_round $ROUNDS \
    --clients_per_round $CLIENTS \
    --num_epoch $EPOCH \
    --batch_size 32 \
    --lr $LR \
    --seed $SEED \
    --gpu \
    --task_aware \
    --run_name "ditto_lp1_${BASE}"

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
    --mu $STP_MU \
    --lambda_old 0 \
    --lambda_s $STP_LS \
    --lambda_l $STP_LL \
    --alpha 0.9 \
    --beta_mode fixed \
    --beta_fixed 0.5 \
    --seed $SEED \
    --gpu \
    --task_aware \
    --run_name "stp_full_${BASE}"

done
