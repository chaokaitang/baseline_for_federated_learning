#!/bin/bash
set -e

DATASET="emnist_balanced_0_dirichlet_t3_a0p3_niid"
MODEL="2nn"

ROUNDS=40
CLIENTS=50
EPOCH=1
LR=0.01

SEEDS=(0 1 2)

for SEED in "${SEEDS[@]}"
do
  echo "===== Seed $SEED ====="

  # ---------- FedAvg ----------
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
    --task_aware

  # ---------- FedProx ----------
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
    --task_aware

  # ---------- Ditto ----------
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
    --task_aware

  # ---------- STP-FedCL ----------
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
    --mu 0.001 \
    --lambda_old 0 \
    --lambda_s 0.01 \
    --lambda_l 0.01 \
    --alpha 0.9 \
    --beta_mode fixed \
    --beta_fixed 0.5 \
    --seed $SEED \
    --gpu \
    --task_aware \
    --log_reg_terms \
    --reg_log_every 10

done