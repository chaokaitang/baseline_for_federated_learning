#!/bin/bash
set -e

DIR_DATASET="emnist_balanced_0_dirichlet_t3_a0p3_niid"
# 如果你的 shard 数据集名字不同，只改这一行
SHARD_DATASET="emnist_balanced_0_shard_continual_t3_spt5_niid"
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

run_pair () {
  DATASET="$1"
  TAG="$2"

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
    --run_name "ditto_${TAG}_${MODEL}_sd${SEED}"

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
    --beta_fixed 0.5 \
    --seed $SEED \
    --gpu \
    --task_aware \
    --run_name "stp_${TAG}_${MODEL}_sd${SEED}"
}

echo "===== Dirichlet ====="
run_pair "$DIR_DATASET" "dir"

echo "===== Shard ====="
run_pair "$SHARD_DATASET" "shard"
