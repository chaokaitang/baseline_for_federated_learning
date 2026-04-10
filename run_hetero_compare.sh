#!/bin/bash
set -e

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

DATASETS=(
  "emnist_balanced_0_dirichlet_t3_a0p3_niid"
  "emnist_balanced_0_shard_continual_t3_spt5_niid"
)

for DATASET in "${DATASETS[@]}"
do
  echo "===== Dataset: $DATASET ====="

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
    --beta_fixed 0.5 \
    --seed $SEED \
    --gpu \
    --task_aware \
    --run_name "hetero_${DATASET}"
done