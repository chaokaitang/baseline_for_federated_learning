#!/bin/bash
set -e

# Force a non-GUI matplotlib backend to avoid Tk/Tcl thread teardown crashes.
export MPLBACKEND=Agg

# DATASET="emnist_balanced_0_shard_continual_t3_spt5_niid_for_20u"
DATASET="emnist_balanced_0_dirichlet_t3_a0p3_niid"
MODEL="2nn"

ROUNDS=40
CLIENTS=50
EPOCH=1
LR=0.01

SEEDS=(2)

# ===== STP 最终参数 =====
STP_MU=0
STP_LO=1.0
STP_LS=0.001
STP_LL=0.001

for SEED in "${SEEDS[@]}"
do
  BASE="baseline_dirichlet_${MODEL}_r${ROUNDS}_c${CLIENTS}_lr${LR}_sd${SEED}"

  echo "===== Seed $SEED ====="

  # ===== FedAvg =====
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

  # ===== FedProx =====
  python main.py \
    --algo fedprox \
    --dataset $DATASET \
    --model $MODEL \
    --mu 0.1 \
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
    --run_name "fedprox_${BASE}"

  # ===== FedAvg + EWC =====
  python main.py \
    --algo fedavg_ewc \
    --dataset $DATASET \
    --model $MODEL \
    --sequential_cl \
    --num_tasks 3 \
    --num_round $ROUNDS \
    --clients_per_round $CLIENTS \
    --num_epoch $EPOCH \
    --batch_size 32 \
    --lr $LR \
    --lambda_ewc 10.0 \
    --ewc_fisher_samples 128 \
    --seed $SEED \
    --gpu \
    --task_aware \
    --run_name "fedavg_ewc_${BASE}"

  # ===== Ditto =====
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
    --run_name "ditto_${BASE}"

  # ===== STP-FedCL =====
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
    --lambda_old $STP_LO \
    --lambda_s $STP_LS \
    --lambda_l $STP_LL \
    --alpha 0.9 \
    --beta_mode fixed \
    --beta_fixed 0.4 \
    --seed $SEED \
    --gpu \
    --task_aware \
    --run_name "stp_${BASE}"
done