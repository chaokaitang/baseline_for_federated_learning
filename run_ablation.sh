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

BASE="ablation_${MODEL}_r${ROUNDS}_c${CLIENTS}_lr${LR}_sd${SEED}"

run_stp () {
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
    --mu "$1" \
    --lambda_old "$2" \
    --lambda_s "$3" \
    --lambda_l "$4" \
    --alpha 0.9 \
    --beta_mode fixed \
    --beta_fixed 0.5 \
    --seed $SEED \
    --gpu \
    --task_aware \
    --run_name "$5"
}

echo "===== Full ====="
run_stp $MU 0 $LS $LL "stp_full_${BASE}"

echo "===== no mu ====="
run_stp 0 0 $LS $LL "stp_nom_${BASE}"

echo "===== no short ====="
run_stp $MU 0 0 $LL "stp_nos_${BASE}"

echo "===== no long ====="
run_stp $MU 0 $LS 0 "stp_nol_${BASE}"

echo "===== old only ====="
run_stp 0 0.01 0 0 "stp_oldonly_${BASE}"
