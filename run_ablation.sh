#!/bin/bash
set -e
# DATASET="emnist_balanced_0_shard_continual_t3_spt5_niid_for_20u"
DATASET="emnist_balanced_0_dirichlet_t3_a0p3_niid"
MODEL="2nn"

ROUNDS=40
CLIENTS=50
EPOCH=1
LR=0.01

SEEDS=(0)

MU=0
LO=1.0
LS=0.001
LL=0.001

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
    --seed "$5" \
    --gpu \
    --task_aware \
    --run_name "$6"
}

for SEED in "${SEEDS[@]}"
do
  BASE="dirichlet_ablation_${MODEL}_r${ROUNDS}_c${CLIENTS}_lr${LR}_sd${SEED}"

    echo "===== Seed $SEED | global_only ====="
  run_stp 0 $LO 0 0 $SEED "stp_globalonly_${BASE}"

    echo "===== Seed $SEED | full ====="
  run_stp 0 $LO $LS $LL $SEED "stp_full_${BASE}"

  echo "===== Seed $SEED | no_reg ====="
  run_stp 0 0 0 0 $SEED "stp_noreg_${BASE}"


  echo "===== Seed $SEED | personal_only ====="
  run_stp 0 0 $LS $LL $SEED "stp_personalonly_${BASE}"


done

# for SEED in "${SEEDS[@]}"
# do
#   BASE="ablation_${MODEL}_r${ROUNDS}_c${CLIENTS}_lr${LR}_sd${SEED}"

#   echo "===== Seed $SEED | no_reg ====="
#   run_stp 0 0 0 0 $SEED "stp_noreg_${BASE}"

#   echo "===== Seed $SEED | global_only ====="
#   run_stp 0 $LO 0 0 $SEED "stp_globalonly_${BASE}"

#   echo "===== Seed $SEED | personal_only ====="
#   run_stp 0 0 $LS $LL $SEED "stp_personalonly_${BASE}"

#   echo "===== Seed $SEED | full ====="
#   run_stp 0 $LO $LS $LL $SEED "stp_full_${BASE}"
# done