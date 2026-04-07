#!/bin/bash
set -e

DATASET="emnist_balanced_0_dirichlet_t3_a0p3_niid"
MODEL="2nn"

python main.py \
  --algo stp_fedcl \
  --dataset $DATASET \
  --model $MODEL \
  --sequential_cl \
  --num_tasks 3 \
  --num_round 3 \
  --clients_per_round 5 \
  --num_epoch 1 \
  --batch_size 32 \
  --lr 0.01 \
  --mu 0.001 \
  --lambda_old 0 \
  --lambda_s 0.01 \
  --lambda_l 0.01 \
  --alpha 0.9 \
  --beta_mode fixed \
  --beta_fixed 0.5 \
  --seed 0 \
  --gpu \
  --task_aware \
  --run_name sanity_stp_dir_t3
