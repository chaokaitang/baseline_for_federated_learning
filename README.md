# Federated Learning Baseline

This repository provides a minimal-yet-complete baseline framework for federated learning (FL) experiments, aimed at helping students quickly build graduation-project demos or research prototypes.

## Overview

- **Single entry point**: `main.py` parses CLI flags, loads data shards, instantiates the configured trainer/model, and launches the FL loop.
- **Config-driven**: `config.py` lists supported datasets (currently MNIST), available models, and maps algorithm names to trainer classes.
- **Trainer architecture**: `src/trainers/` implements the reusable `BaseTrainer` and a FedAvg example; add new aggregation methods by subclassing.
- **Client abstraction**: `Client` and `Worker` (in `src/models/model.py`) encapsulate local training/evaluation and communication/compute statistics.
- **Utilities**: `src/utils/worker_utils.py` handles reading federated `.pkl` data, experiment directory setup, TensorBoard writers, and metric logging.
- **Data layout**: datasets live under `data/<dataset>/data/{train,test}`; each `.pkl` contains `users`, optional `hierarchies`, and `user_data` with `x`/`y` arrays per client.

## Quickstart

1. **Prepare data** following the expected structure above.
2. **Run a baseline experiment** (MNIST + logistic regression + FedAvg):
   ```bash
   python main.py \
     --dataset mnist_all_data_0_equal_niid \
     --model logistic \
     --algo fedavg
   ```

   Common knobs: learning rate (`--lr`), local epochs (`--num_epoch`), batch size (`--batch_size`), total rounds (`--num_round`), clients per round (`--clients_per_round`).
3. **Inspect results** in `result/<dataset>/<exp_name>/` (TensorBoard events and `metrics.json` with accuracy, loss, gradient stats, and communication bytes).

## Extending the framework

- **New algorithms**: create a trainer subclass in `src/trainers/` (e.g., `fedprox.py`), register it in `config.py`, and override `train`/`aggregate` as needed.
- **New models**: add architectures in `src/models/model.py` and update `choose_model`/`MODEL_PARAMS` for the dataset’s input shape and class count.
- **New datasets**: ensure preprocessing outputs compatible `.pkl` shards and extend `MODEL_PARAMS` in `config.py` if input dimensions differ.

## Tips for experiments

- Start with the provided MNIST setup to validate your environment and logging pipeline.
- Use TensorBoard logs in `result/...` to compare communication cost, gradient differences, and accuracy across runs.
- Keep experiment names descriptive when launching multiple trials (e.g., `--exp_name lr0.05_c5_e5`).

## Requirements

- Python 3 with PyTorch installed.
- Enough disk space under `data/` for dataset shards and under `result/` for logs.

This README should give newcomers enough context to navigate the codebase, run baseline experiments, and iterate on federated learning ideas.
