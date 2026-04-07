# Federated Learning Baseline

This repository provides a minimal-yet-complete baseline framework for federated learning (FL) experiments, aimed at helping students quickly build graduation-project demos or research prototypes.

## Overview

- **Single entry point**: `main.py` parses CLI flags, loads data shards, instantiates the configured trainer/model, and launches the FL loop.
- **Config-driven**: `config.py` lists supported datasets (currently MNIST), available models, and maps algorithm names to trainer classes.
- **Trainer architecture**: `src/trainers/` implements the reusable `BaseTrainer` and a FedAvg example; add new aggregation methods by subclassing.
- **Client abstraction**: `Client` and `Worker` (in `src/models/model.py`) encapsulate local training/evaluation and communication/compute statistics.
- **Utilities**: `src/utils/worker_utils.py` handles reading federated `.pkl` data, experiment directory setup, TensorBoard writers, and metric logging.
- **Data layout**: datasets live under `data/<dataset>/data/{train,test}`; each `.pkl` contains `users`, optional `hierarchies`, and `user_data` with `x`/`y` arrays per client.

## Installation

- Create a Python 3.8+ environment and activate it (recommended).
- Install dependencies. Choose the appropriate PyTorch wheel for your platform and CUDA version from https://pytorch.org/. Example CPU-only install:

```bash
pip install -r requirements.txt
```

If you need GPU/CUDA support, follow the instructions on https://pytorch.org/ to install a compatible `torch`/`torchvision` wheel before or instead of the generic `pip install -r requirements.txt` step.

Optional developer tools:

```bash
pip install flake8 black pytest
```

## Data preparation

This project expects federated dataset shards under `data/<dataset>/data/train` and `data/<dataset>/data/test` as `.pkl` files. Example included datasets and helpers:

- EMNIST helpers: see `data/emnist/download_emnist.py` and `data/emnist/generate_emnist_*` scripts. Typical workflow:

```bash
python data/emnist/download_emnist.py
python data/emnist/generate_emnist_iid.py
# or
python data/emnist/generate_emnist_niid_dirichlet.py
```

- MNIST helpers: see `data/mnist/generate_equal.py` and `data/mnist/generate_random_niid.py`.

If you run into import errors for `src.*` modules, ensure you run `main.py` from the repository root (the folder containing `main.py`) so Python can resolve local imports, or add the project root to `PYTHONPATH`.

## Quickstart

1. **Prepare data** following the expected structure above.
2. **Run a baseline experiment** (MNIST + logistic regression + FedAvg):
   ```bash
   python main.py \
     --dataset mnist_all_data_0_equal_niid \
     --model logistic \
   --algo fedavg \
   --run_name mnist_fedavg_demo
   ```

Example EMNIST run (after generating shards):

```bash
python main.py --dataset emnist_balanced_0_shard_continual_t3_spt5_niid --model lenet --algo fedavg
```

   Common knobs: learning rate (`--lr`), local epochs (`--num_epoch`), batch size (`--batch_size`), total rounds (`--num_round`), clients per round (`--clients_per_round`), output folder (`--run_name`).
3. **Inspect results** in `result/<run_name>/`.

Output organization:

- Non-sequential run: all artifacts are stored under `result/<run_name>/`.
- Sequential CL run: all artifacts for one command are grouped under `result/<run_name>/`, with per-task subfolders:
  - `result/<run_name>/task1/`
  - `result/<run_name>/task2/`
  - `result/<run_name>/task3/`
  and summary/matrix files are saved in `result/<run_name>/`.
- Terminal log: the full stdout/stderr stream is automatically appended to
  `result/<run_name>/run.log` for each command run.

If `--run_name` is not provided, it is auto-generated from timestamp and key parameters.

## Extending the framework

- **New algorithms**: create a trainer subclass in `src/trainers/` (e.g., `fedprox.py`), register it in `config.py`, and override `train`/`aggregate` as needed.
- **New models**: add architectures in `src/models/model.py` and update `choose_model`/`MODEL_PARAMS` for the dataset’s input shape and class count.
- **New datasets**: ensure preprocessing outputs compatible `.pkl` shards and extend `MODEL_PARAMS` in `config.py` if input dimensions differ.

## Notes on algorithm naming

- The current `pfedme` trainer is implemented as an **approximate / pFedMe-like baseline** for quick comparison, not a strict line-by-line reimplementation of the original paper.

## Tips for experiments

- Start with the provided MNIST setup to validate your environment and logging pipeline.
- Use TensorBoard logs in `result/...` to compare communication cost, gradient differences, and accuracy across runs.
- Keep experiment names descriptive when launching multiple trials (e.g., `--exp_name lr0.05_c5_e5`).

## Requirements

- Python 3 with PyTorch installed.
- Enough disk space under `data/` for dataset shards and under `result/` for logs.

**Note on PyTorch**: Depending on your platform and whether you want GPU acceleration, you should install a `torch`/`torchvision` wheel that matches your CUDA toolkit. See https://pytorch.org/ for commands tailored to your environment.

This README should give newcomers enough context to navigate the codebase, run baseline experiments, and iterate on federated learning ideas.
