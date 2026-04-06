"""
EMNIST-balanced Task-wise Dirichlet Non-IID Data Generator.

Key properties:
1. Split labels into tasks first.
2. Apply class-wise Dirichlet partition on train inside each task.
3. Allocate test inside each task by matching train client-class counts.
4. Enforce quality constraints per client per task.
"""

import argparse
import gzip
import os
import pickle
import struct
from typing import List, Tuple

import numpy as np
import torch
from torchvision import datasets

cpath = os.path.dirname(__file__)
DATASET_FILE = os.path.join(cpath, "data")


class ImageDataset(object):
    def __init__(self, images, labels, image_data=False, normalize=False):
        if isinstance(images, torch.Tensor):
            if not image_data:
                self.data = images.view(images.size(0), -1).numpy() / 255
            else:
                self.data = images.numpy()
        else:
            self.data = images

        if normalize and not image_data:
            mu = np.mean(self.data.astype(np.float32), 0)
            sigma = np.std(self.data.astype(np.float32), 0)
            self.data = (self.data.astype(np.float32) - mu) / (sigma + 0.001)

        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        self.target = labels.astype(np.int64)

    def __len__(self):
        return len(self.target)


def read_idx_images_gz(gz_path):
    """Read IDX image file from .gz and return numpy array shape (N, rows*cols)."""
    with gzip.open(gz_path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic != 2051:
            raise ValueError(f"Invalid magic number in image file: {magic}")
        num_images = struct.unpack(">I", f.read(4))[0]
        rows = struct.unpack(">I", f.read(4))[0]
        cols = struct.unpack(">I", f.read(4))[0]
        buf = f.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows * cols)
        return data


def read_idx_labels_gz(gz_path):
    """Read IDX label file from .gz and return numpy array shape (N,)."""
    with gzip.open(gz_path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic != 2049:
            raise ValueError(f"Invalid magic number in label file: {magic}")
        num_labels = struct.unpack(">I", f.read(4))[0]
        buf = f.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels


def extract_balanced_from_zip_if_present(zip_path, dest_raw_dir):
    """If zip exists, extract balanced files into dest_raw_dir and return True if any extracted."""
    import zipfile

    extracted = False
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                lower = member.lower()
                if "balanced" in lower and (lower.endswith(".gz") or lower.endswith(".ubyte")):
                    print(f"Extracting {member} from zip...")
                    try:
                        zf.extract(member, path=os.path.dirname(zip_path))
                        src = os.path.join(os.path.dirname(zip_path), member)
                        if not os.path.exists(dest_raw_dir):
                            os.makedirs(dest_raw_dir, exist_ok=True)
                        dst = os.path.join(dest_raw_dir, os.path.basename(member))
                        if os.path.exists(src):
                            try:
                                os.replace(src, dst)
                            except Exception:
                                os.rename(src, dst)
                        else:
                            alt = os.path.join(os.path.dirname(zip_path), os.path.basename(member))
                            if os.path.exists(alt):
                                os.replace(alt, dst)
                        extracted = True
                    except Exception as ex:
                        print(f"  Failed to extract {member}: {ex}")
    except Exception as e:
        print(f"Error opening zip file {zip_path}: {e}")
    return extracted


def load_emnist_balanced(dataset_file):
    print("\n>>> Loading EMNIST-balanced dataset from raw files...")
    raw_dir = os.path.join(dataset_file, "EMNIST", "raw")
    train_images_gz = os.path.join(raw_dir, "emnist-balanced-train-images-idx3-ubyte.gz")
    train_labels_gz = os.path.join(raw_dir, "emnist-balanced-train-labels-idx1-ubyte.gz")
    test_images_gz = os.path.join(raw_dir, "emnist-balanced-test-images-idx3-ubyte.gz")
    test_labels_gz = os.path.join(raw_dir, "emnist-balanced-test-labels-idx1-ubyte.gz")

    if (
        os.path.exists(train_images_gz)
        and os.path.exists(train_labels_gz)
        and os.path.exists(test_images_gz)
        and os.path.exists(test_labels_gz)
    ):
        try:
            train_images = read_idx_images_gz(train_images_gz)
            train_labels = read_idx_labels_gz(train_labels_gz)
            test_images = read_idx_images_gz(test_images_gz)
            test_labels = read_idx_labels_gz(test_labels_gz)
            print("Loaded raw EMNIST files successfully.")
            return train_images, train_labels, test_images, test_labels
        except Exception as e:
            print(f"Error reading raw EMNIST files: {e}")
            print("Falling back to torchvision EMNIST.")

    else:
        print(f"Raw EMNIST files not found in: {raw_dir}")
        zip_path = os.path.join(dataset_file, "gzip.zip")
        if os.path.exists(zip_path):
            print(f"Found zip at {zip_path}, extracting balanced files into {raw_dir} ...")
            ok = extract_balanced_from_zip_if_present(zip_path, raw_dir)
            if ok:
                try:
                    train_images = read_idx_images_gz(train_images_gz)
                    train_labels = read_idx_labels_gz(train_labels_gz)
                    test_images = read_idx_images_gz(test_images_gz)
                    test_labels = read_idx_labels_gz(test_labels_gz)
                    print("Loaded raw EMNIST files successfully after extracting from ZIP.")
                    return train_images, train_labels, test_images, test_labels
                except Exception as e:
                    print(f"Error reading extracted raw files: {e}")
                    print("Falling back to torchvision EMNIST.")
            else:
                print("No balanced files found in zip or extraction failed. Falling back to torchvision.")
        else:
            print(f"No gzip.zip found at {zip_path}. Falling back to torchvision.")

    trainset = datasets.EMNIST(dataset_file, split="balanced", download=False, train=True)
    testset = datasets.EMNIST(dataset_file, split="balanced", download=False, train=False)

    if hasattr(trainset, "data"):
        train_images = trainset.data.numpy() if isinstance(trainset.data, torch.Tensor) else trainset.data
        train_labels = trainset.targets.numpy() if hasattr(trainset, "targets") else trainset.train_labels
    else:
        train_images = trainset.train_data.numpy()
        train_labels = trainset.train_labels

    if hasattr(testset, "data"):
        test_images = testset.data.numpy() if isinstance(testset.data, torch.Tensor) else testset.data
        test_labels = testset.targets.numpy() if hasattr(testset, "targets") else testset.test_labels
    else:
        test_images = testset.test_data.numpy()
        test_labels = testset.test_labels

    return train_images, train_labels, test_images, test_labels


def alpha_tag(alpha: float) -> str:
    text = f"{alpha:.6g}"
    return text.replace(".", "p").replace("-", "m")


def integer_counts(num_samples: int, proportions: np.ndarray) -> np.ndarray:
    """Convert proportions to integer counts that sum to num_samples."""
    if num_samples == 0:
        return np.zeros_like(proportions, dtype=np.int64)

    expected = proportions * num_samples
    counts = np.floor(expected).astype(np.int64)
    residual = int(num_samples - counts.sum())
    if residual > 0:
        frac = expected - counts
        order = np.argsort(-frac)
        counts[order[:residual]] += 1
    return counts


def _sample_task_dirichlet_train_indices(
    labels: np.ndarray,
    candidate_indices: np.ndarray,
    class_values: np.ndarray,
    num_users: int,
    alpha: float,
    rng: np.random.RandomState,
) -> List[List[int]]:
    user_indices = [[] for _ in range(num_users)]

    for cls in class_values:
        cls_indices = candidate_indices[labels[candidate_indices] == int(cls)]
        if len(cls_indices) == 0:
            continue

        shuffled = cls_indices.copy()
        rng.shuffle(shuffled)
        proportions = rng.dirichlet(np.ones(num_users) * alpha)
        counts = integer_counts(len(shuffled), proportions)

        start = 0
        for uid, cnt in enumerate(counts):
            if cnt <= 0:
                continue
            end = start + int(cnt)
            user_indices[uid].extend(shuffled[start:end].tolist())
            start = end

    return user_indices


def _compute_client_class_counts(
    labels: np.ndarray,
    user_indices: List[List[int]],
    class_values: np.ndarray,
) -> np.ndarray:
    class_to_col = {int(c): i for i, c in enumerate(class_values.tolist())}
    counts = np.zeros((len(user_indices), len(class_values)), dtype=np.int64)
    for uid, indices in enumerate(user_indices):
        if not indices:
            continue
        y = labels[np.array(indices, dtype=np.int64)]
        vals, cnts = np.unique(y, return_counts=True)
        for v, c in zip(vals.tolist(), cnts.tolist()):
            if int(v) in class_to_col:
                counts[uid, class_to_col[int(v)]] = int(c)
    return counts


def _allocate_test_counts_from_train(
    num_test_samples: int,
    train_weights: np.ndarray,
) -> np.ndarray:
    active = np.where(train_weights > 0)[0]
    alloc = np.zeros_like(train_weights, dtype=np.int64)

    if num_test_samples == 0 or len(active) == 0:
        return alloc

    active_weights = train_weights[active].astype(np.float64)
    active_weights = active_weights / np.sum(active_weights)

    if num_test_samples >= len(active):
        # Prefer at least one test sample for each active client when class test pool is enough.
        base = np.ones(len(active), dtype=np.int64)
        remain = num_test_samples - len(active)
        extra = integer_counts(remain, active_weights)
        alloc_active = base + extra
    else:
        alloc_active = integer_counts(num_test_samples, active_weights)

    alloc[active] = alloc_active
    return alloc


def _matched_task_test_indices(
    test_labels: np.ndarray,
    task_test_indices: np.ndarray,
    class_values: np.ndarray,
    train_counts: np.ndarray,
    rng: np.random.RandomState,
) -> Tuple[List[List[int]], np.ndarray, np.ndarray]:
    num_users = train_counts.shape[0]
    test_user_indices = [[] for _ in range(num_users)]
    test_counts = np.zeros_like(train_counts, dtype=np.int64)
    class_totals = np.zeros(len(class_values), dtype=np.int64)

    for col, cls in enumerate(class_values.tolist()):
        cls_indices = task_test_indices[test_labels[task_test_indices] == int(cls)]
        shuffled = cls_indices.copy()
        rng.shuffle(shuffled)
        class_totals[col] = int(len(shuffled))

        alloc = _allocate_test_counts_from_train(len(shuffled), train_counts[:, col])
        if int(np.sum(alloc)) != int(len(shuffled)):
            raise RuntimeError(f"test allocation conservation failed for class={cls}")

        start = 0
        for uid, cnt in enumerate(alloc.tolist()):
            if cnt <= 0:
                continue
            end = start + int(cnt)
            chunk = shuffled[start:end]
            test_user_indices[uid].extend(chunk.tolist())
            test_counts[uid, col] = int(cnt)
            start = end

        if start != len(shuffled):
            raise RuntimeError(f"test allocation cursor mismatch for class={cls}, start={start}, total={len(shuffled)}")

    return test_user_indices, test_counts, class_totals


def _validate_task_constraints(
    train_counts: np.ndarray,
    test_counts: np.ndarray,
    min_train_per_client_per_task: int,
    min_test_per_client_per_task: int,
    min_classes_per_client_per_task: int,
    class_totals: np.ndarray,
    task_idx: int,
):
    tr_samples = np.sum(train_counts, axis=1)
    te_samples = np.sum(test_counts, axis=1)
    tr_classes = np.sum(train_counts > 0, axis=1)

    if int(np.min(tr_samples)) < min_train_per_client_per_task:
        raise ValueError(
            f"task-{task_idx}: min train samples/client={int(np.min(tr_samples))} < {min_train_per_client_per_task}"
        )
    if int(np.min(te_samples)) < min_test_per_client_per_task:
        raise ValueError(
            f"task-{task_idx}: min test samples/client={int(np.min(te_samples))} < {min_test_per_client_per_task}"
        )
    if int(np.min(tr_classes)) < min_classes_per_client_per_task:
        raise ValueError(
            f"task-{task_idx}: min train classes/client={int(np.min(tr_classes))} < {min_classes_per_client_per_task}"
        )

    support_violation = np.any((test_counts > 0) & (train_counts <= 0))
    if support_violation:
        raise ValueError(f"task-{task_idx}: found test label assigned to client without train support")

    per_class_sum = np.sum(test_counts, axis=0)
    if not np.array_equal(per_class_sum.astype(np.int64), class_totals.astype(np.int64)):
        raise ValueError(f"task-{task_idx}: class-level test conservation check failed")


def _validate_global_support(train_y: List[list], test_y: List[list]):
    for uid in range(len(train_y)):
        tr = set(int(v) for v in train_y[uid])
        te = set(int(v) for v in test_y[uid])
        if not te.issubset(tr):
            missing = sorted(list(te - tr))
            raise RuntimeError(f"client {uid}: test labels not in train labels: {missing}")


def build_label_distribution(labels_per_user: List[list], class_values: np.ndarray) -> np.ndarray:
    class_to_col = {int(c): i for i, c in enumerate(class_values.tolist())}
    dist = np.zeros((len(labels_per_user), len(class_values)), dtype=np.int64)

    for uid, labels in enumerate(labels_per_user):
        if len(labels) == 0:
            continue
        vals, cnts = np.unique(np.array(labels, dtype=np.int64), return_counts=True)
        for v, c in zip(vals.tolist(), cnts.tolist()):
            if int(v) in class_to_col:
                dist[uid, class_to_col[int(v)]] = int(c)

    return dist


def validate_no_overlap_and_conservation(
    user_indices: List[List[int]],
    total_size: int,
    name: str,
):
    flat_parts = [np.array(v, dtype=np.int64) for v in user_indices if len(v) > 0]
    flat = np.concatenate(flat_parts, axis=0) if flat_parts else np.array([], dtype=np.int64)
    if len(flat) != total_size:
        raise RuntimeError(f"{name}: sample conservation failed, got {len(flat)} != {total_size}")
    if len(np.unique(flat)) != total_size:
        raise RuntimeError(f"{name}: overlap detected across clients (duplicate indices found).")


def print_distribution_stats(label_distribution: np.ndarray, num_samples: List[int], split_name: str):
    print(f"\n>>> {split_name} Distribution Statistics:")
    print(f"Total clients: {label_distribution.shape[0]}")
    print(f"Total classes: {label_distribution.shape[1]}")

    classes_per_client = (label_distribution > 0).sum(axis=1)
    samples_per_client = np.array(num_samples, dtype=np.int64)

    with np.errstate(divide="ignore", invalid="ignore"):
        probs = label_distribution / np.maximum(label_distribution.sum(axis=1, keepdims=True), 1)
        top1 = probs.max(axis=1)
        entropy = -np.sum(np.where(probs > 0, probs * np.log(probs + 1e-12), 0.0), axis=1)

    clients_per_class = (label_distribution > 0).sum(axis=0)

    print(
        f"Classes/client - Min: {classes_per_client.min()}, Max: {classes_per_client.max()}, "
        f"Mean: {classes_per_client.mean():.2f}, Std: {classes_per_client.std():.2f}"
    )
    print(
        f"Samples/client - Min: {samples_per_client.min()}, Max: {samples_per_client.max()}, "
        f"Mean: {samples_per_client.mean():.2f}, Std: {samples_per_client.std():.2f}"
    )
    print(
        f"Clients/class - Min: {clients_per_class.min()}, Max: {clients_per_class.max()}, "
        f"Mean: {clients_per_class.mean():.2f}, Std: {clients_per_class.std():.2f}"
    )
    print(
        f"Top1 class fraction/client - Min: {top1.min():.4f}, Max: {top1.max():.4f}, "
        f"Mean: {top1.mean():.4f}, Std: {top1.std():.4f}"
    )
    print(
        f"Label entropy/client - Min: {entropy.min():.4f}, Max: {entropy.max():.4f}, "
        f"Mean: {entropy.mean():.4f}, Std: {entropy.std():.4f}"
    )


def plot_label_heatmap(
    label_distribution: np.ndarray,
    save_path: str,
    class_values: np.ndarray,
    title_prefix: str,
):
    """Generate and save label distribution heatmaps (proportions and raw counts)."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Heatmap plotting requires matplotlib and seaborn. "
            "Install them or run with --no-save."
        ) from e

    plt.figure(figsize=(20, 12))

    label_proportions = label_distribution / (label_distribution.sum(axis=1, keepdims=True) + 1e-10)
    ax = sns.heatmap(
        label_proportions,
        cmap="YlOrRd",
        cbar_kws={"label": "Proportion of samples"},
        xticklabels=[int(v) for v in class_values.tolist()],
        yticklabels=False,
    )

    n_clients = label_distribution.shape[0]
    if n_clients <= 50:
        ax.set_yticks(np.arange(n_clients) + 0.5)
        ax.set_yticklabels([str(i) for i in range(n_clients)], fontsize=8)
    else:
        max_ticks = 50
        step = max(1, n_clients // max_ticks)
        positions = np.arange(0, n_clients, step)
        ax.set_yticks(positions + 0.5)
        ax.set_yticklabels([str(int(p)) for p in positions], fontsize=7)

    plt.title(f"{title_prefix} Label Distribution Heatmap", fontsize=16, fontweight="bold")
    plt.xlabel("Class ID", fontsize=14)
    plt.ylabel("Client ID", fontsize=14)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f">>> Heatmap saved to: {save_path}")

    plt.figure(figsize=(20, 12))
    ax2 = sns.heatmap(
        label_distribution,
        cmap="Blues",
        cbar_kws={"label": "Number of samples"},
        xticklabels=[int(v) for v in class_values.tolist()],
        yticklabels=False,
        fmt="g",
    )

    if n_clients <= 50:
        ax2.set_yticks(np.arange(n_clients) + 0.5)
        ax2.set_yticklabels([str(i) for i in range(n_clients)], fontsize=8)
    else:
        step = max(1, n_clients // 50)
        positions = np.arange(0, n_clients, step)
        ax2.set_yticks(positions + 0.5)
        ax2.set_yticklabels([str(int(p)) for p in positions], fontsize=7)

    plt.title(f"{title_prefix} Label Distribution Heatmap - Sample Counts", fontsize=16, fontweight="bold")
    plt.xlabel("Class ID", fontsize=14)
    plt.ylabel("Client ID", fontsize=14)
    plt.tight_layout()

    counts_save_path = save_path.replace(".png", "_counts.png")
    plt.savefig(counts_save_path, dpi=300, bbox_inches="tight")
    print(f">>> Counts heatmap saved to: {counts_save_path}")

    plt.close("all")


def create_taskwise_dirichlet_partition(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    num_users: int,
    num_tasks: int,
    alpha: float,
    min_train_per_client_per_task: int,
    min_test_per_client_per_task: int,
    min_classes_per_client_per_task: int,
    max_resample_rounds: int,
    rng: np.random.RandomState,
):
    unique_labels = np.unique(train_labels)
    task_splits = [np.array(x, dtype=np.int64) for x in np.array_split(unique_labels, num_tasks)]

    last_error = "unknown"
    for round_i in range(1, max_resample_rounds + 1):
        agg_train_indices = [[] for _ in range(num_users)]
        agg_test_indices = [[] for _ in range(num_users)]
        round_ok = True

        try:
            for task_idx, task_labels in enumerate(task_splits, start=1):
                task_train_indices = np.where(np.isin(train_labels, task_labels))[0]
                task_test_indices = np.where(np.isin(test_labels, task_labels))[0]

                if len(task_train_indices) == 0 or len(task_test_indices) == 0:
                    raise ValueError(
                        f"task-{task_idx}: empty task split (train={len(task_train_indices)}, test={len(task_test_indices)})"
                    )

                train_user_indices = _sample_task_dirichlet_train_indices(
                    labels=train_labels,
                    candidate_indices=task_train_indices,
                    class_values=task_labels,
                    num_users=num_users,
                    alpha=alpha,
                    rng=rng,
                )

                train_counts = _compute_client_class_counts(
                    labels=train_labels,
                    user_indices=train_user_indices,
                    class_values=task_labels,
                )

                test_user_indices, test_counts, class_totals = _matched_task_test_indices(
                    test_labels=test_labels,
                    task_test_indices=task_test_indices,
                    class_values=task_labels,
                    train_counts=train_counts,
                    rng=rng,
                )

                _validate_task_constraints(
                    train_counts=train_counts,
                    test_counts=test_counts,
                    min_train_per_client_per_task=min_train_per_client_per_task,
                    min_test_per_client_per_task=min_test_per_client_per_task,
                    min_classes_per_client_per_task=min_classes_per_client_per_task,
                    class_totals=class_totals,
                    task_idx=task_idx,
                )

                for uid in range(num_users):
                    agg_train_indices[uid].extend(train_user_indices[uid])
                    agg_test_indices[uid].extend(test_user_indices[uid])

        except Exception as e:
            round_ok = False
            last_error = f"round={round_i}: {type(e).__name__}: {e}"

        if round_ok:
            validate_no_overlap_and_conservation(agg_train_indices, len(train_labels), "train")
            validate_no_overlap_and_conservation(agg_test_indices, len(test_labels), "test")
            print(f">>> Resample succeeded at round {round_i}/{max_resample_rounds}")
            break
    else:
        raise RuntimeError(
            "Failed to generate a valid task-wise Dirichlet partition after "
            f"{max_resample_rounds} rounds. Last error: {last_error}"
        )

    train_x = [[] for _ in range(num_users)]
    train_y = [[] for _ in range(num_users)]
    test_x = [[] for _ in range(num_users)]
    test_y = [[] for _ in range(num_users)]

    for uid in range(num_users):
        tr_idx = np.array(agg_train_indices[uid], dtype=np.int64)
        te_idx = np.array(agg_test_indices[uid], dtype=np.int64)

        rng.shuffle(tr_idx)
        rng.shuffle(te_idx)

        train_x[uid] = train_data[tr_idx].tolist()
        train_y[uid] = train_labels[tr_idx].tolist()
        test_x[uid] = test_data[te_idx].tolist()
        test_y[uid] = test_labels[te_idx].tolist()

    _validate_global_support(train_y, test_y)

    return train_x, train_y, test_x, test_y, unique_labels


def main():
    parser = argparse.ArgumentParser(description="Generate EMNIST-balanced task-wise Dirichlet non-IID partition.")
    parser.add_argument("--num_user", type=int, default=100, help="Number of clients")
    parser.add_argument("--num_classes", type=int, default=47, help="Expected number of classes")
    parser.add_argument("--num_tasks", type=int, default=3, help="Number of continual tasks")
    parser.add_argument("--alpha", type=float, default=0.3, help="Dirichlet concentration alpha per task")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min_train_per_client_per_task", type=int, default=20)
    parser.add_argument("--min_test_per_client_per_task", type=int, default=5)
    parser.add_argument("--min_classes_per_client_per_task", type=int, default=2)
    parser.add_argument("--max_resample_rounds", type=int, default=20)
    parser.add_argument("--save", action="store_true", default=True, help="Whether to save output pkl files")
    parser.add_argument("--no-save", action="store_false", dest="save", help="Disable saving files")
    parser.add_argument("--image_data", action="store_true", default=False, help="Keep image tensor layout")
    args = parser.parse_args()

    if args.alpha <= 0:
        raise ValueError("--alpha must be > 0 for Dirichlet sampling.")
    if args.num_user <= 0:
        raise ValueError("--num_user must be > 0.")
    if args.num_classes <= 0:
        raise ValueError("--num_classes must be > 0.")
    if args.num_tasks <= 0:
        raise ValueError("--num_tasks must be > 0.")

    rng = np.random.RandomState(args.seed)

    print("=" * 80)
    print("EMNIST-balanced Task-wise Dirichlet Non-IID Data Generator")
    print("=" * 80)
    print(
        f">>> Config: num_user={args.num_user}, num_classes={args.num_classes}, num_tasks={args.num_tasks}, "
        f"alpha={args.alpha}, seed={args.seed}, save={args.save}"
    )

    train_images, train_labels, test_images, test_labels = load_emnist_balanced(DATASET_FILE)

    train_dataset = ImageDataset(train_images, train_labels, image_data=args.image_data, normalize=False)
    test_dataset = ImageDataset(test_images, test_labels, image_data=args.image_data, normalize=False)

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    train_X, train_y, test_X, test_y, class_values = create_taskwise_dirichlet_partition(
        train_dataset.data,
        train_dataset.target,
        test_dataset.data,
        test_dataset.target,
        num_users=args.num_user,
        num_tasks=args.num_tasks,
        alpha=args.alpha,
        min_train_per_client_per_task=args.min_train_per_client_per_task,
        min_test_per_client_per_task=args.min_test_per_client_per_task,
        min_classes_per_client_per_task=args.min_classes_per_client_per_task,
        max_resample_rounds=args.max_resample_rounds,
        rng=rng,
    )

    if len(class_values) != args.num_classes:
        print(
            f"Warning: detected {len(class_values)} classes in data, "
            f"but --num_classes={args.num_classes}."
        )

    train_data = {"users": [], "user_data": {}, "num_samples": []}
    test_data = {"users": [], "user_data": {}, "num_samples": []}
    for i in range(args.num_user):
        uname = i
        train_data["users"].append(uname)
        train_data["user_data"][uname] = {"x": train_X[i], "y": train_y[i]}
        train_data["num_samples"].append(len(train_X[i]))
        test_data["users"].append(uname)
        test_data["user_data"][uname] = {"x": test_X[i], "y": test_y[i]}
        test_data["num_samples"].append(len(test_X[i]))

    train_label_distribution = build_label_distribution(train_y, class_values)
    test_label_distribution = build_label_distribution(test_y, class_values)

    print_distribution_stats(train_label_distribution, train_data["num_samples"], split_name="Train")
    print_distribution_stats(test_label_distribution, test_data["num_samples"], split_name="Test")
    print(f"\n>>> Total training size: {sum(train_data['num_samples'])}")
    print(f">>> Total testing size: {sum(test_data['num_samples'])}")

    image_flag = 1 if args.image_data else 0
    a_tag = alpha_tag(args.alpha)
    dataset_tag = f"emnist_balanced_{image_flag}_dirichlet_t{args.num_tasks}_a{a_tag}_niid"
    train_path = f"{cpath}/data/train/{dataset_tag}.pkl"
    test_path = f"{cpath}/data/test/{dataset_tag}.pkl"
    train_dist_path = f"{cpath}/{dataset_tag}_train_label_distribution.npy"
    test_dist_path = f"{cpath}/{dataset_tag}_test_label_distribution.npy"
    train_heatmap_path = f"{cpath}/{dataset_tag}_train_label_heatmap.png"
    test_heatmap_path = f"{cpath}/{dataset_tag}_test_label_heatmap.png"

    for path in [train_path, test_path]:
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    if args.save:
        print("\n>>> Generating train/test label distribution heatmaps...")
        plot_label_heatmap(
            train_label_distribution,
            train_heatmap_path,
            class_values,
            title_prefix=f"Train (Task-wise Dirichlet, alpha={args.alpha:.4g}, tasks={args.num_tasks})",
        )
        plot_label_heatmap(
            test_label_distribution,
            test_heatmap_path,
            class_values,
            title_prefix=f"Test (Matched to train, alpha={args.alpha:.4g}, tasks={args.num_tasks})",
        )

        print("\n>>> Saving data files...")
        with open(train_path, "wb") as f:
            pickle.dump(train_data, f)
        with open(test_path, "wb") as f:
            pickle.dump(test_data, f)
        np.save(train_dist_path, train_label_distribution)
        np.save(test_dist_path, test_label_distribution)
        print(f"Train data saved to: {train_path}")
        print(f"Test data saved to: {test_path}")
        print(f"Train label distribution saved to: {train_dist_path}")
        print(f"Test label distribution saved to: {test_dist_path}")
        print(f"Train heatmap saved to: {train_heatmap_path}")
        print(f"Test heatmap saved to: {test_heatmap_path}")

    print("\n" + "=" * 80)
    print("Data generation completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
