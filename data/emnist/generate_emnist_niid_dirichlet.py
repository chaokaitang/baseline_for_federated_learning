"""
EMNIST-balanced Dirichlet Non-IID Data Generator.

This script keeps the existing shard-based generator untouched and adds a
Dirichlet-based non-IID partition with CLI arguments.
"""

import argparse
import gzip
import os
import pickle
import struct
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
        self.target = labels

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


def dirichlet_counts(num_samples: int, proportions: np.ndarray) -> np.ndarray:
    """Convert class proportions into integer counts that sum to num_samples."""
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


def sample_dirichlet_partition_indices(
    labels: np.ndarray,
    num_users: int,
    num_classes: int,
    alpha: float,
    rng: np.random.RandomState,
) -> List[List[int]]:
    """Partition sample indices by class with class-wise Dirichlet proportions."""
    user_indices = [[] for _ in range(num_users)]
    for cls in range(num_classes):
        cls_indices = np.where(labels == cls)[0]
        if len(cls_indices) == 0:
            continue

        rng.shuffle(cls_indices)
        proportions = rng.dirichlet(np.ones(num_users) * alpha)
        counts = dirichlet_counts(len(cls_indices), proportions)

        start = 0
        for uid, cnt in enumerate(counts):
            if cnt <= 0:
                continue
            end = start + int(cnt)
            user_indices[uid].extend(cls_indices[start:end].tolist())
            start = end
    return user_indices


def repair_empty_clients(user_indices: List[List[int]], rng: np.random.RandomState) -> List[List[int]]:
    """Move samples from data-rich clients to empty ones as a final fallback."""
    counts = np.array([len(v) for v in user_indices], dtype=np.int64)
    empty_clients = np.where(counts == 0)[0].tolist()
    if not empty_clients:
        return user_indices

    for empty_uid in empty_clients:
        donor_order = np.argsort(-counts)
        donated = False
        for donor_uid in donor_order:
            if counts[donor_uid] <= 1:
                continue
            pick_pos = int(rng.randint(0, len(user_indices[donor_uid])))
            moved_idx = user_indices[donor_uid].pop(pick_pos)
            user_indices[empty_uid].append(moved_idx)
            counts[donor_uid] -= 1
            counts[empty_uid] += 1
            donated = True
            break
        if not donated:
            raise RuntimeError("Unable to repair empty clients because all clients have <= 1 sample.")

    return user_indices


def build_non_overlapping_user_data(
    data: np.ndarray,
    labels: np.ndarray,
    user_indices: List[List[int]],
    rng: np.random.RandomState,
) -> Tuple[List[list], List[list]]:
    user_x = [[] for _ in range(len(user_indices))]
    user_y = [[] for _ in range(len(user_indices))]

    for uid, indices in enumerate(user_indices):
        if len(indices) == 0:
            continue
        idx = np.array(indices, dtype=np.int64)
        rng.shuffle(idx)
        user_x[uid] = data[idx].tolist()
        user_y[uid] = labels[idx].tolist()
    return user_x, user_y


def build_label_distribution(train_y: List[list], num_classes: int) -> np.ndarray:
    dist = np.zeros((len(train_y), num_classes), dtype=np.int64)
    for uid, labels in enumerate(train_y):
        if len(labels) == 0:
            continue
        vals, cnts = np.unique(np.array(labels, dtype=np.int64), return_counts=True)
        dist[uid, vals] = cnts
    return dist


def validate_no_overlap_and_conservation(
    user_indices: List[List[int]],
    total_size: int,
    name: str,
):
    flat = np.concatenate([np.array(v, dtype=np.int64) for v in user_indices if len(v) > 0], axis=0)
    if len(flat) != total_size:
        raise RuntimeError(f"{name}: sample conservation failed, got {len(flat)} != {total_size}")
    if len(np.unique(flat)) != total_size:
        raise RuntimeError(f"{name}: overlap detected across clients (duplicate indices found).")


def print_distribution_stats(label_distribution: np.ndarray, train_num_samples: List[int]):
    print("\n>>> Distribution Statistics:")
    print(f"Total clients: {label_distribution.shape[0]}")
    print(f"Total classes: {label_distribution.shape[1]}")

    classes_per_client = (label_distribution > 0).sum(axis=1)
    samples_per_client = np.array(train_num_samples, dtype=np.int64)

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


def plot_label_heatmap(label_distribution: np.ndarray, save_path: str, num_classes: int, alpha: float):
    """Generate and save label distribution heatmaps (proportions and raw counts)."""
    plt.figure(figsize=(20, 12))

    label_proportions = label_distribution / (label_distribution.sum(axis=1, keepdims=True) + 1e-10)
    ax = sns.heatmap(
        label_proportions,
        cmap="YlOrRd",
        cbar_kws={"label": "Proportion of samples"},
        xticklabels=range(num_classes),
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

    plt.title(
        f"Label Distribution Heatmap (Dirichlet Non-IID, alpha={alpha:.4g})",
        fontsize=16,
        fontweight="bold",
    )
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
        xticklabels=range(num_classes),
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

    plt.title(
        f"Label Distribution Heatmap - Sample Counts (Dirichlet, alpha={alpha:.4g})",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Class ID", fontsize=14)
    plt.ylabel("Client ID", fontsize=14)
    plt.tight_layout()

    counts_save_path = save_path.replace(".png", "_counts.png")
    plt.savefig(counts_save_path, dpi=300, bbox_inches="tight")
    print(f">>> Counts heatmap saved to: {counts_save_path}")

    plt.close("all")


def create_dirichlet_partition(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    num_users: int,
    num_classes: int,
    alpha: float,
    rng: np.random.RandomState,
    max_resample_rounds: int = 20,
):
    train_user_indices = None
    for _ in range(max_resample_rounds):
        sampled = sample_dirichlet_partition_indices(
            train_labels, num_users=num_users, num_classes=num_classes, alpha=alpha, rng=rng
        )
        if min(len(v) for v in sampled) > 0:
            train_user_indices = sampled
            break
    if train_user_indices is None:
        sampled = sample_dirichlet_partition_indices(
            train_labels, num_users=num_users, num_classes=num_classes, alpha=alpha, rng=rng
        )
        train_user_indices = repair_empty_clients(sampled, rng)

    test_user_indices = sample_dirichlet_partition_indices(
        test_labels, num_users=num_users, num_classes=num_classes, alpha=alpha, rng=rng
    )

    validate_no_overlap_and_conservation(train_user_indices, len(train_labels), "train")
    validate_no_overlap_and_conservation(test_user_indices, len(test_labels), "test")

    train_X, train_y = build_non_overlapping_user_data(train_data, train_labels, train_user_indices, rng)
    test_X, test_y = build_non_overlapping_user_data(test_data, test_labels, test_user_indices, rng)
    label_distribution = build_label_distribution(train_y, num_classes)

    return train_X, train_y, test_X, test_y, label_distribution


def main():
    parser = argparse.ArgumentParser(description="Generate EMNIST-balanced Dirichlet non-IID partition.")
    parser.add_argument("--num_user", type=int, default=100, help="Number of clients")
    parser.add_argument("--num_classes", type=int, default=47, help="Number of classes")
    parser.add_argument("--alpha", type=float, default=0.3, help="Dirichlet concentration alpha")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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

    rng = np.random.RandomState(args.seed)

    print("=" * 80)
    print("EMNIST-balanced Dirichlet Non-IID Data Generator")
    print("=" * 80)
    print(
        f">>> Config: num_user={args.num_user}, num_classes={args.num_classes}, "
        f"alpha={args.alpha}, seed={args.seed}, save={args.save}"
    )

    train_images, train_labels, test_images, test_labels = load_emnist_balanced(DATASET_FILE)

    train_dataset = ImageDataset(train_images, train_labels, image_data=args.image_data, normalize=False)
    test_dataset = ImageDataset(test_images, test_labels, image_data=args.image_data, normalize=False)

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    train_X, train_y, test_X, test_y, label_distribution = create_dirichlet_partition(
        train_dataset.data,
        train_dataset.target,
        test_dataset.data,
        test_dataset.target,
        num_users=args.num_user,
        num_classes=args.num_classes,
        alpha=args.alpha,
        rng=rng,
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

    print_distribution_stats(label_distribution, train_data["num_samples"])
    print(f"\n>>> Total training size: {sum(train_data['num_samples'])}")
    print(f">>> Total testing size: {sum(test_data['num_samples'])}")

    image_flag = 1 if args.image_data else 0
    a_tag = alpha_tag(args.alpha)
    train_path = f"{cpath}/data/train/emnist_balanced_{image_flag}_dirichlet_a{a_tag}_niid.pkl"
    test_path = f"{cpath}/data/test/emnist_balanced_{image_flag}_dirichlet_a{a_tag}_niid.pkl"
    dist_path = f"{cpath}/emnist_balanced_dirichlet_a{a_tag}_label_distribution.npy"
    heatmap_path = f"{cpath}/emnist_balanced_dirichlet_a{a_tag}_label_heatmap.png"

    for path in [train_path, test_path]:
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    if args.save:
        print("\n>>> Generating label distribution heatmap...")
        plot_label_heatmap(label_distribution, heatmap_path, args.num_classes, args.alpha)

        print("\n>>> Saving data files...")
        with open(train_path, "wb") as f:
            pickle.dump(train_data, f)
        with open(test_path, "wb") as f:
            pickle.dump(test_data, f)
        np.save(dist_path, label_distribution)
        print(f"Train data saved to: {train_path}")
        print(f"Test data saved to: {test_path}")
        print(f"Label distribution saved to: {dist_path}")
        print(f"Heatmap saved to: {heatmap_path}")

    print("\n" + "=" * 80)
    print("Data generation completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
