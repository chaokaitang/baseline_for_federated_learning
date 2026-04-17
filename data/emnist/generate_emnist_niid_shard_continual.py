"""
EMNIST-balanced Continual Shard Non-IID Data Generator.

This generator aligns continual-learning tasks with shard assignment:
1. Split labels into tasks first.
2. For each task, shard only task-local data.
3. Assign each client the same number of shards per task (target 2, fallback to 1).
4. Aggregate task-wise assignments into one train/test pkl pair compatible with main.py.
"""

import argparse
import gzip
import json
import os
import pickle
import struct

import numpy as np
import torch
from torchvision import datasets


cpath = os.path.dirname(__file__)
DATASET_FILE = os.path.join(cpath, 'data')
NUM_CLASSES = 47  # EMNIST-balanced classes


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
    with gzip.open(gz_path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        if magic != 2051:
            raise ValueError(f'Invalid magic number in image file: {magic}')
        num_images = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]
        buf = f.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        return data.reshape(num_images, rows * cols)


def read_idx_labels_gz(gz_path):
    with gzip.open(gz_path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        if magic != 2049:
            raise ValueError(f'Invalid magic number in label file: {magic}')
        num_labels = struct.unpack('>I', f.read(4))[0]
        buf = f.read(num_labels)
        return np.frombuffer(buf, dtype=np.uint8)


def extract_balanced_from_zip_if_present(zip_path, dest_raw_dir):
    import zipfile

    extracted = False
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.namelist():
                lower = member.lower()
                if 'balanced' in lower and (lower.endswith('.gz') or lower.endswith('.ubyte')):
                    try:
                        zf.extract(member, path=os.path.dirname(zip_path))
                        src = os.path.join(os.path.dirname(zip_path), member)
                        os.makedirs(dest_raw_dir, exist_ok=True)
                        dst = os.path.join(dest_raw_dir, os.path.basename(member))
                        if os.path.exists(src):
                            os.replace(src, dst)
                        else:
                            alt = os.path.join(os.path.dirname(zip_path), os.path.basename(member))
                            if os.path.exists(alt):
                                os.replace(alt, dst)
                        extracted = True
                    except Exception as ex:
                        print(f'  Failed to extract {member}: {ex}')
    except Exception as e:
        print(f'Error opening zip file {zip_path}: {e}')

    return extracted


def load_emnist_balanced(dataset_file):
    print('\n>>> Loading EMNIST-balanced dataset from raw files...')
    raw_dir = os.path.join(dataset_file, 'EMNIST', 'raw')
    train_images_gz = os.path.join(raw_dir, 'emnist-balanced-train-images-idx3-ubyte.gz')
    train_labels_gz = os.path.join(raw_dir, 'emnist-balanced-train-labels-idx1-ubyte.gz')
    test_images_gz = os.path.join(raw_dir, 'emnist-balanced-test-images-idx3-ubyte.gz')
    test_labels_gz = os.path.join(raw_dir, 'emnist-balanced-test-labels-idx1-ubyte.gz')

    if os.path.exists(train_images_gz) and os.path.exists(train_labels_gz) and os.path.exists(test_images_gz) and os.path.exists(test_labels_gz):
        try:
            train_images = read_idx_images_gz(train_images_gz)
            train_labels = read_idx_labels_gz(train_labels_gz)
            test_images = read_idx_images_gz(test_images_gz)
            test_labels = read_idx_labels_gz(test_labels_gz)
            print('Loaded raw EMNIST files successfully.')
            return train_images, train_labels, test_images, test_labels
        except Exception as e:
            print(f'Error reading raw EMNIST files: {e}')
            print('Falling back to torchvision EMNIST (download=False).')

    else:
        print(f'Raw EMNIST files not found in: {raw_dir}')
        zip_path = os.path.join(dataset_file, 'gzip.zip')
        if os.path.exists(zip_path):
            print(f'Found zip at {zip_path}, extracting balanced files into {raw_dir} ...')
            extracted = extract_balanced_from_zip_if_present(zip_path, raw_dir)
            if extracted:
                try:
                    train_images = read_idx_images_gz(train_images_gz)
                    train_labels = read_idx_labels_gz(train_labels_gz)
                    test_images = read_idx_images_gz(test_images_gz)
                    test_labels = read_idx_labels_gz(test_labels_gz)
                    print('Loaded raw EMNIST files successfully after extracting from ZIP.')
                    return train_images, train_labels, test_images, test_labels
                except Exception as e:
                    print(f'Error reading extracted raw files: {e}')
        else:
            print(f'No gzip.zip found at {zip_path}.')

    trainset = datasets.EMNIST(dataset_file, split='balanced', download=False, train=True)
    testset = datasets.EMNIST(dataset_file, split='balanced', download=False, train=False)

    if hasattr(trainset, 'data'):
        train_images = trainset.data.numpy() if isinstance(trainset.data, torch.Tensor) else trainset.data
        train_labels = trainset.targets.numpy() if hasattr(trainset, 'targets') else trainset.train_labels
    else:
        train_images = trainset.train_data.numpy()
        train_labels = trainset.train_labels

    if hasattr(testset, 'data'):
        test_images = testset.data.numpy() if isinstance(testset.data, torch.Tensor) else testset.data
        test_labels = testset.targets.numpy() if hasattr(testset, 'targets') else testset.test_labels
    else:
        test_images = testset.test_data.numpy()
        test_labels = testset.test_labels

    return train_images, train_labels, test_images, test_labels


def _plot_task_heatmap(label_distribution, task_idx, save_dir, shards_per_client):
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'emnist_balanced_shard_continual_task{task_idx}_heatmap.png')

    plt.figure(figsize=(20, 12))
    proportions = label_distribution / (label_distribution.sum(axis=1, keepdims=True) + 1e-10)
    n_clients, n_classes = label_distribution.shape

    ax = sns.heatmap(
        proportions,
        cmap='YlOrRd',
        cbar_kws={'label': 'Proportion of samples'},
        xticklabels=range(n_classes),
        yticklabels=False,
    )
    if n_clients <= 50:
        ax.set_yticks(np.arange(n_clients) + 0.5)
        ax.set_yticklabels([str(i) for i in range(n_clients)], fontsize=8)
    else:
        step = max(1, n_clients // 50)
        positions = np.arange(0, n_clients, step)
        ax.set_yticks(positions + 0.5)
        ax.set_yticklabels([str(int(p)) for p in positions], fontsize=7)

    plt.title(
        f'Task-{task_idx} Label Distribution (continual shard, {shards_per_client} shard/client)',
        fontsize=16,
        fontweight='bold',
    )
    plt.xlabel('Class ID')
    plt.ylabel('Client ID')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f'>>> Task-{task_idx} heatmap saved: {save_path}')


def build_label_distribution(train_y, num_classes):
    dist = np.zeros((len(train_y), num_classes), dtype=np.int64)
    for uid, labels in enumerate(train_y):
        if len(labels) == 0:
            continue
        vals, cnts = np.unique(np.array(labels, dtype=np.int64), return_counts=True)
        dist[uid, vals] = cnts
    return dist


def print_distribution_stats(label_distribution, num_samples, split_name):
    print(f'\n>>> {split_name} Distribution Statistics:')
    print(f'Total clients: {label_distribution.shape[0]}')
    print(f'Total classes: {label_distribution.shape[1]}')

    classes_per_client = (label_distribution > 0).sum(axis=1)
    samples_per_client = np.array(num_samples, dtype=np.int64)
    clients_per_class = (label_distribution > 0).sum(axis=0)

    print(
        f'Classes/client - Min: {classes_per_client.min()}, Max: {classes_per_client.max()}, '
        f'Mean: {classes_per_client.mean():.2f}, Std: {classes_per_client.std():.2f}'
    )
    print(
        f'Samples/client - Min: {samples_per_client.min()}, Max: {samples_per_client.max()}, '
        f'Mean: {samples_per_client.mean():.2f}, Std: {samples_per_client.std():.2f}'
    )
    print(
        f'Clients/class - Min: {clients_per_class.min()}, Max: {clients_per_class.max()}, '
        f'Mean: {clients_per_class.mean():.2f}, Std: {clients_per_class.std():.2f}'
    )


def plot_label_heatmap(label_distribution, save_path, num_classes, title_prefix):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(20, 12))
    label_proportions = label_distribution / (label_distribution.sum(axis=1, keepdims=True) + 1e-10)
    ax = sns.heatmap(
        label_proportions,
        cmap='YlOrRd',
        cbar_kws={'label': 'Proportion of samples'},
        xticklabels=range(num_classes),
        yticklabels=False,
    )

    n_clients = label_distribution.shape[0]
    if n_clients <= 50:
        ax.set_yticks(np.arange(n_clients) + 0.5)
        ax.set_yticklabels([str(i) for i in range(n_clients)], fontsize=8)
    else:
        step = max(1, n_clients // 50)
        positions = np.arange(0, n_clients, step)
        ax.set_yticks(positions + 0.5)
        ax.set_yticklabels([str(int(p)) for p in positions], fontsize=7)

    plt.title(
        f'{title_prefix} Label Distribution Heatmap',
        fontsize=16,
        fontweight='bold',
    )
    plt.xlabel('Class ID', fontsize=14)
    plt.ylabel('Client ID', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'>>> Heatmap saved to: {save_path}')

    plt.figure(figsize=(20, 12))
    ax2 = sns.heatmap(
        label_distribution,
        cmap='Blues',
        cbar_kws={'label': 'Number of samples'},
        xticklabels=range(num_classes),
        yticklabels=False,
        fmt='g',
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
        f'{title_prefix} Label Distribution Heatmap - Sample Counts',
        fontsize=16,
        fontweight='bold',
    )
    plt.xlabel('Class ID', fontsize=14)
    plt.ylabel('Client ID', fontsize=14)
    plt.tight_layout()

    counts_save_path = save_path.replace('.png', '_counts.png')
    plt.savefig(counts_save_path, dpi=300, bbox_inches='tight')
    print(f'>>> Counts heatmap saved to: {counts_save_path}')
    plt.close('all')


def _integer_counts(num_samples, proportions):
    if num_samples == 0:
        return np.zeros_like(proportions, dtype=np.int64)
    expected = proportions * float(num_samples)
    counts = np.floor(expected).astype(np.int64)
    residual = int(num_samples - int(np.sum(counts)))
    if residual > 0:
        frac = expected - counts
        order = np.argsort(-frac)
        counts[order[:residual]] += 1
    return counts


def _allocate_test_counts_from_train(num_test_samples, train_weights):
    alloc = np.zeros_like(train_weights, dtype=np.int64)
    active = np.where(train_weights > 0)[0]
    if num_test_samples == 0 or len(active) == 0:
        return alloc

    weights = train_weights[active].astype(np.float64)
    weights = weights / np.sum(weights)

    if num_test_samples >= len(active):
        base = np.ones(len(active), dtype=np.int64)
        remain = int(num_test_samples - len(active))
        extra = _integer_counts(remain, weights)
        alloc_active = base + extra
    else:
        alloc_active = _integer_counts(int(num_test_samples), weights)

    alloc[active] = alloc_active
    return alloc


def _determine_effective_spt(task_train_size, num_user, target_spt):
    if task_train_size >= num_user * target_spt:
        return target_spt
    if task_train_size >= num_user:
        return 1
    raise ValueError(
        f'Task train size={task_train_size} is too small for num_user={num_user}. '
        'Need at least one sample per client.'
    )


def _assign_task_shards(
    train_data,
    train_labels,
    test_data,
    test_labels,
    task_labels,
    num_user,
    target_spt,
    rng,
):
    label_set = set(int(x) for x in task_labels)

    tr_mask = np.isin(train_labels, list(label_set))
    te_mask = np.isin(test_labels, list(label_set))

    tr_x = train_data[tr_mask]
    tr_y = train_labels[tr_mask]
    te_x = test_data[te_mask]
    te_y = test_labels[te_mask]

    if tr_y.size == 0:
        raise ValueError(f'Task labels {sorted(label_set)} has no training samples.')
    if te_y.size == 0:
        raise ValueError(f'Task labels {sorted(label_set)} has no testing samples.')

    effective_spt = _determine_effective_spt(len(tr_y), num_user, target_spt)
    if effective_spt < target_spt:
        print(
            f'Warning: task labels={sorted(label_set)} fallback shards_per_client '
            f'from {target_spt} to {effective_spt} due to limited samples ({len(tr_y)}).'
        )

    num_shards = num_user * effective_spt

    sorted_idx = np.argsort(tr_y)
    tr_x_sorted = tr_x[sorted_idx]
    tr_y_sorted = tr_y[sorted_idx]

    shard_size = len(tr_y_sorted) // num_shards
    if shard_size == 0:
        raise ValueError(f'Computed shard_size=0 for task labels {sorted(label_set)}.')

    train_shards = []
    for i in range(num_shards):
        start = i * shard_size
        end = (i + 1) * shard_size if i < num_shards - 1 else len(tr_y_sorted)
        train_shards.append((tr_x_sorted[start:end], tr_y_sorted[start:end]))

    shard_indices = np.arange(num_shards)
    rng.shuffle(shard_indices)

    task_train_x = [[] for _ in range(num_user)]
    task_train_y = [[] for _ in range(num_user)]
    task_test_x = [[] for _ in range(num_user)]
    task_test_y = [[] for _ in range(num_user)]

    full_label_min = int(np.min(train_labels))
    full_label_max = int(np.max(train_labels))
    label_distribution = np.zeros((num_user, full_label_max - full_label_min + 1), dtype=np.int64)
    task_labels_sorted = np.sort(np.array(list(label_set), dtype=np.int64))
    task_class_to_col = {int(v): i for i, v in enumerate(task_labels_sorted.tolist())}
    task_train_counts = np.zeros((num_user, len(task_labels_sorted)), dtype=np.int64)

    for uid in range(num_user):
        my_indices = shard_indices[uid * effective_spt:(uid + 1) * effective_spt]
        for sid in my_indices:
            sx, sy = train_shards[int(sid)]
            task_train_x[uid].extend(sx.tolist())
            task_train_y[uid].extend(sy.tolist())
            for lab in sy:
                ilab = int(lab)
                label_distribution[uid, ilab - full_label_min] += 1
                task_train_counts[uid, task_class_to_col[ilab]] += 1

    task_test_counts = np.zeros_like(task_train_counts, dtype=np.int64)
    for col, class_id in enumerate(task_labels_sorted.tolist()):
        class_te_idx = np.where(te_y == int(class_id))[0]
        class_te_idx = class_te_idx.copy()
        rng.shuffle(class_te_idx)

        alloc = _allocate_test_counts_from_train(len(class_te_idx), task_train_counts[:, col])
        if int(np.sum(alloc)) != int(len(class_te_idx)):
            raise RuntimeError(
                f'task class={class_id}: test allocation conservation failed '
                f'({int(np.sum(alloc))} != {len(class_te_idx)})'
            )

        start = 0
        for uid, cnt in enumerate(alloc.tolist()):
            if cnt <= 0:
                continue
            end = start + int(cnt)
            selected = class_te_idx[start:end]
            task_test_x[uid].extend(te_x[selected].tolist())
            task_test_y[uid].extend([int(class_id)] * len(selected))
            task_test_counts[uid, col] += int(cnt)
            start = end

        if start != len(class_te_idx):
            raise RuntimeError(
                f'task class={class_id}: test allocation cursor mismatch '
                f'({start} != {len(class_te_idx)})'
            )

    if not np.array_equal(np.sum(task_test_counts, axis=0), np.array([np.sum(te_y == c) for c in task_labels_sorted])):
        raise RuntimeError('Task class-level test conservation check failed.')

    support_violation = np.any((task_test_counts > 0) & (task_train_counts <= 0))
    if support_violation:
        raise RuntimeError('Task matched-test support violation: some client got unseen class in test.')

    for uid in range(num_user):
        if len(task_train_x[uid]) > 0:
            p = rng.permutation(len(task_train_x[uid]))
            task_train_x[uid] = [task_train_x[uid][i] for i in p]
            task_train_y[uid] = [task_train_y[uid][i] for i in p]
        if len(task_test_x[uid]) > 0:
            p = rng.permutation(len(task_test_x[uid]))
            task_test_x[uid] = [task_test_x[uid][i] for i in p]
            task_test_y[uid] = [task_test_y[uid][i] for i in p]

    per_client_stats = []
    for uid in range(num_user):
        tr_labels = np.array(task_train_y[uid], dtype=np.int64)
        te_labels = np.array(task_test_y[uid], dtype=np.int64)
        per_client_stats.append(
            {
                'client_id': uid,
                'train_samples': int(len(task_train_x[uid])),
                'test_samples': int(len(task_test_x[uid])),
                'train_num_classes': int(len(np.unique(tr_labels))) if tr_labels.size else 0,
                'test_num_classes': int(len(np.unique(te_labels))) if te_labels.size else 0,
            }
        )

    return {
        'effective_spt': int(effective_spt),
        'task_train_x': task_train_x,
        'task_train_y': task_train_y,
        'task_test_x': task_test_x,
        'task_test_y': task_test_y,
        'label_distribution': label_distribution,
        'task_labels': sorted([int(v) for v in task_labels]),
        'per_client_stats': per_client_stats,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Generate EMNIST balanced continual shard non-IID partition.')
    parser.add_argument('--num_user', type=int, default=20, help='Number of clients')
    parser.add_argument('--num_tasks', type=int, default=3, help='Number of continual tasks')
    parser.add_argument('--shards_per_client_per_task', type=int, default=5, help='Target shards per client per task')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--image_data', action='store_true', help='Keep image shape instead of flattening')
    parser.add_argument('--save', action='store_true', default=True, help='Save generated files')
    parser.add_argument('--no-save', action='store_false', dest='save', help='Do not save generated files')
    parser.add_argument('--plot_task_heatmap', action='store_true', help='Save per-task heatmaps')
    return parser.parse_args()


def _validate_global_support(agg_train_y, agg_test_y):
    for uid in range(len(agg_train_y)):
        tr = set(int(v) for v in agg_train_y[uid])
        te = set(int(v) for v in agg_test_y[uid])
        if not te.issubset(tr):
            missing = sorted(list(te - tr))
            raise ValueError(f'Client {uid} has unseen test labels: {missing}')


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print('=' * 80)
    print('EMNIST-balanced Continual Shard Non-IID Data Generator')
    print('=' * 80)
    print(f'num_user={args.num_user}, num_tasks={args.num_tasks}, target_spt={args.shards_per_client_per_task}, seed={args.seed}')

    train_images, train_labels, test_images, test_labels = load_emnist_balanced(DATASET_FILE)
    train_dataset = ImageDataset(train_images, train_labels, image_data=args.image_data, normalize=False)
    test_dataset = ImageDataset(test_images, test_labels, image_data=args.image_data, normalize=False)

    unique_labels = np.unique(train_dataset.target)
    if len(unique_labels) != NUM_CLASSES:
        print(f'Warning: detected {len(unique_labels)} unique labels instead of expected {NUM_CLASSES}.')

    split_labels = [np.array(x, dtype=np.int64) for x in np.array_split(unique_labels, args.num_tasks)]

    print('\n>>> Task label splits:')
    for i, labels in enumerate(split_labels, start=1):
        print(f'  Task-{i}: {labels.tolist()}')

    agg_train_x = [[] for _ in range(args.num_user)]
    agg_train_y = [[] for _ in range(args.num_user)]
    agg_test_x = [[] for _ in range(args.num_user)]
    agg_test_y = [[] for _ in range(args.num_user)]

    task_summaries = []

    for task_idx, task_labels in enumerate(split_labels, start=1):
        print(f'\n>>> Building Task-{task_idx}/{args.num_tasks} ...')
        task_ret = _assign_task_shards(
            train_dataset.data,
            train_dataset.target,
            test_dataset.data,
            test_dataset.target,
            task_labels,
            args.num_user,
            args.shards_per_client_per_task,
            rng,
        )

        for uid in range(args.num_user):
            agg_train_x[uid].extend(task_ret['task_train_x'][uid])
            agg_train_y[uid].extend(task_ret['task_train_y'][uid])
            agg_test_x[uid].extend(task_ret['task_test_x'][uid])
            agg_test_y[uid].extend(task_ret['task_test_y'][uid])

        if args.plot_task_heatmap:
            _plot_task_heatmap(
                task_ret['label_distribution'],
                task_idx,
                cpath,
                task_ret['effective_spt'],
            )

        task_summaries.append(
            {
                'task_idx': int(task_idx),
                'task_labels': task_ret['task_labels'],
                'effective_spt': int(task_ret['effective_spt']),
                'per_client_stats': task_ret['per_client_stats'],
            }
        )

    # Final shuffle per client after concatenating all tasks.
    for uid in range(args.num_user):
        if len(agg_train_x[uid]) > 0:
            p = rng.permutation(len(agg_train_x[uid]))
            agg_train_x[uid] = [agg_train_x[uid][i] for i in p]
            agg_train_y[uid] = [agg_train_y[uid][i] for i in p]
        if len(agg_test_x[uid]) > 0:
            p = rng.permutation(len(agg_test_x[uid]))
            agg_test_x[uid] = [agg_test_x[uid][i] for i in p]
            agg_test_y[uid] = [agg_test_y[uid][i] for i in p]

    for uid in range(args.num_user):
        if len(agg_train_x[uid]) == 0 or len(agg_test_x[uid]) == 0:
            raise ValueError(f'Client {uid} has empty train/test after aggregation. Train={len(agg_train_x[uid])}, Test={len(agg_test_x[uid])}')

    _validate_global_support(agg_train_y, agg_test_y)

    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    for uid in range(args.num_user):
        train_data['users'].append(uid)
        train_data['user_data'][uid] = {'x': agg_train_x[uid], 'y': agg_train_y[uid]}
        train_data['num_samples'].append(len(agg_train_x[uid]))

        test_data['users'].append(uid)
        test_data['user_data'][uid] = {'x': agg_test_x[uid], 'y': agg_test_y[uid]}
        test_data['num_samples'].append(len(agg_test_x[uid]))

    print('\n>>> Aggregated statistics:')
    train_samples = np.array(train_data['num_samples'])
    test_samples = np.array(test_data['num_samples'])
    print(f'  Train samples/client: min={train_samples.min()}, max={train_samples.max()}, mean={train_samples.mean():.2f}')
    print(f'  Test samples/client : min={test_samples.min()}, max={test_samples.max()}, mean={test_samples.mean():.2f}')
    if int(np.sum(train_samples)) != len(train_dataset):
        raise RuntimeError(f'Train conservation failed: {int(np.sum(train_samples))} != {len(train_dataset)}')
    if int(np.sum(test_samples)) != len(test_dataset):
        raise RuntimeError(f'Test conservation failed: {int(np.sum(test_samples))} != {len(test_dataset)}')

    train_label_distribution = build_label_distribution(agg_train_y, NUM_CLASSES)
    test_label_distribution = build_label_distribution(agg_test_y, NUM_CLASSES)
    print_distribution_stats(train_label_distribution, train_data['num_samples'], split_name='Train')
    print_distribution_stats(test_label_distribution, test_data['num_samples'], split_name='Test')

    image_flag = 1 if args.image_data else 0
    dataset_tag = f'emnist_balanced_{image_flag}_shard_continual_t{args.num_tasks}_spt{args.shards_per_client_per_task}_niid_for_{args.num_user}u'
    train_path = os.path.join(cpath, 'data', 'train', f'{dataset_tag}.pkl')
    test_path = os.path.join(cpath, 'data', 'test', f'{dataset_tag}.pkl')
    summary_path = os.path.join(cpath, f'{dataset_tag}_summary.json')
    train_dist_path = os.path.join(cpath, f'{dataset_tag}_train_label_distribution.npy')
    test_dist_path = os.path.join(cpath, f'{dataset_tag}_test_label_distribution.npy')
    train_heatmap_path = os.path.join(cpath, f'{dataset_tag}_train_label_heatmap.png')
    test_heatmap_path = os.path.join(cpath, f'{dataset_tag}_test_label_heatmap.png')

    summary = {
        'dataset_tag': dataset_tag,
        'num_user': int(args.num_user),
        'num_tasks': int(args.num_tasks),
        'target_spt': int(args.shards_per_client_per_task),
        'seed': int(args.seed),
        'task_summaries': task_summaries,
        'total_train_samples': int(np.sum(train_samples)),
        'total_test_samples': int(np.sum(test_samples)),
        'train_samples_per_client': [int(x) for x in train_samples.tolist()],
        'test_samples_per_client': [int(x) for x in test_samples.tolist()],
    }

    if args.save:
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_path), exist_ok=True)

        print('\n>>> Generating train/test label distribution heatmaps...')
        plot_label_heatmap(
            train_label_distribution,
            train_heatmap_path,
            NUM_CLASSES,
            title_prefix=f'Train (Continual Shard, tasks={args.num_tasks}, target_spt={args.shards_per_client_per_task})',
        )
        plot_label_heatmap(
            test_label_distribution,
            test_heatmap_path,
            NUM_CLASSES,
            title_prefix=f'Test (Matched to train, tasks={args.num_tasks}, target_spt={args.shards_per_client_per_task})',
        )

        with open(train_path, 'wb') as f:
            pickle.dump(train_data, f)
        with open(test_path, 'wb') as f:
            pickle.dump(test_data, f)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        np.save(train_dist_path, train_label_distribution)
        np.save(test_dist_path, test_label_distribution)

        print(f'>>> Train data saved: {train_path}')
        print(f'>>> Test data saved : {test_path}')
        print(f'>>> Summary saved   : {summary_path}')
        print(f'>>> Train distribution saved: {train_dist_path}')
        print(f'>>> Test distribution saved : {test_dist_path}')
        print(f'>>> Train heatmap saved: {train_heatmap_path}')
        print(f'>>> Test heatmap saved : {test_heatmap_path}')
    else:
        print('>>> --no-save enabled, skipped writing files.')

    print('=' * 80)
    print('Done.')
    print('=' * 80)


if __name__ == '__main__':
    main()
