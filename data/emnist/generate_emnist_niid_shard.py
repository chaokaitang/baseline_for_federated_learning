"""
EMNIST-balanced Class-skew Non-IID Data Generator with Shard-based Method

This script generates a class-skew non-IID partition of EMNIST-balanced dataset for federated learning
using the shard-based method:
1. Sort all data by label
2. Divide into num_shards = num_clients * shards_per_client shards
3. Each client randomly receives shards_per_client shards
4. Each shard typically contains samples from 1-2 consecutive classes

The label distribution is visualized using a heatmap to verify the non-IID nature.
"""

import torch
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets
import gzip
import struct

cpath = os.path.dirname(__file__)

# Configuration
NUM_USER = 100  # Number of clients
NUM_CLASSES = 47  # EMNIST-balanced has 47 classes
SHARDS_PER_CLIENT = 2  # Number of shards (class partitions) per client
SAVE = True
DATASET_FILE = os.path.join(cpath, 'data')
IMAGE_DATA = False
np.random.seed(42)


class ImageDataset(object):
    def __init__(self, images, labels, normalize=False):
        if isinstance(images, torch.Tensor):
            if not IMAGE_DATA:
                self.data = images.view(images.size(0), -1).numpy() / 255
            else:
                self.data = images.numpy()
        else:
            self.data = images
        
        if normalize and not IMAGE_DATA:
            mu = np.mean(self.data.astype(np.float32), 0)
            sigma = np.std(self.data.astype(np.float32), 0)
            self.data = (self.data.astype(np.float32) - mu) / (sigma + 0.001)
        
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        self.target = labels

    def __len__(self):
        return len(self.target)


def create_shard_based_partition(train_data, train_labels, test_data, test_labels, 
                                  num_users, num_classes, shards_per_client):
    """
    Create class-skew non-IID partition using shard-based method.
    
    Method:
    1. Sort all training data by label
    2. Divide sorted data into num_shards = num_users * shards_per_client shards
    3. Each client randomly gets shards_per_client shards
    4. Each shard contains samples from 1-2 consecutive classes
    
    Args:
        train_data: Training data array
        train_labels: Training labels array
        test_data: Test data array
        test_labels: Test labels array
        num_users: Number of clients
        num_classes: Number of classes
        shards_per_client: Number of shards each client receives
    
    Returns:
        train_X, train_y, test_X, test_y: Lists of data for each client
        label_distribution: Matrix showing label distribution per client
    """
    # Total number of shards
    num_shards = num_users * shards_per_client
    
    # Sort training data by label
    sorted_indices = np.argsort(train_labels)
    train_data_sorted = train_data[sorted_indices]
    train_labels_sorted = train_labels[sorted_indices]
    
    # Calculate shard size
    shard_size = len(train_data_sorted) // num_shards
    
    # Create shards
    print(f'Creating {num_shards} shards with ~{shard_size} samples each...')
    train_shards = []
    for i in range(num_shards):
        start_idx = i * shard_size
        end_idx = (i + 1) * shard_size if i < num_shards - 1 else len(train_data_sorted)
        train_shards.append({
            'data': train_data_sorted[start_idx:end_idx],
            'labels': train_labels_sorted[start_idx:end_idx]
        })
        # Print shard info for first few shards
        if i < 5:
            unique_labels = np.unique(train_labels_sorted[start_idx:end_idx])
            print(f'  Shard {i}: {end_idx - start_idx} samples, classes: {unique_labels}')
    
    # Randomly assign shards to clients
    shard_indices = np.arange(num_shards)
    np.random.shuffle(shard_indices)
    
    train_X = [[] for _ in range(num_users)]
    train_y = [[] for _ in range(num_users)]
    test_X = [[] for _ in range(num_users)]
    test_y = [[] for _ in range(num_users)]
    
    # Track label distribution for heatmap
    label_distribution = np.zeros((num_users, num_classes))
    
    # Assign shards to clients
    for user_id in range(num_users):
        # Get shards for this client
        client_shard_indices = shard_indices[user_id * shards_per_client:(user_id + 1) * shards_per_client]
        
        # Collect data from assigned shards
        for shard_idx in client_shard_indices:
            shard = train_shards[shard_idx]
            train_X[user_id].extend(shard['data'].tolist())
            train_y[user_id].extend(shard['labels'].tolist())
            
            # Update label distribution
            for label in shard['labels']:
                label_distribution[user_id, label] += 1
        
        # Get corresponding test data based on client's class distribution
        client_classes = np.unique(train_y[user_id])
        for class_id in client_classes:
            # Get test samples from this class
            class_test_indices = np.where(test_labels == class_id)[0]
            
            # Calculate proportional test samples
            n_train_samples = np.sum(np.array(train_y[user_id]) == class_id)
            n_test_samples = max(1, min(len(class_test_indices), n_train_samples // 5))
            
            # Randomly sample test data
            if len(class_test_indices) > 0:
                selected_test_indices = np.random.choice(class_test_indices, 
                                                        min(n_test_samples, len(class_test_indices)), 
                                                        replace=False)
                test_X[user_id].extend(test_data[selected_test_indices].tolist())
                test_y[user_id].extend([class_id] * len(selected_test_indices))
    
    # Shuffle data for each client
    for user_id in range(num_users):
        if len(train_X[user_id]) > 0:
            perm = np.random.permutation(len(train_X[user_id]))
            train_X[user_id] = [train_X[user_id][i] for i in perm]
            train_y[user_id] = [train_y[user_id][i] for i in perm]
        
        if len(test_X[user_id]) > 0:
            perm = np.random.permutation(len(test_X[user_id]))
            test_X[user_id] = [test_X[user_id][i] for i in perm]
            test_y[user_id] = [test_y[user_id][i] for i in perm]
    
    return train_X, train_y, test_X, test_y, label_distribution


def read_idx_images_gz(gz_path):
    """Read IDX image file from .gz and return numpy array shape (N, rows, cols) or (N, features)"""
    with gzip.open(gz_path, 'rb') as f:
        # Read magic number and dimensions
        magic = struct.unpack('>I', f.read(4))[0]
        if magic != 2051:
            raise ValueError(f'Invalid magic number in image file: {magic}')
        num_images = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]
        buf = f.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows * cols)
        return data


def read_idx_labels_gz(gz_path):
    """Read IDX label file from .gz and return numpy array shape (N,)"""
    with gzip.open(gz_path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        if magic != 2049:
            raise ValueError(f'Invalid magic number in label file: {magic}')
        num_labels = struct.unpack('>I', f.read(4))[0]
        buf = f.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels


def extract_balanced_from_zip_if_present(zip_path, dest_raw_dir):
    """If zip exists, extract balanced files into dest_raw_dir and return True if any extracted."""
    import zipfile
    extracted = False
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.namelist():
                lower = member.lower()
                if 'balanced' in lower and (lower.endswith('.gz') or lower.endswith('.ubyte')):
                    print(f'Extracting {member} from zip...')
                    try:
                        zf.extract(member, path=os.path.dirname(zip_path))
                        # Move to dest_raw_dir
                        src = os.path.join(os.path.dirname(zip_path), member)
                        if not os.path.exists(dest_raw_dir):
                            os.makedirs(dest_raw_dir, exist_ok=True)
                        dst = os.path.join(dest_raw_dir, os.path.basename(member))
                        if os.path.exists(src):
                            try:
                                os.replace(src, dst)
                            except Exception:
                                # fallback to rename
                                os.rename(src, dst)
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


def plot_label_heatmap(label_distribution, save_path):
    """
    Generate and save label distribution heatmap.
    
    Args:
        label_distribution: Matrix (num_users x num_classes) showing sample counts
        save_path: Path to save the heatmap image
    """
    plt.figure(figsize=(20, 12))
    
    # Normalize by row to show proportion within each client
    label_proportions = label_distribution / (label_distribution.sum(axis=1, keepdims=True) + 1e-10)
    
    # Create heatmap
    ax = sns.heatmap(label_proportions,
                     cmap='YlOrRd',
                     cbar_kws={'label': 'Proportion of samples'},
                     xticklabels=range(NUM_CLASSES),
                     yticklabels=False)

    # Configure y-axis tick labels. If there are many clients, show a sampled subset
    n_clients = label_distribution.shape[0]
    if n_clients <= 50:
        ax.set_yticks(np.arange(n_clients) + 0.5)
        ax.set_yticklabels([str(i) for i in range(n_clients)], fontsize=8)
    else:
        # Show up to ~50 tick labels evenly spaced to avoid clutter
        max_ticks = 50
        step = max(1, n_clients // max_ticks)
        positions = np.arange(0, n_clients, step)
        ax.set_yticks(positions + 0.5)
        ax.set_yticklabels([str(int(p)) for p in positions], fontsize=7)

    plt.title(f'Label Distribution Heatmap (Shard-based Non-IID, {SHARDS_PER_CLIENT} shards/client)',
              fontsize=16, fontweight='bold')
    plt.xlabel('Class ID', fontsize=14)
    plt.ylabel('Client ID', fontsize=14)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'>>> Heatmap saved to: {save_path}')
    
    # Also save raw counts heatmap
    plt.figure(figsize=(20, 12))
    ax2 = sns.heatmap(label_distribution,
                      cmap='Blues',
                      cbar_kws={'label': 'Number of samples'},
                      xticklabels=range(NUM_CLASSES),
                      yticklabels=False,
                      fmt='g')

    # Configure y-axis tick labels for counts heatmap similarly
    if n_clients <= 50:
        ax2.set_yticks(np.arange(n_clients) + 0.5)
        ax2.set_yticklabels([str(i) for i in range(n_clients)], fontsize=8)
    else:
        step = max(1, n_clients // 50)
        positions = np.arange(0, n_clients, step)
        ax2.set_yticks(positions + 0.5)
        ax2.set_yticklabels([str(int(p)) for p in positions], fontsize=7)

    plt.title(f'Label Distribution Heatmap - Sample Counts (Shard-based, {SHARDS_PER_CLIENT} shards/client)',
              fontsize=16, fontweight='bold')
    plt.xlabel('Class ID', fontsize=14)
    plt.ylabel('Client ID', fontsize=14)
    plt.tight_layout()

    counts_save_path = save_path.replace('.png', '_counts.png')
    plt.savefig(counts_save_path, dpi=300, bbox_inches='tight')
    print(f'>>> Counts heatmap saved to: {counts_save_path}')
    
    plt.close('all')


def print_distribution_stats(label_distribution, train_data):
    """Print statistics about the data distribution."""
    print('\n>>> Distribution Statistics:')
    print(f'Total clients: {NUM_USER}')
    print(f'Total classes: {NUM_CLASSES}')
    print(f'Shards per client: {SHARDS_PER_CLIENT}')
    
    # Classes per client
    classes_per_client = (label_distribution > 0).sum(axis=1)
    print(f'Classes per client - Min: {classes_per_client.min()}, Max: {classes_per_client.max()}, '
          f'Mean: {classes_per_client.mean():.2f}, Std: {classes_per_client.std():.2f}')
    
    # Samples per client
    samples_per_client = np.array(train_data['num_samples'])
    print(f'Samples per client - Min: {samples_per_client.min()}, Max: {samples_per_client.max()}, '
          f'Mean: {samples_per_client.mean():.2f}, Std: {samples_per_client.std():.2f}')
    
    # Clients per class
    clients_per_class = (label_distribution > 0).sum(axis=0)
    print(f'Clients per class - Min: {clients_per_class.min()}, Max: {clients_per_class.max()}, '
          f'Mean: {clients_per_class.mean():.2f}, Std: {clients_per_class.std():.2f}')


def main():
    print('=' * 80)
    print('EMNIST-balanced Shard-based Non-IID Data Generator')
    print('=' * 80)
    
    # Load EMNIST-balanced dataset from raw .gz files (created by download_emnist.py)
    print('\n>>> Loading EMNIST-balanced dataset from raw files...')
    raw_dir = os.path.join(DATASET_FILE, 'EMNIST', 'raw')
    train_images_gz = os.path.join(raw_dir, 'emnist-balanced-train-images-idx3-ubyte.gz')
    train_labels_gz = os.path.join(raw_dir, 'emnist-balanced-train-labels-idx1-ubyte.gz')
    test_images_gz = os.path.join(raw_dir, 'emnist-balanced-test-images-idx3-ubyte.gz')
    test_labels_gz = os.path.join(raw_dir, 'emnist-balanced-test-labels-idx1-ubyte.gz')

    if os.path.exists(train_images_gz) and os.path.exists(train_labels_gz) \
       and os.path.exists(test_images_gz) and os.path.exists(test_labels_gz):
        try:
            train_images = read_idx_images_gz(train_images_gz)
            train_labels = read_idx_labels_gz(train_labels_gz)
            test_images = read_idx_images_gz(test_images_gz)
            test_labels = read_idx_labels_gz(test_labels_gz)
            print('Loaded raw EMNIST files successfully.')
        except Exception as e:
            print(f'Error reading raw EMNIST files: {e}')
            print('Falling back to torchvision EMNIST (requires that torchvision can access files).')
            trainset = datasets.EMNIST(DATASET_FILE, split='balanced', download=False, train=True)
            testset = datasets.EMNIST(DATASET_FILE, split='balanced', download=False, train=False)
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
    else:
        print('Raw EMNIST files not found in: {}'.format(raw_dir))
        # If zip exists, extract needed balanced files into raw_dir
        zip_path = os.path.join(DATASET_FILE, 'gzip.zip')
        if os.path.exists(zip_path):
            print(f'Found zip at {zip_path}, extracting balanced files into {raw_dir} ...')
            ok = extract_balanced_from_zip_if_present(zip_path, raw_dir)
            if ok:
                # Retry reading extracted files
                try:
                    train_images = read_idx_images_gz(train_images_gz)
                    train_labels = read_idx_labels_gz(train_labels_gz)
                    test_images = read_idx_images_gz(test_images_gz)
                    test_labels = read_idx_labels_gz(test_labels_gz)
                    print('Loaded raw EMNIST files successfully after extracting from ZIP.')
                except Exception as e:
                    print(f'Error reading extracted raw files: {e}')
                    print('Falling back to torchvision EMNIST (requires that torchvision can access files).')
                    trainset = datasets.EMNIST(DATASET_FILE, split='balanced', download=False, train=True)
                    testset = datasets.EMNIST(DATASET_FILE, split='balanced', download=False, train=False)
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
            else:
                print('No balanced files were found inside the zip or extraction failed.')
                print('Please run download_emnist.py to download gzip.zip or place balanced .gz files in the raw directory.')
                # Try torchvision as last resort
                trainset = datasets.EMNIST(DATASET_FILE, split='balanced', download=False, train=True)
                testset = datasets.EMNIST(DATASET_FILE, split='balanced', download=False, train=False)
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
        else:
            print('No gzip.zip found at {} either.'.format(zip_path))
            print('Please run download_emnist.py to download gzip.zip or place balanced .gz files in the raw directory.')
            # Try torchvision as last resort
            trainset = datasets.EMNIST(DATASET_FILE, split='balanced', download=False, train=True)
            testset = datasets.EMNIST(DATASET_FILE, split='balanced', download=False, train=False)
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

    train_dataset = ImageDataset(train_images, train_labels, normalize=False)
    test_dataset = ImageDataset(test_images, test_labels, normalize=False)
    
    print(f'Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}')
    
    # Print class distribution
    print('\n>>> Class distribution in original dataset:')
    for class_id in range(min(10, NUM_CLASSES)):  # Print first 10 classes
        train_count = np.sum(train_dataset.target == class_id)
        test_count = np.sum(test_dataset.target == class_id)
        print(f'Class {class_id:2d}: Train={train_count:5d}, Test={test_count:4d}')
    if NUM_CLASSES > 10:
        print(f'... (showing first 10 of {NUM_CLASSES} classes)')
    
    # Create shard-based non-IID partition
    print(f'\n>>> Creating shard-based non-IID partition for {NUM_USER} clients...')
    print(f'Total shards: {NUM_USER * SHARDS_PER_CLIENT}')
    print(f'Each client gets {SHARDS_PER_CLIENT} shards')
    print(f'Approximate shard size: {len(train_dataset) // (NUM_USER * SHARDS_PER_CLIENT)} samples')
    
    train_X, train_y, test_X, test_y, label_distribution = create_shard_based_partition(
        train_dataset.data, train_dataset.target,
        test_dataset.data, test_dataset.target,
        NUM_USER, NUM_CLASSES, SHARDS_PER_CLIENT
    )
    
    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}
    
    for i in range(NUM_USER):
        uname = i
        
        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': train_X[i], 'y': train_y[i]}
        train_data['num_samples'].append(len(train_X[i]))
        
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': test_X[i], 'y': test_y[i]}
        test_data['num_samples'].append(len(test_X[i]))
    
    # Print statistics
    print_distribution_stats(label_distribution, train_data)
    
    print(f'\n>>> Total training size: {sum(train_data["num_samples"])}')
    print(f'>>> Total testing size: {sum(test_data["num_samples"])}')
    
    # Setup paths
    print('\n>>> Setting up data paths...')
    image_flag = 1 if IMAGE_DATA else 0
    train_path = f'{cpath}/data/train/emnist_balanced_{image_flag}_shard{SHARDS_PER_CLIENT}_niid.pkl'
    test_path = f'{cpath}/data/test/emnist_balanced_{image_flag}_shard{SHARDS_PER_CLIENT}_niid.pkl'
    heatmap_path = f'{cpath}/emnist_balanced_shard{SHARDS_PER_CLIENT}_label_heatmap.png'
    
    # Create directories
    for path in [train_path, test_path]:
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # Generate and save heatmap
    print('\n>>> Generating label distribution heatmap...')
    plot_label_heatmap(label_distribution, heatmap_path)
    
    # Save data
    if SAVE:
        print('\n>>> Saving data files...')
        with open(train_path, 'wb') as f:
            pickle.dump(train_data, f)
        print(f'Train data saved to: {train_path}')
        
        with open(test_path, 'wb') as f:
            pickle.dump(test_data, f)
        print(f'Test data saved to: {test_path}')
        
        # Also save label distribution for analysis
        dist_path = f'{cpath}/emnist_balanced_shard{SHARDS_PER_CLIENT}_label_distribution.npy'
        np.save(dist_path, label_distribution)
        print(f'Label distribution saved to: {dist_path}')
    
    print('\n' + '=' * 80)
    print('Data generation completed successfully!')
    print('=' * 80)


if __name__ == '__main__':
    main()
