import os
import pickle
import numpy as np
import torchvision
from torchvision import datasets
import gzip
import struct
import torch

cpath = os.path.dirname(__file__)
DATASET_FILE = os.path.join(cpath, 'data')
NUM_USER = 100
SAVE = True
IMAGE_DATA = False
np.random.seed(6)


class ImageDataset(object):
    def __init__(self, images, labels):
        if hasattr(images, 'numpy'):
            if not IMAGE_DATA:
                self.data = images.view(-1, 784).numpy() / 255
            else:
                self.data = images.numpy()
        else:
            self.data = images
        self.target = np.array(labels)


def data_split(data, num_split):
    delta, r = len(data) // num_split, len(data) % num_split
    data_lst = []
    i, used_r = 0, 0
    while i < len(data):
        if used_r < r:
            data_lst.append(data[i:i+delta+1])
            i += delta + 1
            used_r += 1
        else:
            data_lst.append(data[i:i+delta])
            i += delta
    return data_lst


def read_idx_images_gz(gz_path):
    """Read IDX image file from .gz and return numpy array shape (N, rows*cols)"""
    with gzip.open(gz_path, 'rb') as f:
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


def main():
    print('>>> Generating EMNIST-balanced IID partition')

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
        zip_path = os.path.join(DATASET_FILE, 'gzip.zip')
        if os.path.exists(zip_path):
            print(f'Found zip at {zip_path}, extracting balanced files into {raw_dir} ...')
            ok = extract_balanced_from_zip_if_present(zip_path, raw_dir)
            if ok:
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

    train = ImageDataset(train_images, train_labels)
    test = ImageDataset(test_images, test_labels)

    # Group by class
    class_buckets = [[] for _ in range(47)]
    for img, lbl in zip(train.data, train.target):
        class_buckets[int(lbl)].append(img)

    class_buckets_test = [[] for _ in range(47)]
    for img, lbl in zip(test.data, test.target):
        class_buckets_test[int(lbl)].append(img)

    # Split each class into NUM_USER parts
    split_train = [data_split(bucket, NUM_USER) for bucket in class_buckets]
    split_test = [data_split(bucket, NUM_USER) for bucket in class_buckets_test]

    # Assign to users round-robin across classes to make IID (approximately balanced)
    train_X = [[] for _ in range(NUM_USER)]
    train_y = [[] for _ in range(NUM_USER)]
    test_X = [[] for _ in range(NUM_USER)]
    test_y = [[] for _ in range(NUM_USER)]

    for cls in range(47):
        for user in range(NUM_USER):
            part = split_train[cls][user]
            if len(part) > 0:
                train_X[user] += part.tolist() if hasattr(part, 'tolist') else list(part)
                train_y[user] += [cls] * len(part)

            partt = split_test[cls][user]
            if len(partt) > 0:
                test_X[user] += partt.tolist() if hasattr(partt, 'tolist') else list(partt)
                test_y[user] += [cls] * len(partt)

    # Build data structures
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

    image_flag = 1 if IMAGE_DATA else 0
    train_path = f'{cpath}/data/train/emnist_balanced_{image_flag}_equal_iid.pkl'
    test_path = f'{cpath}/data/test/emnist_balanced_{image_flag}_equal_iid.pkl'

    for path in [os.path.dirname(train_path), os.path.dirname(test_path)]:
        if not os.path.exists(path):
            os.makedirs(path)

    if SAVE:
        with open(train_path, 'wb') as f:
            pickle.dump(train_data, f)
        with open(test_path, 'wb') as f:
            pickle.dump(test_data, f)
        print(f'Saved IID train to {train_path} and test to {test_path}')


if __name__ == '__main__':
    main()
