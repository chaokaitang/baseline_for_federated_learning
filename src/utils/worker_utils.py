import pickle
import json
import numpy as np
import os
import time
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset
from PIL import Image


__all__ = ['mkdir', 'read_data', 'Metrics', "MiniDataset"]


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def read_data(train_data_dir, test_data_dir, key=None):
    """Parses data in given train and test data directories

    Assumes:
        1. the data in the input directories are .json files with keys 'users' and 'user_data'
        2. the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data (ndarray)
        test_data: dictionary of test data (ndarray)
    """

    clients = []
    groups = []
    train_data = {}
    test_data = {}
    print('>>> Read data from:')

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.pkl')]
    if key is not None:
        train_files = list(filter(lambda x: str(key) in x, train_files))

    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        print('    ', file_path)

        with open(file_path, 'rb') as inf:
            cdata = pickle.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    for cid, v in train_data.items():
        train_data[cid] = MiniDataset(v['x'], v['y'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.pkl')]
    if key is not None:
        test_files = list(filter(lambda x: str(key) in x, test_files))

    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        print('    ', file_path)

        with open(file_path, 'rb') as inf:
            cdata = pickle.load(inf)
        test_data.update(cdata['user_data'])

    for cid, v in test_data.items():
        test_data[cid] = MiniDataset(v['x'], v['y'])

    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data


class MiniDataset(Dataset):
    def __init__(self, data, labels):
        super(MiniDataset, self).__init__()
        self.data = np.array(data)
        self.labels = np.array(labels).astype("int64")

        if self.data.ndim == 4 and self.data.shape[3] == 3:
            self.data = self.data.astype("uint8")
            self.transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(32, 4),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                 ]
            )
        elif self.data.ndim == 4 and self.data.shape[3] == 1:
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))
                 ]
            )
        elif self.data.ndim == 3:
            self.data = self.data.reshape(-1, 28, 28, 1).astype("uint8")
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))
                 ]
            )
        else:
            self.data = self.data.astype("float32")
            self.transform = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, target = self.data[index], self.labels[index]

        if self.data.ndim == 4 and self.data.shape[3] == 3:
            data = Image.fromarray(data)

        if self.transform is not None:
            data = self.transform(data)

        return data, target


class Metrics(object):
    def __init__(self, clients, options, name=''):
        self.options = options
        num_rounds = options['num_round'] + 1
        self.bytes_written = {c.cid: [0] * num_rounds for c in clients}
        self.client_computations = {c.cid: [0] * num_rounds for c in clients}
        self.bytes_read = {c.cid: [0] * num_rounds for c in clients}

        # Statistics in training procedure
        self.loss_on_train_data = [0] * num_rounds
        self.acc_on_train_data = [0] * num_rounds
        self.gradnorm_on_train_data = [0] * num_rounds
        self.graddiff_on_train_data = [0] * num_rounds

        # Statistics in test procedure
        self.loss_on_eval_data = [0] * num_rounds
        self.acc_on_eval_data = [0] * num_rounds
        # Personalized model stats: dict(cid -> list per round)
        self.personalized_loss_on_eval = {c.cid: [0] * num_rounds for c in clients}
        self.personalized_acc_on_eval = {c.cid: [0] * num_rounds for c in clients}
        # Aggregated personalized stats across clients (per round)
        self.personalized_mean_loss = [0] * num_rounds
        self.personalized_mean_acc = [0] * num_rounds
        self.personalized_std_acc = [0] * num_rounds
        # Collaborative (global-personal mixed) stats
        self.collab_loss_on_eval = {c.cid: [0] * num_rounds for c in clients}
        self.collab_acc_on_eval = {c.cid: [0] * num_rounds for c in clients}
        self.collab_mean_loss = [0] * num_rounds
        self.collab_mean_acc = [0] * num_rounds
        self.collab_std_acc = [0] * num_rounds

        self.result_path = mkdir(os.path.join('./result', self.options['dataset']))
        suffix = '{}_sd{}_lr{}_ep{}_bs{}_{}'.format(name,
                                                    options['seed'],
                                                    options['lr'],
                                                    options['num_epoch'],
                                                    options['batch_size'],
                                                    'w' if options['simple_average'] else 'a')

        self.exp_name = '{}_{}_{}_{}'.format(time.strftime('%Y-%m-%dT%H-%M-%S'), options['algo'],
                                             options['model'], suffix)
        if options['dis']:
            suffix = options['dis']
            self.exp_name += '_{}'.format(suffix)
        train_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'train.event'))
        eval_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'eval.event'))
        self.train_writer = SummaryWriter(train_event_folder)
        self.eval_writer = SummaryWriter(eval_event_folder)

    def update_commu_stats(self, round_i, stats):
        cid, bytes_w, comp, bytes_r = \
            stats['id'], stats['bytes_w'], stats['comp'], stats['bytes_r']

        self.bytes_written[cid][round_i] += bytes_w
        self.client_computations[cid][round_i] += comp
        self.bytes_read[cid][round_i] += bytes_r

    def extend_commu_stats(self, round_i, stats_list):
        for stats in stats_list:
            self.update_commu_stats(round_i, stats)

    def update_train_stats(self, round_i, train_stats):
        self.loss_on_train_data[round_i] = train_stats['loss']
        self.acc_on_train_data[round_i] = train_stats['acc']
        self.gradnorm_on_train_data[round_i] = train_stats['gradnorm']
        self.graddiff_on_train_data[round_i] = train_stats['graddiff']

        self.train_writer.add_scalar('train_loss', train_stats['loss'], round_i)
        self.train_writer.add_scalar('train_acc', train_stats['acc'], round_i)
        self.train_writer.add_scalar('gradnorm', train_stats['gradnorm'], round_i)
        self.train_writer.add_scalar('graddiff', train_stats['graddiff'], round_i)

    def update_eval_stats(self, round_i, eval_stats):
        self.loss_on_eval_data[round_i] = eval_stats['loss']
        self.acc_on_eval_data[round_i] = eval_stats['acc']

        self.eval_writer.add_scalar('test_loss', eval_stats['loss'], round_i)
        self.eval_writer.add_scalar('test_acc', eval_stats['acc'], round_i)

    def update_personalized_eval_stats(self, round_i, cid, loss, acc):
        """Record personalized model evaluation for a single client at a given round."""
        if cid not in self.personalized_loss_on_eval:
            # initialize list if unseen
            num_rounds = self.options['num_round'] + 1
            self.personalized_loss_on_eval[cid] = [0] * num_rounds
            self.personalized_acc_on_eval[cid] = [0] * num_rounds

        self.personalized_loss_on_eval[cid][round_i] = loss
        self.personalized_acc_on_eval[cid][round_i] = acc

        # Write scalar to tensorboard under a per-client tag
        self.eval_writer.add_scalar(f'personalized/test_loss/client_{cid}', loss, round_i)
        self.eval_writer.add_scalar(f'personalized/test_acc/client_{cid}', acc, round_i)

    def update_personalized_aggregate(self, round_i, losses_list, accs_list):
        """Record aggregated personalized metrics (mean and std over clients) for this round.

        losses_list, accs_list: lists of per-client loss/acc (floats)
        """
        if len(losses_list) == 0:
            mean_loss = 0.0
        else:
            mean_loss = float(np.mean(losses_list))
        if len(accs_list) == 0:
            mean_acc = 0.0
            std_acc = 0.0
        else:
            mean_acc = float(np.mean(accs_list))
            std_acc = float(np.std(accs_list))

        self.personalized_mean_loss[round_i] = mean_loss
        self.personalized_mean_acc[round_i] = mean_acc
        self.personalized_std_acc[round_i] = std_acc

        # write to tensorboard for easy visualization
        self.eval_writer.add_scalar('personalized/mean_test_loss', mean_loss, round_i)
        self.eval_writer.add_scalar('personalized/mean_test_acc', mean_acc, round_i)
        self.eval_writer.add_scalar('personalized/std_test_acc', std_acc, round_i)
        # Also write combined charts comparing global vs personalized on the same plot
        try:
            global_acc = self.acc_on_eval_data[round_i] if len(self.acc_on_eval_data) > round_i else 0.0
            global_loss = self.loss_on_eval_data[round_i] if len(self.loss_on_eval_data) > round_i else 0.0
            # comparison chart for accuracy (two series in one plot)
            self.eval_writer.add_scalars('comparison/test_acc',
                                         {'global': float(global_acc), 'personalized': float(mean_acc)},
                                         round_i)
            # comparison chart for loss
            self.eval_writer.add_scalars('comparison/test_loss',
                                         {'global': float(global_loss), 'personalized': float(mean_loss)},
                                         round_i)
            # Also add single-series scalars for compatibility with simple names
        except Exception:
            # if add_scalars not supported by writer implementation, ignore
            pass

    def update_collab_eval_stats(self, round_i, cid, loss, acc):
        if cid not in self.collab_loss_on_eval:
            num_rounds = self.options['num_round'] + 1
            self.collab_loss_on_eval[cid] = [0] * num_rounds
            self.collab_acc_on_eval[cid] = [0] * num_rounds

        self.collab_loss_on_eval[cid][round_i] = loss
        self.collab_acc_on_eval[cid][round_i] = acc

        self.eval_writer.add_scalar(f'collab/test_loss/client_{cid}', loss, round_i)
        self.eval_writer.add_scalar(f'collab/test_acc/client_{cid}', acc, round_i)

    def update_collab_aggregate(self, round_i, losses_list, accs_list):
        if len(losses_list) == 0:
            mean_loss = 0.0
        else:
            mean_loss = float(np.mean(losses_list))
        if len(accs_list) == 0:
            mean_acc = 0.0
            std_acc = 0.0
        else:
            mean_acc = float(np.mean(accs_list))
            std_acc = float(np.std(accs_list))

        self.collab_mean_loss[round_i] = mean_loss
        self.collab_mean_acc[round_i] = mean_acc
        self.collab_std_acc[round_i] = std_acc

        self.eval_writer.add_scalar('collab/mean_test_loss', mean_loss, round_i)
        self.eval_writer.add_scalar('collab/mean_test_acc', mean_acc, round_i)
        self.eval_writer.add_scalar('collab/std_test_acc', std_acc, round_i)

    def write(self):
        metrics = dict()

        # String
        metrics['dataset'] = self.options['dataset']
        metrics['num_round'] = self.options['num_round']
        metrics['eval_every'] = self.options['eval_every']
        metrics['lr'] = self.options['lr']
        metrics['num_epoch'] = self.options['num_epoch']
        metrics['batch_size'] = self.options['batch_size']

        metrics['loss_on_train_data'] = self.loss_on_train_data
        metrics['acc_on_train_data'] = self.acc_on_train_data
        metrics['gradnorm_on_train_data'] = self.gradnorm_on_train_data
        metrics['graddiff_on_train_data'] = self.graddiff_on_train_data

        metrics['loss_on_eval_data'] = self.loss_on_eval_data
        metrics['acc_on_eval_data'] = self.acc_on_eval_data
        metrics['personalized_mean_loss'] = self.personalized_mean_loss
        metrics['personalized_mean_acc'] = self.personalized_mean_acc
        metrics['personalized_std_acc'] = self.personalized_std_acc
        metrics['collab_mean_loss'] = self.collab_mean_loss
        metrics['collab_mean_acc'] = self.collab_mean_acc
        metrics['collab_std_acc'] = self.collab_std_acc

        metrics['personalized_loss_on_eval'] = self.personalized_loss_on_eval
        metrics['personalized_acc_on_eval'] = self.personalized_acc_on_eval
        metrics['collab_loss_on_eval'] = self.collab_loss_on_eval
        metrics['collab_acc_on_eval'] = self.collab_acc_on_eval

        # Dict(key=cid, value=list(stats for each round))
        metrics['client_computations'] = self.client_computations
        metrics['bytes_written'] = self.bytes_written
        metrics['bytes_read'] = self.bytes_read

        metrics_dir = os.path.join(self.result_path, self.exp_name, 'metrics.json')

        with open(metrics_dir, 'w') as ouf:
            json.dump(metrics, ouf, indent=2)
