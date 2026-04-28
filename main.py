import numpy as np
import argparse
import importlib
import torch
import os
import time
import sys
import traceback
import contextlib

# Default to a headless backend for all plotting paths in training scripts.
os.environ.setdefault('MPLBACKEND', 'Agg')

from src.utils.worker_utils import read_data, MiniDataset
from config import OPTIMIZERS, DATASETS, MODEL_PARAMS, TRAINERS


class TeeStream:
    """Write stream output to terminal and a log file at the same time."""

    def __init__(self, console_stream, log_stream):
        self.console_stream = console_stream
        self.log_stream = log_stream

    def write(self, data):
        self.console_stream.write(data)
        self.log_stream.write(data)
        return len(data)

    def flush(self):
        self.console_stream.flush()
        self.log_stream.flush()

    def isatty(self):
        return self.console_stream.isatty()


def _build_default_run_name(options):
    return '{}_{}_{}_{}_sd{}_lr{}_ep{}_bs{}'.format(
        time.strftime('%Y-%m-%dT%H-%M-%S'),
        options['algo'],
        options['dataset'],
        options['model'],
        options['seed'],
        options['lr'],
        options['num_epoch'],
        options['batch_size']
    )


def _sanitize_run_name(run_name):
    return str(run_name).strip().replace('/', '_').replace('\\', '_').replace(' ', '_')


def _preview_run_name_from_argv(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--algo', type=str, default='fedavg')
    parser.add_argument('--dataset', type=str, default='mnist_all_data_0_equal_niid')
    parser.add_argument('--model', type=str, default='logistic')
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--num_epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    known, _ = parser.parse_known_args(argv)
    opt = vars(known)
    raw_run_name = str(opt.get('run_name', '')).strip()
    if raw_run_name == '':
        return _build_default_run_name(opt)
    return _sanitize_run_name(raw_run_name)


def read_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--algo',
                        help='name of trainer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='fedavg')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        default='mnist_all_data_0_equal_niid')
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='logistic')
    parser.add_argument('--run_name',
                        help='custom output folder name under ./result; if empty, auto-generate by params',
                        type=str,
                        default='')
    parser.add_argument('--wd',
                        help='weight decay parameter;',
                        type=float,
                        default=0.001)
    parser.add_argument('--gpu',
                        action='store_true',
                        default=False,
                        help='use gpu (default: False)')
    parser.add_argument('--noprint',
                        action='store_true',
                        default=False,
                        help='whether to print inner result (default: False)')
    parser.add_argument('--simple_average',
                        action='store_true',
                        default=False,
                        help='whether to average local solutions according to sample numbers (default: False)')
    parser.add_argument('--device',
                        help='selected CUDA device',
                        default=0,
                        type=int)
    parser.add_argument('--num_round',
                        help='number of rounds to simulate;',
                        type=int,
                        default=20)
    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=5)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=10)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=32)
    parser.add_argument('--num_epoch',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=5)
    parser.add_argument('--personal_num_epoch',
                        help='number of epochs for Ditto personalized proximal update; '
                             'if <=0 or unset, fallback to --num_epoch',
                        type=int,
                        default=None)
    parser.add_argument('--lr',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.1)
    parser.add_argument('--mu',
                        help='Proximal regularization coefficient (mu); in stp_fedcl it is applied on the global branch.',
                        type=float,
                        default=0.5)
    parser.add_argument('--lambda_p',
                        help='Ditto personalization regularizer (lambda_p);',
                        type=float,
                        default=0.1)
    parser.add_argument('--eta',
                        help='pFedMe personal model inner-step size (eta);',
                        type=float,
                        default=0.5)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--dis',
                        help='add more information;',
                        type=str,
                        default='')
    parser.add_argument('--server_beta',
                        help='pFedMe server model mixing parameter (server_beta);',
                        type=float,
                        default=1.0)
    parser.add_argument('--sequential_cl',
                        action='store_true',
                        default=False,
                        help='enable sequential continual-learning training across tasks')
    parser.add_argument('--num_tasks',
                        help='number of sequential tasks for continual learning',
                        type=int,
                        default=3)
    parser.add_argument('--lambda_old',
                        help='L2 regularization coefficient to previous task global model',
                        type=float,
                        default=0.0)
    parser.add_argument('--lambda_ewc',
                        help='EWC regularization coefficient for fedavg_ewc (sequential CL)',
                        type=float,
                        default=0.0)
    parser.add_argument('--ewc_fisher_samples',
                        help='max local samples per client for diagonal Fisher estimation at task boundary',
                        type=int,
                        default=128)
    parser.add_argument('--lambda_s',
                        help='short-term anchor regularization coefficient (to client prev model)',
                        type=float,
                        default=0.0)
    parser.add_argument('--lambda_l',
                        help='long-term anchor regularization coefficient (to client EMA model)',
                        type=float,
                        default=0.0)
    parser.add_argument('--alpha',
                        help='EMA momentum for client long-term memory',
                        type=float,
                        default=0.9)
    parser.add_argument('--beta_mode',
                        help='collaboration beta mode: adaptive_search or fixed',
                        type=str,
                        default='adaptive_search')
    parser.add_argument('--beta_fixed',
                        help='fixed collaboration beta when beta_mode=fixed',
                        type=float,
                        default=0.5)
    parser.add_argument('--beta_candidates',
                        help='comma-separated beta candidates for adaptive search',
                        type=str,
                        default='0,0.25,0.5,0.75,1')
    parser.add_argument('--beta_val_ratio',
                        help='validation split ratio on each client for beta search',
                        type=float,
                        default=0.1)
    parser.add_argument('--log_reg_terms',
                        action='store_true',
                        default=False,
                        help='print regularization term values during local personalized training')
    parser.add_argument('--reg_log_every',
                        help='print regularization terms every N local mini-batches (when --log_reg_terms is set)',
                        type=int,
                        default=20)
    parser.add_argument('--task_aware',
                        action='store_true',
                        default=True,
                        help='enable task-aware logit slicing/remap (TIL setting); default False for class-incremental eval')
    parsed = parser.parse_args()
    options = parsed.__dict__
    raw_run_name = str(options.get('run_name', '')).strip()
    if raw_run_name == '':
        options['run_name'] = _build_default_run_name(options)
    else:
        safe_run_name = _sanitize_run_name(raw_run_name)
        if safe_run_name != raw_run_name:
            print(f"Warning[read_options]: normalized --run_name from `{raw_run_name}` to `{safe_run_name}`")
        options['run_name'] = safe_run_name

    # For a single command run, default output is one folder under ./result/<run_name>.
    # Sequential CL further stores each task under ./result/<run_name>/task{idx}.
    options['result_path_override'] = './result'
    options['exp_name_override'] = options['run_name']

    options['gpu'] = options['gpu'] and torch.cuda.is_available()
    if options.get('personal_num_epoch', None) is None:
        options['personal_num_epoch'] = int(options['num_epoch'])
    elif int(options['personal_num_epoch']) <= 0:
        print(f"Warning[read_options]: invalid --personal_num_epoch={options['personal_num_epoch']}, fallback to --num_epoch={options['num_epoch']}")
        options['personal_num_epoch'] = int(options['num_epoch'])
    else:
        options['personal_num_epoch'] = int(options['personal_num_epoch'])
    options['beta_mode'] = str(options.get('beta_mode', 'adaptive_search')).lower()
    raw_candidates = str(options.get('beta_candidates', '0,0.25,0.5,0.75,1'))
    try:
        beta_candidates = [float(x.strip()) for x in raw_candidates.split(',') if x.strip() != '']
    except Exception as e:
        print(f"Warning[read_options]: failed to parse --beta_candidates `{raw_candidates}`, fallback to default due to {type(e).__name__}: {e}")
        beta_candidates = [0.0, 0.25, 0.5, 0.75, 1.0]
    beta_candidates = [float(max(0.0, min(1.0, b))) for b in beta_candidates]
    if len(beta_candidates) == 0:
        beta_candidates = [0.0, 0.25, 0.5, 0.75, 1.0]
    options['beta_candidates'] = beta_candidates
    options['ewc_fisher_samples'] = int(max(0, int(options.get('ewc_fisher_samples', 128))))

    if options.get('algo', '') != 'fedavg_ewc':
        options['lambda_ewc'] = 0.0
    if options.get('algo', '') == 'fedavg_ewc' and (not options.get('sequential_cl', False)):
        print('Warning[read_options]: fedavg_ewc is designed for --sequential_cl; current run will behave as FedAvg (no EWC state transfer).')

    # Set seeds
    np.random.seed(1 + options['seed'])
    torch.manual_seed(12 + options['seed'])
    if options['gpu']:
        torch.cuda.manual_seed_all(123 + options['seed'])

    # read data
    idx = options['dataset'].find("_")
    if idx != -1:
        dataset_name, sub_data = options['dataset'][:idx], options['dataset'][idx+1:]
    else:
        dataset_name, sub_data = options['dataset'], None
    assert dataset_name in DATASETS, "{} not in dataset {}!".format(dataset_name, DATASETS)

    # Add model arguments
    options.update(MODEL_PARAMS(dataset_name, options['model']))

    # Load selected trainer
    trainer_path = 'src.trainers.%s' % options['algo']
    mod = importlib.import_module(trainer_path)
    trainer_class = getattr(mod, TRAINERS[options['algo']])

    # Print arguments and return
    max_length = max([len(key) for key in options.keys()])
    fmt_string = '\t%' + str(max_length) + 's : %s'
    print('>>> Arguments:')
    for keyPair in sorted(options.items()):
        print(fmt_string % keyPair)

    return options, trainer_class, dataset_name, sub_data


def _evaluate_collaboration_on_client(trainer, client, global_flat, personal_flat, beta):
    """Evaluate mixed logits on one client's eval dataset."""
    from torch.utils.data import DataLoader
    import torch.nn as nn

    dl = DataLoader(client.test_data, batch_size=trainer.batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    total = 0
    total_correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for x, y in dl:
            x = trainer.worker.flatten_data(x)
            if trainer.gpu:
                x, y = x.cuda(), y.cuda()

            trainer.worker.set_flat_model_params(global_flat)
            pred_g = trainer.worker.model(x)
            trainer.worker.set_flat_model_params(personal_flat)
            pred_p = trainer.worker.model(x)

            pred_mix = (1.0 - float(beta)) * pred_g + float(beta) * pred_p
            pred_mix, y_local = trainer.worker._apply_task_aware_logits_labels(pred_mix, y)

            loss = criterion(pred_mix, y_local)
            _, pred_label = torch.max(pred_mix, 1)
            correct = pred_label.eq(y_local).sum().item()

            bs = y_local.size(0)
            total += bs
            total_correct += correct
            total_loss += float(loss.item()) * bs

    trainer.worker.set_flat_model_params(global_flat)

    if total == 0:
        return 0.0, 0.0
    return float(total_correct) / float(total), float(total_loss) / float(total)


def _post_train_eval_and_save(trainer):
    """Evaluate final model(s) after training and save artifacts."""
    try:
        import json
        import matplotlib.pyplot as plt

        result_dir = os.path.join(trainer.metrics.result_path, trainer.metrics.exp_name)
        os.makedirs(result_dir, exist_ok=True)

        print('>>> Evaluating final global model on each client test set...')
        client_acc = {}
        classes_per_client = {}
        top1_frac = {}
        label_entropy = {}

        for c in trainer.clients:
            c.set_flat_model_params(trainer.latest_model)
            tot_correct, num_sample, loss = c.local_test(use_eval_data=True)
            acc = float(tot_correct) / float(num_sample) if num_sample > 0 else 0.0
            client_acc[int(c.cid)] = acc

            try:
                labels = c.train_data.labels
            except Exception as e:
                print(f"Warning[_post_train_eval_and_save]: failed to read train labels for cid={c.cid} due to {type(e).__name__}: {e}")
                labels = None

            if labels is not None and len(labels) > 0:
                vals, counts = np.unique(labels, return_counts=True)
                classes_per_client[int(c.cid)] = int(len(vals))
                probs = counts.astype('float64') / counts.sum()
                top1 = probs.max()
                top1_frac[int(c.cid)] = float(top1)
                ent = -np.sum([p * np.log(p + 1e-12) for p in probs])
                label_entropy[int(c.cid)] = float(ent)
            else:
                classes_per_client[int(c.cid)] = 0
                top1_frac[int(c.cid)] = 0.0
                label_entropy[int(c.cid)] = 0.0

        acc_json_path = os.path.join(result_dir, 'client_acc.json')
        with open(acc_json_path, 'w') as outf:
            json.dump({'client_acc': client_acc,
                       'classes_per_client': classes_per_client,
                       'top1_frac': top1_frac,
                       'label_entropy': label_entropy}, outf)
        print(f'>>> Saved per-client accuracy and stats to {acc_json_path}')

        ids_sorted = sorted(client_acc.keys())
        accs = np.array([client_acc[i] for i in ids_sorted])
        np.save(os.path.join(result_dir, 'client_acc.npy'), accs)

        stats = dict()
        if accs.size > 0:
            stats['mean'] = float(np.mean(accs))
            stats['std'] = float(np.std(accs))
            stats['min'] = float(np.min(accs))
            stats['max'] = float(np.max(accs))
            p10, p90 = np.percentile(accs, [10, 90])
            stats['10th_percentile'] = float(p10)
            stats['90th_percentile'] = float(p90)
        else:
            stats['mean'] = stats['std'] = stats['min'] = stats['max'] = 0.0
            stats['10th_percentile'] = stats['90th_percentile'] = 0.0

        stats_path = os.path.join(result_dir, 'client_acc_stats.json')
        with open(stats_path, 'w') as sf:
            json.dump(stats, sf)
        print('>>> Client accuracy stats:')
        print(stats)

        plt.figure(figsize=(12, 4))
        plt.bar(ids_sorted, accs)
        plt.xlabel('Client ID')
        plt.ylabel('Accuracy')
        plt.title('Per-client Test Accuracy (global model)')
        plt.tight_layout()
        barpath = os.path.join(result_dir, 'client_acc_bar.png')
        plt.savefig(barpath, dpi=200)
        plt.close()
        print(f'>>> Saved client accuracy bar plot to {barpath}')

        plt.figure(figsize=(8, 4))
        acc_sorted = np.sort(accs)
        plt.plot(acc_sorted, marker='o')
        plt.xlabel('Client (sorted)')
        plt.ylabel('Accuracy')
        plt.title('Per-client Test Accuracy (sorted)')
        plt.tight_layout()
        sortpath = os.path.join(result_dir, 'client_acc_sorted.png')
        plt.savefig(sortpath, dpi=200)
        plt.close()
        print(f'>>> Saved sorted client accuracy plot to {sortpath}')

        bundle_path = os.path.join(result_dir, 'client_acc_bundle.json')
        try:
            with open(bundle_path, 'w') as bf:
                json.dump({'client_ids': ids_sorted, 'client_acc': accs.tolist(), 'stats': stats}, bf)
            print(f'>>> Saved client_acc bundle to {bundle_path}')
        except Exception as e:
            print(f"Warning[_post_train_eval_and_save]: failed to write bundle file `{bundle_path}` due to {type(e).__name__}: {e}")

        try:
            if hasattr(trainer, 'personal_models') and isinstance(trainer.personal_models, dict):
                print('>>> Evaluating personalized models on each client test set...')
                personal_acc = {}
                personal_loss = {}
                for c in trainer.clients:
                    p = trainer.personal_models.get(c.cid, trainer.latest_model)
                    try:
                        c.set_flat_model_params(p)
                        tot_correct, num_sample, loss = c.local_test(use_eval_data=True)
                        acc = float(tot_correct) / float(num_sample) if num_sample > 0 else 0.0
                        personal_acc[int(c.cid)] = acc
                        personal_loss[int(c.cid)] = float(loss) / float(num_sample) if num_sample > 0 else 0.0
                    except Exception as e:
                        print(f"Warning[_post_train_eval_and_save]: personalized eval fallback cid={c.cid} due to {type(e).__name__}: {e}")
                        personal_acc[int(c.cid)] = 0.0
                        personal_loss[int(c.cid)] = 0.0

                personal_json_path = os.path.join(result_dir, 'client_acc_personal.json')
                with open(personal_json_path, 'w') as outf:
                    json.dump({'client_acc_personal': personal_acc}, outf)
                print(f'>>> Saved per-client personalized accuracy to {personal_json_path}')

                ids_sorted_p = sorted(personal_acc.keys())
                accs_p = np.array([personal_acc[i] for i in ids_sorted_p])
                np.save(os.path.join(result_dir, 'client_acc_personal.npy'), accs_p)

                pstats = dict()
                if accs_p.size > 0:
                    pstats['mean'] = float(np.mean(accs_p))
                    pstats['std'] = float(np.std(accs_p))
                    pstats['min'] = float(np.min(accs_p))
                    pstats['max'] = float(np.max(accs_p))
                    p10, p90 = np.percentile(accs_p, [10, 90])
                    pstats['10th_percentile'] = float(p10)
                    pstats['90th_percentile'] = float(p90)
                else:
                    pstats['mean'] = pstats['std'] = pstats['min'] = pstats['max'] = 0.0
                    pstats['10th_percentile'] = pstats['90th_percentile'] = 0.0

                pstats_path = os.path.join(result_dir, 'client_acc_personal_stats.json')
                with open(pstats_path, 'w') as sf:
                    json.dump(pstats, sf)

                try:
                    plt.figure(figsize=(12, 4))
                    plt.bar(ids_sorted_p, accs_p)
                    plt.xlabel('Client ID')
                    plt.ylabel('Personalized Accuracy')
                    plt.title('Per-client Test Accuracy (personalized model)')
                    plt.tight_layout()
                    barpath_p = os.path.join(result_dir, 'client_acc_personal_bar.png')
                    plt.savefig(barpath_p, dpi=200)
                    plt.close()
                except Exception as e:
                    print(f"Warning[_post_train_eval_and_save]: failed to save personalized bar plot due to {type(e).__name__}: {e}")

                try:
                    plt.figure(figsize=(8, 4))
                    acc_sorted_p = np.sort(accs_p)
                    plt.plot(acc_sorted_p, marker='o')
                    plt.xlabel('Client (sorted)')
                    plt.ylabel('Personalized Accuracy')
                    plt.title('Per-client Personalized Test Accuracy (sorted)')
                    plt.tight_layout()
                    sortpath_p = os.path.join(result_dir, 'client_acc_personal_sorted.png')
                    plt.savefig(sortpath_p, dpi=200)
                    plt.close()
                except Exception as e:
                    print(f"Warning[_post_train_eval_and_save]: failed to save personalized sorted plot due to {type(e).__name__}: {e}")

                try:
                    with open(bundle_path, 'r') as bf:
                        bund = json.load(bf)
                except Exception as e:
                    print(f"Warning[_post_train_eval_and_save]: failed to read bundle file `{bundle_path}`, rebuilding due to {type(e).__name__}: {e}")
                    bund = {'client_ids': ids_sorted, 'client_acc': accs.tolist(), 'stats': stats}
                bund.update({'client_acc_personal': accs_p.tolist(), 'personal_stats': pstats})

                # Collaboration evaluation (global + personal with client-specific beta)
                if hasattr(trainer, 'client_betas') and isinstance(trainer.client_betas, dict):
                    print('>>> Evaluating collaborative outputs on each client test set...')
                    collab_acc = {}
                    collab_loss = {}
                    client_beta = {}
                    global_flat = trainer.latest_model.detach().clone()

                    for c in trainer.clients:
                        cid = int(c.cid)
                        beta = float(trainer.client_betas.get(c.cid, 0.5))
                        p = trainer.personal_models.get(c.cid, trainer.latest_model).detach().clone()
                        try:
                            acc_c, loss_c = _evaluate_collaboration_on_client(
                                trainer, c, global_flat, p, beta
                            )
                            collab_acc[cid] = float(acc_c)
                            collab_loss[cid] = float(loss_c)
                            client_beta[cid] = beta
                        except Exception as e:
                            print(f"Warning[_post_train_eval_and_save]: collaborative eval fallback cid={cid} due to {type(e).__name__}: {e}")
                            collab_acc[cid] = 0.0
                            collab_loss[cid] = 0.0
                            client_beta[cid] = beta

                    collab_json_path = os.path.join(result_dir, 'client_acc_collab.json')
                    with open(collab_json_path, 'w') as outf:
                        json.dump({'client_acc_collab': collab_acc, 'client_loss_collab': collab_loss}, outf)
                    np.save(os.path.join(result_dir, 'client_acc_collab.npy'),
                            np.array([collab_acc[i] for i in sorted(collab_acc.keys())]))

                    beta_json_path = os.path.join(result_dir, 'client_beta.json')
                    with open(beta_json_path, 'w') as outf:
                        json.dump({'client_beta': client_beta}, outf)

                    collab_arr = np.array(list(collab_acc.values()), dtype=np.float64)
                    collab_stats = {
                        'mean': float(np.mean(collab_arr)) if collab_arr.size > 0 else 0.0,
                        'std': float(np.std(collab_arr)) if collab_arr.size > 0 else 0.0,
                        'min': float(np.min(collab_arr)) if collab_arr.size > 0 else 0.0,
                        'max': float(np.max(collab_arr)) if collab_arr.size > 0 else 0.0
                    }

                    try:
                        final_round = int(trainer.num_round)
                        for c in trainer.clients:
                            cid = int(c.cid)
                            trainer.metrics.update_collab_eval_stats(
                                final_round, cid, float(collab_loss.get(cid, 0.0)), float(collab_acc.get(cid, 0.0))
                            )
                        trainer.metrics.update_collab_aggregate(
                            final_round,
                            [float(v) for v in collab_loss.values()],
                            [float(v) for v in collab_acc.values()]
                        )
                    except Exception as e:
                        print(f"Warning[_post_train_eval_and_save]: failed to update collab metrics at final_round={final_round} due to {type(e).__name__}: {e}")

                    bund.update({
                        'client_acc_collab': [collab_acc[i] for i in sorted(collab_acc.keys())],
                        'client_beta': [client_beta[i] for i in sorted(client_beta.keys())],
                        'collab_stats': collab_stats
                    })

                try:
                    with open(bundle_path, 'w') as bf:
                        json.dump(bund, bf)
                except Exception as e:
                    print(f"Warning[_post_train_eval_and_save]: failed to persist updated bundle `{bundle_path}` due to {type(e).__name__}: {e}")
        except Exception as e:
            print('Error during personalized model evaluation:', e)

        try:
            trainer.metrics.write()
        except Exception as e:
            print(f"Warning[_post_train_eval_and_save]: trainer.metrics.write failed due to {type(e).__name__}: {e}")

    except Exception as e:
        print(f'Error during post-train per-client evaluation: {e}')


def _split_dataset_into_tasks(all_data_info, num_tasks=3):
    """Split a FL dataset tuple into sequential label tasks.

    Returns:
        task_datasets: list of tuples (users, groups, train_data, test_data)
        task_label_lists: list of label lists per task
    """
    users, groups, train_data, test_data = all_data_info

    all_labels = []
    for u in users:
        labels = np.array(train_data[u].labels).astype('int64')
        if labels.size > 0:
            all_labels.append(labels)
    if len(all_labels) == 0:
        raise ValueError(
            'Cannot build CL tasks: no training labels found. '
            'Likely causes: (1) dataset key filter matched no .pkl files, '
            '(2) selected dataset has empty train labels. '
            'Please verify --dataset and files under data/<dataset>/data/train.'
        )

    unique_labels = np.unique(np.concatenate(all_labels))
    split_labels = [np.array(x, dtype=np.int64) for x in np.array_split(unique_labels, num_tasks)]

    task_datasets = []
    task_label_lists = []

    for task_i, task_labels in enumerate(split_labels, start=1):
        label_set = set([int(x) for x in task_labels.tolist()])
        task_users = []
        task_train = {}
        task_test = {}

        for u in users:
            tr = train_data[u]
            te = test_data[u]

            tr_y = np.array(tr.labels).astype('int64')
            te_y = np.array(te.labels).astype('int64')

            tr_mask = np.array([int(y) in label_set for y in tr_y])
            te_mask = np.array([int(y) in label_set for y in te_y])

            if tr_mask.sum() == 0 or te_mask.sum() == 0:
                continue

            tr_x = np.array(tr.data)[tr_mask]
            tr_y_f = tr_y[tr_mask]
            te_x = np.array(te.data)[te_mask]
            te_y_f = te_y[te_mask]

            task_train[u] = MiniDataset(tr_x, tr_y_f)
            task_test[u] = MiniDataset(te_x, te_y_f)
            task_users.append(u)

        if len(task_users) == 0:
            raise ValueError(f'Task-{task_i} has no clients with both train/test samples.')

        task_users = sorted(task_users)
        task_datasets.append((task_users, [], task_train, task_test))
        task_label_lists.append(sorted(list(label_set)))

    return task_datasets, task_label_lists


def _evaluate_global_on_dataset(trainer, dataset_tuple, model_flat, active_labels=None):
    """Evaluate a flat global model on a task dataset tuple.

    Returns a dict with weighted accuracy/loss and sample count.
    """
    from torch.utils.data import DataLoader

    users, _, _, test_data = dataset_tuple
    trainer.worker.set_flat_model_params(model_flat)

    # Temporarily switch worker to the evaluated task label space
    old_active_labels = trainer.worker.options.get('active_labels', None)
    old_label_map = trainer.worker.options.get('label_map', None)
    if bool(trainer.worker.options.get('task_aware', False)) and active_labels is not None:
        trainer.worker.options['active_labels'] = [int(v) for v in active_labels]
        trainer.worker.options['label_map'] = {
            int(g): int(i) for i, g in enumerate(active_labels)
        }

    tot_correct_sum = 0.0
    tot_loss_sum = 0.0
    tot_num = 0

    try:
        for u in users:
            ds = test_data[u]
            if len(ds) == 0:
                continue
            dl = DataLoader(ds, batch_size=trainer.batch_size, shuffle=False)
            tot_correct, loss = trainer.worker.local_test(dl)
            num = len(ds)
            tot_correct_sum += float(tot_correct)
            tot_loss_sum += float(loss)
            tot_num += int(num)
    finally:
        # Restore previous worker task context
        trainer.worker.options['active_labels'] = old_active_labels
        trainer.worker.options['label_map'] = old_label_map

    if tot_num == 0:
        return {'acc': 0.0, 'loss': 0.0, 'num_samples': 0}

    return {
        'acc': float(tot_correct_sum / float(tot_num)),
        'loss': float(tot_loss_sum / float(tot_num)),
        'num_samples': int(tot_num)
    }


def _evaluate_personal_on_dataset(trainer, dataset_tuple, personal_models, active_labels=None, weighted=True):
    """Evaluate personalized models on a task dataset tuple.

    For each client, evaluate with its own personalized model, then aggregate across clients.
    By default uses sample-weighted aggregation.
    """
    from torch.utils.data import DataLoader

    users, _, _, test_data = dataset_tuple
    # Build mapping consistent with BaseTrainer.setup_clients user-id conversion.
    cid_to_user = {}
    for u in users:
        if isinstance(u, str) and len(u) >= 5:
            uid = int(u[-5:])
        else:
            uid = int(u)
        cid_to_user[uid] = u

    # Temporarily switch worker to the evaluated task label space.
    old_active_labels = trainer.worker.options.get('active_labels', None)
    old_label_map = trainer.worker.options.get('label_map', None)
    if bool(trainer.worker.options.get('task_aware', False)) and active_labels is not None:
        trainer.worker.options['active_labels'] = [int(v) for v in active_labels]
        trainer.worker.options['label_map'] = {
            int(g): int(i) for i, g in enumerate(active_labels)
        }

    per_client = []
    tot_correct_sum = 0.0
    tot_loss_sum = 0.0
    tot_num = 0

    try:
        for c in trainer.clients:
            if c.cid not in cid_to_user:
                continue
            u = cid_to_user[c.cid]
            ds = test_data.get(u, None)
            if ds is None or len(ds) == 0:
                continue

            p = personal_models.get(c.cid, trainer.latest_model)
            trainer.worker.set_flat_model_params(p)

            dl = DataLoader(ds, batch_size=trainer.batch_size, shuffle=False)
            tot_correct, loss = trainer.worker.local_test(dl)
            num = len(ds)
            acc = float(tot_correct) / float(num) if num > 0 else 0.0
            avg_loss = float(loss) / float(num) if num > 0 else 0.0

            per_client.append({
                'cid': int(c.cid),
                'acc': float(acc),
                'loss': float(avg_loss),
                'num_samples': int(num)
            })

            tot_correct_sum += float(tot_correct)
            tot_loss_sum += float(loss)
            tot_num += int(num)
    finally:
        trainer.worker.options['active_labels'] = old_active_labels
        trainer.worker.options['label_map'] = old_label_map
        trainer.worker.set_flat_model_params(trainer.latest_model)

    if len(per_client) == 0:
        return {'acc': 0.0, 'loss': 0.0, 'num_samples': 0, 'per_client': []}

    if weighted:
        acc = float(tot_correct_sum / float(tot_num)) if tot_num > 0 else 0.0
        loss = float(tot_loss_sum / float(tot_num)) if tot_num > 0 else 0.0
        num_samples = int(tot_num)
    else:
        acc = float(np.mean([v['acc'] for v in per_client]))
        loss = float(np.mean([v['loss'] for v in per_client]))
        num_samples = int(sum([v['num_samples'] for v in per_client]))

    return {
        'acc': float(acc),
        'loss': float(loss),
        'num_samples': int(num_samples),
        'per_client': per_client
    }


def _evaluate_collab_on_dataset(trainer, dataset_tuple, personal_models, active_labels=None, weighted=True):
    """Evaluate collaborative (global/personal beta-mixed) outputs on a task dataset tuple."""
    from torch.utils.data import DataLoader
    import torch.nn as nn

    users, _, _, test_data = dataset_tuple

    # Build mapping consistent with BaseTrainer.setup_clients user-id conversion.
    cid_to_user = {}
    for u in users:
        if isinstance(u, str) and len(u) >= 5:
            uid = int(u[-5:])
        else:
            uid = int(u)
        cid_to_user[uid] = u

    old_active_labels = trainer.worker.options.get('active_labels', None)
    old_label_map = trainer.worker.options.get('label_map', None)
    if bool(trainer.worker.options.get('task_aware', False)) and active_labels is not None:
        trainer.worker.options['active_labels'] = [int(v) for v in active_labels]
        trainer.worker.options['label_map'] = {
            int(g): int(i) for i, g in enumerate(active_labels)
        }

    global_flat = trainer.latest_model.detach().clone()
    criterion = nn.CrossEntropyLoss()
    per_client = []
    tot_correct_sum = 0.0
    tot_loss_sum = 0.0
    tot_num = 0

    try:
        for c in trainer.clients:
            if c.cid not in cid_to_user:
                continue
            u = cid_to_user[c.cid]
            ds = test_data.get(u, None)
            if ds is None or len(ds) == 0:
                continue

            personal_flat = personal_models.get(c.cid, trainer.latest_model).detach().clone()
            beta = float(getattr(trainer, 'client_betas', {}).get(c.cid, trainer.options.get('beta_fixed', 0.5)))

            dl = DataLoader(ds, batch_size=trainer.batch_size, shuffle=False)
            total = 0
            total_correct = 0
            total_loss = 0.0

            with torch.no_grad():
                for x, y in dl:
                    x = trainer.worker.flatten_data(x)
                    if trainer.gpu:
                        x, y = x.cuda(), y.cuda()

                    trainer.worker.set_flat_model_params(global_flat)
                    pred_g = trainer.worker.model(x)
                    trainer.worker.set_flat_model_params(personal_flat)
                    pred_p = trainer.worker.model(x)

                    pred_mix = (1.0 - beta) * pred_g + beta * pred_p
                    pred_mix, y_local = trainer.worker._apply_task_aware_logits_labels(pred_mix, y)

                    loss = criterion(pred_mix, y_local)
                    _, pred_label = torch.max(pred_mix, 1)
                    correct = pred_label.eq(y_local).sum().item()

                    bs = y_local.size(0)
                    total += bs
                    total_correct += correct
                    total_loss += float(loss.item()) * bs

            if total == 0:
                continue

            acc = float(total_correct) / float(total)
            avg_loss = float(total_loss) / float(total)
            per_client.append({
                'cid': int(c.cid),
                'beta': float(beta),
                'acc': float(acc),
                'loss': float(avg_loss),
                'num_samples': int(total)
            })

            tot_correct_sum += float(total_correct)
            tot_loss_sum += float(total_loss)
            tot_num += int(total)
    finally:
        trainer.worker.options['active_labels'] = old_active_labels
        trainer.worker.options['label_map'] = old_label_map
        trainer.worker.set_flat_model_params(trainer.latest_model)

    if len(per_client) == 0:
        return {'acc': 0.0, 'loss': 0.0, 'num_samples': 0, 'per_client': []}

    if weighted:
        acc = float(tot_correct_sum / float(tot_num)) if tot_num > 0 else 0.0
        loss = float(tot_loss_sum / float(tot_num)) if tot_num > 0 else 0.0
        num_samples = int(tot_num)
    else:
        acc = float(np.mean([v['acc'] for v in per_client]))
        loss = float(np.mean([v['loss'] for v in per_client]))
        num_samples = int(sum([v['num_samples'] for v in per_client]))

    return {
        'acc': float(acc),
        'loss': float(loss),
        'num_samples': int(num_samples),
        'per_client': per_client
    }


def _regenerate_cl_matrices_from_summary_json(summary_path):
    """Regenerate visualization matrices from saved sequential CL summary JSON.

    This helps quickly inspect seen-task accuracy and forgetting from saved artifacts.
    """
    try:
        import json
        import matplotlib.pyplot as plt

        with open(summary_path, 'r') as f:
            summary = json.load(f)
        algo_name = summary.get('options', {}).get('algo', 'unknown')

        base = os.path.splitext(summary_path)[0]

        def _regen_one(acc_key, forgetting_key, tag):
            acc_mat = np.array(summary.get(acc_key, []), dtype=np.float64)
            if acc_mat.size == 0:
                print(f'Warning: empty {acc_key} in {summary_path}')
                return

            num_tasks = acc_mat.shape[0]
            tag_suffix = f'_{tag}' if tag else ''

            # 1) Accuracy matrix heatmap
            try:
                plt.figure(figsize=(6, 5))
                vis = np.where(np.isnan(acc_mat), 0.0, acc_mat)
                im = plt.imshow(vis, cmap='viridis', vmin=0.0, vmax=1.0)
                for i in range(vis.shape[0]):
                    for j in range(vis.shape[1]):
                        txt = '-' if np.isnan(acc_mat[i, j]) else f"{acc_mat[i, j]:.3f}"
                        plt.text(j, i, txt, ha='center', va='center', color='white', fontsize=9)
                plt.colorbar(im, fraction=0.046, pad=0.04)
                ticks = np.arange(num_tasks)
                labels = [f'T{i+1}' for i in ticks]
                plt.xticks(ticks, labels)
                plt.yticks(ticks, labels)
                plt.xlabel('Evaluated Task')
                plt.ylabel('After Training Task')
                plt.title(f'[{algo_name}] CL Seen-Task Accuracy Matrix ({tag})')
                plt.tight_layout()
                acc_png = base + f'_regen_acc_matrix{tag_suffix}.png'
                plt.savefig(acc_png, dpi=220)
                plt.close()
                print(f'>>> Saved regenerated accuracy matrix to {acc_png}')
            except Exception as e:
                print(f'Warning: failed to regenerate acc heatmap ({tag}): {e}')

            # 2) Combined matrix: [accuracy matrix; forgetting row]
            forgetting_dict = summary.get(forgetting_key, {})
            forgetting_row = np.full((1, num_tasks), np.nan, dtype=np.float64)
            for j in range(num_tasks):
                v = forgetting_dict.get(f'task_{j + 1}', None)
                if v is None:
                    continue
                try:
                    forgetting_row[0, j] = float(v)
                except Exception as e:
                    print(f"Warning[_regenerate_cl_matrices_from_summary_json]: failed to parse {forgetting_key} for task_{j + 1} due to {type(e).__name__}: {e}")

            combined = np.vstack([acc_mat, forgetting_row])

            npy_path = base + f'_acc_forgetting_matrix{tag_suffix}.npy'
            np.save(npy_path, combined)

            csv_path = base + f'_acc_forgetting_matrix{tag_suffix}.csv'
            with open(csv_path, 'w') as cf:
                header = ['stage/eval'] + [f'T{j+1}' for j in range(num_tasks)]
                cf.write(','.join(header) + '\n')
                for i in range(num_tasks):
                    vals = ['nan' if np.isnan(x) else f'{x:.6f}' for x in combined[i]]
                    cf.write(','.join([f'after_T{i+1}'] + vals) + '\n')
                vals = ['nan' if np.isnan(x) else f'{x:.6f}' for x in combined[-1]]
                cf.write(','.join(['forgetting'] + vals) + '\n')

            try:
                plt.figure(figsize=(6.5, 5.5))
                vis2 = np.where(np.isnan(combined), 0.0, combined)
                vmax = max(1.0, float(np.nanmax(vis2)) if vis2.size > 0 else 1.0)
                im = plt.imshow(vis2, cmap='magma', vmin=0.0, vmax=vmax)
                for i in range(vis2.shape[0]):
                    for j in range(vis2.shape[1]):
                        txt = '-' if np.isnan(combined[i, j]) else f"{combined[i, j]:.3f}"
                        plt.text(j, i, txt, ha='center', va='center', color='white', fontsize=9)
                plt.colorbar(im, fraction=0.046, pad=0.04)
                xticks = np.arange(num_tasks)
                yticks = np.arange(num_tasks + 1)
                xlabels = [f'T{i+1}' for i in xticks]
                ylabels = [f'after_T{i+1}' for i in range(num_tasks)] + ['forgetting']
                plt.xticks(xticks, xlabels)
                plt.yticks(yticks, ylabels)
                plt.xlabel('Evaluated Task')
                plt.ylabel('Training Stage')
                plt.title(f'CL Accuracy + Forgetting Matrix ({tag})')
                plt.tight_layout()
                combo_png = base + f'_regen_acc_forgetting_matrix{tag_suffix}.png'
                plt.savefig(combo_png, dpi=220)
                plt.close()
                print(f'>>> Saved regenerated acc+forgetting matrix to {combo_png}')
                print(f'>>> Saved numeric matrices: {npy_path}, {csv_path}')
            except Exception as e:
                print(f'Warning: failed to regenerate combined matrix heatmap ({tag}): {e}')

        _regen_one('eval_acc_matrix_global', 'forgetting_global', 'global')
        _regen_one('eval_acc_matrix_personalized', 'forgetting_personalized', 'personalized')
        _regen_one('eval_acc_matrix_collab_beta', 'forgetting_collab_beta', 'collab-beta')
        # Backward compatibility for old summaries that only have global keys.
        if 'eval_acc_matrix_global' not in summary and 'eval_acc_matrix' in summary:
            _regen_one('eval_acc_matrix', 'forgetting', 'global')
    except Exception as e:
        print(f'Warning: failed to parse summary JSON for matrix regeneration: {e}')


def _run_sequential_tasks(options, trainer_class, all_data_info):
    """Run sequential CL training across tasks, carrying global model forward."""
    num_tasks = max(1, int(options.get('num_tasks', 3)))
    task_datasets, task_label_lists = _split_dataset_into_tasks(all_data_info, num_tasks=num_tasks)

    print(f'>>> Sequential CL enabled: {len(task_datasets)} tasks')
    for i, ls in enumerate(task_label_lists, start=1):
        print(f'    Task-{i} labels: {ls}')

    run_output_dir = os.path.join('./result', options['run_name'])
    os.makedirs(run_output_dir, exist_ok=True)

    prev_global_model = None
    prev_client_state = None
    task_summaries = []
    eval_acc_matrix_global = np.full((len(task_datasets), len(task_datasets)), np.nan, dtype=np.float64)
    eval_loss_matrix_global = np.full((len(task_datasets), len(task_datasets)), np.nan, dtype=np.float64)
    eval_acc_matrix_personalized = np.full((len(task_datasets), len(task_datasets)), np.nan, dtype=np.float64)
    eval_loss_matrix_personalized = np.full((len(task_datasets), len(task_datasets)), np.nan, dtype=np.float64)
    eval_acc_matrix_collab_beta = np.full((len(task_datasets), len(task_datasets)), np.nan, dtype=np.float64)
    eval_loss_matrix_collab_beta = np.full((len(task_datasets), len(task_datasets)), np.nan, dtype=np.float64)

    for task_idx, task_dataset in enumerate(task_datasets, start=1):
        task_options = dict(options)
        task_options['dataset'] = f"{options['dataset']}_task{task_idx}"
        task_options['dis'] = (task_options.get('dis', '') + f"_cl_task{task_idx}").strip('_')
        task_options['result_path_override'] = run_output_dir
        task_options['exp_name_override'] = f'task{task_idx}'
        task_options['active_labels'] = [int(v) for v in task_label_lists[task_idx - 1]]
        task_options['label_map'] = {
            int(g): int(i) for i, g in enumerate(task_label_lists[task_idx - 1])
        }
        task_options['num_active_classes'] = len(task_label_lists[task_idx - 1])

        # Task-1 has no historical task anchor; disable all CL regularization terms.
        if task_idx == 1:
            task_options['lambda_old'] = 0.0
            task_options['lambda_s'] = 0.0
            task_options['lambda_l'] = 0.0

        # Task-1 has no previous-task anchor; Task-2/3/... use previous global model snapshot
        if prev_global_model is not None:
            task_options['prev_model'] = prev_global_model.detach().clone()
        else:
            task_options['prev_model'] = None

        print('\n' + '=' * 90)
        print(f'>>> Start sequential Task-{task_idx}/{len(task_datasets)}')
        print(
            f">>> Active regularization: "
            f"lambda_old={task_options.get('lambda_old', 0.0)}, "
            f"lambda_s={task_options.get('lambda_s', 0.0)}, "
            f"lambda_l={task_options.get('lambda_l', 0.0)}"
        )
        print(f">>> Prev model loaded: {task_options.get('prev_model', None) is not None}")
        print('=' * 90)

        trainer = trainer_class(task_options, task_dataset)

        if prev_global_model is not None:
            trainer.worker.set_flat_model_params(prev_global_model)
            trainer.latest_model = prev_global_model.detach().clone()
        if prev_client_state is not None and hasattr(trainer, 'import_client_state'):
            try:
                trainer.import_client_state(prev_client_state)
                print(f'>>> Restored trainer client state for Task-{task_idx}')
            except Exception as e:
                print(f'Warning[_run_sequential_tasks]: failed to restore client state for task={task_idx} due to {type(e).__name__}: {e}')

        trainer.train()
        _post_train_eval_and_save(trainer)

        # CL seen-task evaluation: after task t, evaluate on tasks 1..t
        stage_eval = {}
        has_personal_models = hasattr(trainer, 'personal_models') and isinstance(trainer.personal_models, dict)
        has_collab_eval = has_personal_models and hasattr(trainer, 'client_betas') and isinstance(trainer.client_betas, dict)
        if not has_personal_models:
            print(f"Warning[_run_sequential_tasks]: trainer {type(trainer).__name__} has no personal_models; personalized CL matrix will use NaN.")
        if has_personal_models and not has_collab_eval:
            print(f"Warning[_run_sequential_tasks]: trainer {type(trainer).__name__} has no client_betas; collab-beta CL matrix will use NaN.")
        for eval_task_idx in range(task_idx):
            eval_ret_global = _evaluate_global_on_dataset(
                trainer,
                task_datasets[eval_task_idx],
                trainer.latest_model,
                active_labels=task_label_lists[eval_task_idx]
            )
            eval_acc_matrix_global[task_idx - 1, eval_task_idx] = eval_ret_global['acc']
            eval_loss_matrix_global[task_idx - 1, eval_task_idx] = eval_ret_global['loss']

            if has_personal_models:
                eval_ret_personal = _evaluate_personal_on_dataset(
                    trainer,
                    task_datasets[eval_task_idx],
                    trainer.personal_models,
                    active_labels=task_label_lists[eval_task_idx],
                    weighted=True
                )
                eval_acc_matrix_personalized[task_idx - 1, eval_task_idx] = eval_ret_personal['acc']
                eval_loss_matrix_personalized[task_idx - 1, eval_task_idx] = eval_ret_personal['loss']
            else:
                eval_ret_personal = {'acc': np.nan, 'loss': np.nan, 'num_samples': 0, 'per_client': []}

            if has_collab_eval:
                eval_ret_collab = _evaluate_collab_on_dataset(
                    trainer,
                    task_datasets[eval_task_idx],
                    trainer.personal_models,
                    active_labels=task_label_lists[eval_task_idx],
                    weighted=True
                )
                eval_acc_matrix_collab_beta[task_idx - 1, eval_task_idx] = eval_ret_collab['acc']
                eval_loss_matrix_collab_beta[task_idx - 1, eval_task_idx] = eval_ret_collab['loss']
            else:
                eval_ret_collab = {'acc': np.nan, 'loss': np.nan, 'num_samples': 0, 'per_client': []}

            stage_eval[f'task_{eval_task_idx + 1}'] = {
                'global': {
                    'acc': float(eval_ret_global['acc']),
                    'loss': float(eval_ret_global['loss']),
                    'num_samples': int(eval_ret_global['num_samples'])
                },
                'personalized': {
                    'acc': None if np.isnan(eval_ret_personal['acc']) else float(eval_ret_personal['acc']),
                    'loss': None if np.isnan(eval_ret_personal['loss']) else float(eval_ret_personal['loss']),
                    'num_samples': int(eval_ret_personal['num_samples'])
                },
                'collab_beta': {
                    'acc': None if np.isnan(eval_ret_collab['acc']) else float(eval_ret_collab['acc']),
                    'loss': None if np.isnan(eval_ret_collab['loss']) else float(eval_ret_collab['loss']),
                    'num_samples': int(eval_ret_collab['num_samples'])
                }
            }

        # Save stage-level seen-task eval into current task folder
        try:
            import json
            stage_path = os.path.join(trainer.metrics.result_path,
                                      trainer.metrics.exp_name,
                                      f'seen_task_eval_after_task{task_idx}.json')
            with open(stage_path, 'w') as sf:
                json.dump({'after_task': task_idx,
                           'eval': stage_eval}, sf, indent=2)
            print(f'>>> Saved seen-task eval after task{task_idx} to {stage_path}')
        except Exception as e:
            print(f'Warning: failed to save stage seen-task eval for task{task_idx}: {e}')

        prev_global_model = trainer.latest_model.detach().clone()
        if hasattr(trainer, 'export_client_state'):
            try:
                prev_client_state = trainer.export_client_state()
                print(f'>>> Exported trainer client state after Task-{task_idx}')
            except Exception as e:
                prev_client_state = None
                print(f'Warning[_run_sequential_tasks]: failed to export client state for task={task_idx} due to {type(e).__name__}: {e}')

        task_result_dir = os.path.join(trainer.metrics.result_path, trainer.metrics.exp_name)
        task_summaries.append({
            'task_idx': task_idx,
            'labels': task_label_lists[task_idx - 1],
            'num_clients': len(task_dataset[0]),
            'result_dir': task_result_dir,
            'seen_task_eval': stage_eval
        })

    def _compute_forgetting(acc_matrix):
        final_stage = acc_matrix.shape[0] - 1
        forgetting_dict = {}
        forgetting_vals = []
        for j in range(acc_matrix.shape[1]):
            diag = acc_matrix[j, j] if j <= final_stage else np.nan
            final_acc = acc_matrix[final_stage, j]
            if np.isnan(diag) or np.isnan(final_acc):
                forgetting_dict[f'task_{j + 1}'] = None
                continue
            fj = float(diag - final_acc)
            forgetting_dict[f'task_{j + 1}'] = fj
            if j < final_stage:
                forgetting_vals.append(fj)
        mean_f = float(np.mean(forgetting_vals)) if len(forgetting_vals) > 0 else 0.0
        return forgetting_dict, mean_f

    forgetting_global, mean_forgetting_global = _compute_forgetting(eval_acc_matrix_global)
    forgetting_personalized, mean_forgetting_personalized = _compute_forgetting(eval_acc_matrix_personalized)
    forgetting_collab_beta, mean_forgetting_collab_beta = _compute_forgetting(eval_acc_matrix_collab_beta)

    summary_path = os.path.join(run_output_dir, f"{options['dataset']}_cl{len(task_datasets)}_sequential_summary_{time.strftime('%Y-%m-%dT%H-%M-%S')}.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as sf:
        import json
        json.dump({'options': options,
                   'tasks': task_summaries,
                   'labels_per_task': task_label_lists,
                   # Backward-compatible aliases (global metrics)
                   'eval_acc_matrix': eval_acc_matrix_global.tolist(),
                   'eval_loss_matrix': eval_loss_matrix_global.tolist(),
                   'forgetting': forgetting_global,
                   'mean_forgetting': mean_forgetting_global,
                   # Explicit global/personalized matrices
                   'eval_acc_matrix_global': eval_acc_matrix_global.tolist(),
                   'eval_loss_matrix_global': eval_loss_matrix_global.tolist(),
                   'eval_acc_matrix_personalized': eval_acc_matrix_personalized.tolist(),
                   'eval_loss_matrix_personalized': eval_loss_matrix_personalized.tolist(),
                   'eval_acc_matrix_collab_beta': eval_acc_matrix_collab_beta.tolist(),
                   'eval_loss_matrix_collab_beta': eval_loss_matrix_collab_beta.tolist(),
                   'forgetting_global': forgetting_global,
                   'mean_forgetting_global': mean_forgetting_global,
                   'forgetting_personalized': forgetting_personalized,
                   'mean_forgetting_personalized': mean_forgetting_personalized,
                   'forgetting_collab_beta': forgetting_collab_beta,
                   'mean_forgetting_collab_beta': mean_forgetting_collab_beta}, sf, indent=2)
    print(f'>>> Saved sequential CL summary to {summary_path}')

    # Regenerate matrices from the saved summary JSON for easier inspection
    _regenerate_cl_matrices_from_summary_json(summary_path)

    ts = time.strftime('%Y-%m-%dT%H-%M-%S')
    # Save matrix as npy for easier downstream analysis
    np.save(os.path.join(run_output_dir, f"{options['dataset']}_cl_acc_matrix_{ts}.npy"),
        eval_acc_matrix_global)
    np.save(os.path.join(run_output_dir, f"{options['dataset']}_cl_acc_matrix_global_{ts}.npy"),
        eval_acc_matrix_global)
    np.save(os.path.join(run_output_dir, f"{options['dataset']}_cl_acc_matrix_personalized_{ts}.npy"),
        eval_acc_matrix_personalized)
    np.save(os.path.join(run_output_dir, f"{options['dataset']}_cl_acc_matrix_collab_beta_{ts}.npy"),
        eval_acc_matrix_collab_beta)

    def _save_cl_heatmap(acc_matrix, tag):
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 5))
            vis = np.where(np.isnan(acc_matrix), 0.0, acc_matrix)
            im = plt.imshow(vis, cmap='viridis', vmin=0.0, vmax=1.0)
            for i in range(vis.shape[0]):
                for j in range(vis.shape[1]):
                    txt = '-' if np.isnan(acc_matrix[i, j]) else f"{acc_matrix[i, j]:.3f}"
                    plt.text(j, i, txt, ha='center', va='center', color='white', fontsize=9)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel('Evaluated Task')
            plt.ylabel('After Training Task')
            plt.title(f'[{options.get("algo", "unknown")}] Sequential CL Seen-Task Accuracy Matrix ({tag})')
            ticks = np.arange(len(task_datasets))
            labels = [f'T{i+1}' for i in ticks]
            plt.xticks(ticks, labels)
            plt.yticks(ticks, labels)
            plt.tight_layout()
            heat_path = os.path.join(run_output_dir, f"{options['dataset']}_cl_acc_matrix_{tag}_{ts}.png")
            plt.savefig(heat_path, dpi=200)
            plt.close()
            print(f'>>> Saved CL accuracy matrix heatmap ({tag}) to {heat_path}')
        except Exception as e:
            print(f'Warning: failed to save CL matrix heatmap ({tag}): {e}')

    # Keep legacy global heatmap + add explicit global/personalized heatmaps
    _save_cl_heatmap(eval_acc_matrix_global, 'global')
    _save_cl_heatmap(eval_acc_matrix_personalized, 'personalized')
    _save_cl_heatmap(eval_acc_matrix_collab_beta, 'collab-beta')


def main():
    run_root = './result'
    run_name = _preview_run_name_from_argv(sys.argv[1:])
    run_output_dir = os.path.join(run_root, run_name)
    os.makedirs(run_output_dir, exist_ok=True)
    log_path = os.path.join(run_output_dir, 'run.log')

    with open(log_path, 'a', encoding='utf-8') as log_file:
        tee_stdout = TeeStream(sys.__stdout__, log_file)
        tee_stderr = TeeStream(sys.__stderr__, log_file)

        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
            print(f'>>> Logging stdout/stderr to {log_path}')
            try:
                # Parse command line arguments after logger setup so all prints are captured.
                options, trainer_class, dataset_name, sub_data = read_options()

                train_path = os.path.join('./data', dataset_name, 'data', 'train')
                test_path = os.path.join('./data', dataset_name, 'data', 'test')

                # `dataset` is a tuple like (cids, groups, train_data, test_data)
                all_data_info = read_data(train_path, test_path, sub_data)

                if len(all_data_info[0]) == 0:
                    raise ValueError(
                        f'No client data loaded from train_path={train_path}. '
                        f'Please check --dataset="{options["dataset"]}" and data files.'
                    )

                # Call appropriate trainer
                if options.get('sequential_cl', False):
                    _run_sequential_tasks(options, trainer_class, all_data_info)
                else:
                    trainer = trainer_class(options, all_data_info)
                    trainer.train()
                    _post_train_eval_and_save(trainer)
            except Exception:
                print('>>> Fatal error captured, traceback follows:')
                traceback.print_exc()
                raise

if __name__ == '__main__':
    main()
