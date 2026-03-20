import numpy as np
import argparse
import importlib
import torch
import os
import time

from src.utils.worker_utils import read_data, MiniDataset
from config import OPTIMIZERS, DATASETS, MODEL_PARAMS, TRAINERS


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
                        default=200)
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
                        default=64)
    parser.add_argument('--num_epoch',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=5)
    parser.add_argument('--lr',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.1)
    parser.add_argument('--mu',
                        help='FedProx proximal term coefficient (mu);',
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
    parsed = parser.parse_args()
    options = parsed.__dict__
    options['gpu'] = options['gpu'] and torch.cuda.is_available()

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
            except Exception:
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
        except Exception:
            pass

        try:
            if hasattr(trainer, 'personal_models') and isinstance(trainer.personal_models, dict):
                print('>>> Evaluating personalized models on each client test set...')
                personal_acc = {}
                for c in trainer.clients:
                    p = trainer.personal_models.get(c.cid, trainer.latest_model)
                    try:
                        c.set_flat_model_params(p)
                        tot_correct, num_sample, loss = c.local_test(use_eval_data=True)
                        acc = float(tot_correct) / float(num_sample) if num_sample > 0 else 0.0
                        personal_acc[int(c.cid)] = acc
                    except Exception:
                        personal_acc[int(c.cid)] = 0.0

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
                except Exception:
                    pass

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
                except Exception:
                    pass

                try:
                    with open(bundle_path, 'r') as bf:
                        bund = json.load(bf)
                except Exception:
                    bund = {'client_ids': ids_sorted, 'client_acc': accs.tolist(), 'stats': stats}
                bund.update({'client_acc_personal': accs_p.tolist(), 'personal_stats': pstats})
                try:
                    with open(bundle_path, 'w') as bf:
                        json.dump(bund, bf)
                except Exception:
                    pass
        except Exception as e:
            print('Error during personalized model evaluation:', e)

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
        raise ValueError('Cannot build CL tasks: no training labels found.')

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
    if active_labels is not None:
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


def _run_sequential_tasks(options, trainer_class, all_data_info):
    """Run sequential CL training across tasks, carrying global model forward."""
    num_tasks = max(1, int(options.get('num_tasks', 3)))
    task_datasets, task_label_lists = _split_dataset_into_tasks(all_data_info, num_tasks=num_tasks)

    print(f'>>> Sequential CL enabled: {len(task_datasets)} tasks')
    for i, ls in enumerate(task_label_lists, start=1):
        print(f'    Task-{i} labels: {ls}')

    prev_global_model = None
    task_summaries = []
    eval_acc_matrix = np.full((len(task_datasets), len(task_datasets)), np.nan, dtype=np.float64)
    eval_loss_matrix = np.full((len(task_datasets), len(task_datasets)), np.nan, dtype=np.float64)

    for task_idx, task_dataset in enumerate(task_datasets, start=1):
        task_options = dict(options)
        task_options['dataset'] = f"{options['dataset']}_task{task_idx}"
        task_options['dis'] = (task_options.get('dis', '') + f"_cl_task{task_idx}").strip('_')
        task_options['active_labels'] = [int(v) for v in task_label_lists[task_idx - 1]]
        task_options['label_map'] = {
            int(g): int(i) for i, g in enumerate(task_label_lists[task_idx - 1])
        }
        task_options['num_active_classes'] = len(task_label_lists[task_idx - 1])

        print('\n' + '=' * 90)
        print(f'>>> Start sequential Task-{task_idx}/{len(task_datasets)}')
        print('=' * 90)

        trainer = trainer_class(task_options, task_dataset)

        if prev_global_model is not None:
            trainer.worker.set_flat_model_params(prev_global_model)
            trainer.latest_model = prev_global_model.detach().clone()

        trainer.train()
        _post_train_eval_and_save(trainer)

        # CL seen-task evaluation: after task t, evaluate on tasks 1..t
        stage_eval = {}
        for eval_task_idx in range(task_idx):
            eval_ret = _evaluate_global_on_dataset(trainer,
                                                   task_datasets[eval_task_idx],
                                                   trainer.latest_model,
                                                   active_labels=task_label_lists[eval_task_idx])
            eval_acc_matrix[task_idx - 1, eval_task_idx] = eval_ret['acc']
            eval_loss_matrix[task_idx - 1, eval_task_idx] = eval_ret['loss']
            stage_eval[f'task_{eval_task_idx + 1}'] = {
                'acc': float(eval_ret['acc']),
                'loss': float(eval_ret['loss']),
                'num_samples': int(eval_ret['num_samples'])
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

        task_result_dir = os.path.join(trainer.metrics.result_path, trainer.metrics.exp_name)
        task_summaries.append({
            'task_idx': task_idx,
            'labels': task_label_lists[task_idx - 1],
            'num_clients': len(task_dataset[0]),
            'result_dir': task_result_dir,
            'seen_task_eval': stage_eval
        })

    # Forgetting statistics from seen-task accuracy matrix
    final_stage = len(task_datasets) - 1
    forgetting = {}
    forgetting_vals = []
    for j in range(len(task_datasets)):
        diag = eval_acc_matrix[j, j] if j <= final_stage else np.nan
        final_acc = eval_acc_matrix[final_stage, j]
        if np.isnan(diag) or np.isnan(final_acc):
            forgetting[f'task_{j + 1}'] = None
            continue
        fj = float(diag - final_acc)
        forgetting[f'task_{j + 1}'] = fj
        if j < final_stage:
            forgetting_vals.append(fj)

    mean_forgetting = float(np.mean(forgetting_vals)) if len(forgetting_vals) > 0 else 0.0

    summary_path = os.path.join('result', f"{options['dataset']}_cl{len(task_datasets)}_sequential_summary_{time.strftime('%Y-%m-%dT%H-%M-%S')}.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as sf:
        import json
        json.dump({'options': options,
                   'tasks': task_summaries,
                   'labels_per_task': task_label_lists,
                   'eval_acc_matrix': eval_acc_matrix.tolist(),
                   'eval_loss_matrix': eval_loss_matrix.tolist(),
                   'forgetting': forgetting,
                   'mean_forgetting': mean_forgetting}, sf, indent=2)
    print(f'>>> Saved sequential CL summary to {summary_path}')

    # Save matrix as npy for easier downstream analysis
    np.save(os.path.join('result', f"{options['dataset']}_cl_acc_matrix_{time.strftime('%Y-%m-%dT%H-%M-%S')}.npy"),
            eval_acc_matrix)

    # Optional heatmap
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 5))
        vis = np.where(np.isnan(eval_acc_matrix), 0.0, eval_acc_matrix)
        im = plt.imshow(vis, cmap='viridis', vmin=0.0, vmax=1.0)
        for i in range(vis.shape[0]):
            for j in range(vis.shape[1]):
                txt = '-' if np.isnan(eval_acc_matrix[i, j]) else f"{eval_acc_matrix[i, j]:.3f}"
                plt.text(j, i, txt, ha='center', va='center', color='white', fontsize=9)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel('Evaluated Task')
        plt.ylabel('After Training Task')
        plt.title('Sequential CL Seen-Task Accuracy Matrix')
        ticks = np.arange(len(task_datasets))
        labels = [f'T{i+1}' for i in ticks]
        plt.xticks(ticks, labels)
        plt.yticks(ticks, labels)
        plt.tight_layout()
        heat_path = os.path.join('result', f"{options['dataset']}_cl_acc_matrix_{time.strftime('%Y-%m-%dT%H-%M-%S')}.png")
        plt.savefig(heat_path, dpi=200)
        plt.close()
        print(f'>>> Saved CL accuracy matrix heatmap to {heat_path}')
    except Exception as e:
        print(f'Warning: failed to save CL matrix heatmap: {e}')


def main():
    # Parse command line arguments
    options, trainer_class, dataset_name, sub_data = read_options()

    train_path = os.path.join('./data', dataset_name, 'data', 'train')
    test_path = os.path.join('./data', dataset_name, 'data', 'test')

    # `dataset` is a tuple like (cids, groups, train_data, test_data)
    all_data_info = read_data(train_path, test_path, sub_data)

    # Call appropriate trainer
    if options.get('sequential_cl', False):
        _run_sequential_tasks(options, trainer_class, all_data_info)
    else:
        trainer = trainer_class(options, all_data_info)
        trainer.train()
        _post_train_eval_and_save(trainer)

if __name__ == '__main__':
    main()
