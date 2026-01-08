import numpy as np
import argparse
import importlib
import torch
import os

from src.utils.worker_utils import read_data
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


def main():
    # Parse command line arguments
    options, trainer_class, dataset_name, sub_data = read_options()

    train_path = os.path.join('./data', dataset_name, 'data', 'train')
    test_path = os.path.join('./data', dataset_name, 'data', 'test')

    # `dataset` is a tuple like (cids, groups, train_data, test_data)
    all_data_info = read_data(train_path, test_path, sub_data)

    # Call appropriate trainer
    trainer = trainer_class(options, all_data_info)
    trainer.train()

    # After training, evaluate the final global model on each client's local test set
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
            # Ensure worker has the final global model
            c.set_flat_model_params(trainer.latest_model)
            tot_correct, num_sample, loss = c.local_test(use_eval_data=True)
            acc = float(tot_correct) / float(num_sample) if num_sample > 0 else 0.0
            client_acc[int(c.cid)] = acc

            # Training label distribution (from client's MiniDataset)
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
                # entropy
                ent = -np.sum([p * np.log(p + 1e-12) for p in probs])
                label_entropy[int(c.cid)] = float(ent)
            else:
                classes_per_client[int(c.cid)] = 0
                top1_frac[int(c.cid)] = 0.0
                label_entropy[int(c.cid)] = 0.0

        # Save client accuracies and stats
        acc_json_path = os.path.join(result_dir, 'client_acc.json')
        with open(acc_json_path, 'w') as outf:
            json.dump({'client_acc': client_acc,
                       'classes_per_client': classes_per_client,
                       'top1_frac': top1_frac,
                       'label_entropy': label_entropy}, outf)
        print(f'>>> Saved per-client accuracy and stats to {acc_json_path}')

        # Also save numpy array of accuracies (ordered by client id)
        ids_sorted = sorted(client_acc.keys())
        accs = np.array([client_acc[i] for i in ids_sorted])
        np.save(os.path.join(result_dir, 'client_acc.npy'), accs)

        # Compute and save statistics for client accuracies
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
            import json as _json
            _json.dump(stats, sf)
        print('>>> Client accuracy stats:')
        print(stats)

        # Plot: bar plot by client id
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

        # Plot: sorted accuracy curve
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

        # Also save a small JSON bundle with acc and stats for easy external comparison
        bundle_path = os.path.join(result_dir, 'client_acc_bundle.json')
        try:
            with open(bundle_path, 'w') as bf:
                import json as _json
                _json.dump({'client_ids': ids_sorted, 'client_acc': accs.tolist(), 'stats': stats}, bf)
            print(f'>>> Saved client_acc bundle to {bundle_path}')
        except Exception:
            pass

        # If trainer maintains personalized models (Ditto, pFedMe), evaluate and save them too
        try:
            if hasattr(trainer, 'personal_models') and isinstance(trainer.personal_models, dict):
                print('>>> Evaluating personalized models on each client test set...')
                personal_acc = {}
                for c in trainer.clients:
                    # load personalized model if exists, otherwise use latest global
                    p = trainer.personal_models.get(c.cid, trainer.latest_model)
                    try:
                        c.set_flat_model_params(p)
                        tot_correct, num_sample, loss = c.local_test(use_eval_data=True)
                        acc = float(tot_correct) / float(num_sample) if num_sample > 0 else 0.0
                        personal_acc[int(c.cid)] = acc
                    except Exception:
                        personal_acc[int(c.cid)] = 0.0

                # Save personal accuracies as json and numpy
                personal_json_path = os.path.join(result_dir, 'client_acc_personal.json')
                with open(personal_json_path, 'w') as outf:
                    json.dump({'client_acc_personal': personal_acc}, outf)
                print(f'>>> Saved per-client personalized accuracy to {personal_json_path}')

                ids_sorted_p = sorted(personal_acc.keys())
                accs_p = np.array([personal_acc[i] for i in ids_sorted_p])
                np.save(os.path.join(result_dir, 'client_acc_personal.npy'), accs_p)

                # Compute and save statistics for personalized accuracies
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
                    import json as _json
                    _json.dump(pstats, sf)
                print('>>> Personalized client accuracy stats:')
                print(pstats)

                # Bar plot for personalized accuracies
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
                    print(f'>>> Saved personalized client accuracy bar plot to {barpath_p}')
                except Exception:
                    pass

                # Sorted plot for personalized accuracies
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
                    print(f'>>> Saved sorted personalized accuracy plot to {sortpath_p}')
                except Exception:
                    pass

                # Append personalized info into the bundle if possible
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

if __name__ == '__main__':
    main()
