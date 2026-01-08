import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
import json
import time


def latest_run_folder(dataset_name):
    base = os.path.join('result', dataset_name)
    if not os.path.exists(base):
        return None
    entries = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    if not entries:
        return None
    entries.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return entries[0]


def load_acc(run_folder, kind='global'):
    """Load client accuracy array from a run folder.

    kind: 'global' (client_acc.npy) or 'personal' (client_acc_personal.npy)
    Falls back to 'client_acc_bundle.json' if numpy not present.
    """
    if kind == 'personal':
        filename = 'client_acc_personal.npy'
        bundle_key = 'client_acc_personal'
    else:
        filename = 'client_acc.npy'
        bundle_key = 'client_acc'

    path = os.path.join(run_folder, filename)
    if os.path.exists(path):
        try:
            return np.load(path)
        except Exception:
            pass

    # try json bundle
    jpath = os.path.join(run_folder, 'client_acc_bundle.json')
    if os.path.exists(jpath):
        try:
            with open(jpath, 'r') as f:
                b = json.load(f)
            return np.array(b.get(bundle_key, []))
        except Exception:
            return None
    # try standalone json files
    jpath2 = os.path.join(run_folder, f'client_acc{"_personal" if kind=="personal" else ""}.json')
    if os.path.exists(jpath2):
        try:
            with open(jpath2, 'r') as f:
                b = json.load(f)
            return np.array(list(b.values())[0])
        except Exception:
            return None
    return None


def stats_from_acc(accs):
    if accs is None or len(accs) == 0:
        return {}
    return {
        'mean': float(np.mean(accs)),
        'std': float(np.std(accs)),
        'min': float(np.min(accs)),
        'max': float(np.max(accs)),
        '10th': float(np.percentile(accs, 10)),
        '90th': float(np.percentile(accs, 90))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noniid_run', help='Path to non-iid run folder or dataset name', default='emnist_balanced_0_shard2_niid')
    parser.add_argument('--iid_run', help='Path to iid run folder or dataset name', default='emnist_balanced_0_equal_iid')
    args = parser.parse_args()

    noniid = args.noniid_run
    iid = args.iid_run

    if os.path.exists(noniid):
        noniid_folder = noniid
    else:
        noniid_folder = latest_run_folder(noniid) or noniid

    if os.path.exists(iid):
        iid_folder = iid
    else:
        iid_folder = latest_run_folder(iid) or iid

    print('Using Non-IID run folder:', noniid_folder)
    print('Using IID run folder:', iid_folder)

    non_acc = load_acc(noniid_folder)
    iid_acc = load_acc(iid_folder)
    # try to load personalized accuracies as well
    non_acc_personal = load_acc(noniid_folder, kind='personal')
    iid_acc_personal = load_acc(iid_folder, kind='personal')

    if non_acc is None:
        print('Could not find client_acc in non-iid folder:', noniid_folder)
        return
    if iid_acc is None:
        print('Could not find client_acc in iid folder:', iid_folder)
        return

    non_sorted = np.sort(non_acc)
    iid_sorted = np.sort(iid_acc)

    # Save IID sorted curve under iid run folder if not present
    try:
        plt.figure(figsize=(8, 4))
        plt.plot(iid_sorted, marker='o')
        plt.xlabel('Client (sorted)')
        plt.ylabel('Accuracy')
        plt.title('IID: Per-client Test Accuracy (sorted)')
        plt.tight_layout()
        save_iid = os.path.join(iid_folder, 'client_acc_sorted_iid.png')
        plt.savefig(save_iid, dpi=200)
        plt.close()
        print('Saved IID sorted curve to', save_iid)
    except Exception as e:
        print('Error saving IID sorted plot:', e)

    # Combined plot
    try:
        plt.figure(figsize=(8, 4))
        plt.plot(non_sorted, marker='o', label='Non-IID')
        plt.plot(iid_sorted, marker='x', label='IID')
        plt.xlabel('Client (sorted)')
        plt.ylabel('Accuracy')
        plt.title('Per-client Test Accuracy: Non-IID vs IID (sorted)')
        plt.legend()
        plt.tight_layout()
        tstamp = time.strftime('%Y-%m-%dT%H-%M-%S')
        outpath = os.path.join('result', f'compare_emnist_iid_vs_noniid_{tstamp}.png')
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath, dpi=200)
        plt.close()
        print('Saved comparison plot to', outpath)
    except Exception as e:
        print('Error saving comparison plot:', e)

    # If personalized accuracies are present for both runs, produce personal comparison
    try:
        if non_acc_personal is not None and iid_acc_personal is not None:
            non_sorted_p = np.sort(non_acc_personal)
            iid_sorted_p = np.sort(iid_acc_personal)
            plt.figure(figsize=(8, 4))
            plt.plot(non_sorted_p, marker='o', label='Non-IID (personal)')
            plt.plot(iid_sorted_p, marker='x', label='IID (personal)')
            plt.xlabel('Client (sorted)')
            plt.ylabel('Personalized Accuracy')
            plt.title('Per-client Personalized Accuracy: Non-IID vs IID (sorted)')
            plt.legend()
            plt.tight_layout()
            tstamp = time.strftime('%Y-%m-%dT%H-%M-%S')
            outpath_p = os.path.join('result', f'compare_emnist_personal_iid_vs_noniid_{tstamp}.png')
            os.makedirs(os.path.dirname(outpath_p), exist_ok=True)
            plt.savefig(outpath_p, dpi=200)
            plt.close()
            print('Saved personalized comparison plot to', outpath_p)
    except Exception as e:
        print('Error saving personalized comparison plot:', e)

    # Print and save stats
    stats = {'non_iid': stats_from_acc(non_acc), 'iid': stats_from_acc(iid_acc)}
    # include personal stats if available
    if non_acc_personal is not None and iid_acc_personal is not None:
        stats['non_iid_personal'] = stats_from_acc(non_acc_personal)
        stats['iid_personal'] = stats_from_acc(iid_acc_personal)
    print('Statistics:')
    print(stats)
    stats_path = os.path.join('result', f'compare_emnist_stats_{time.strftime("%Y-%m-%dT%H-%M-%S")}.json')
    with open(stats_path, 'w') as sf:
        json.dump(stats, sf, indent=2)
    print('Saved comparison stats to', stats_path)


if __name__ == '__main__':
    main()
