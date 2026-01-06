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


def load_acc(run_folder):
    path = os.path.join(run_folder, 'client_acc.npy')
    if not os.path.exists(path):
        # try json bundle
        jpath = os.path.join(run_folder, 'client_acc_bundle.json')
        if os.path.exists(jpath):
            with open(jpath, 'r') as f:
                b = json.load(f)
            return np.array(b.get('client_acc', []))
        return None
    return np.load(path)


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

    # Print and save stats
    stats = {'non_iid': stats_from_acc(non_acc), 'iid': stats_from_acc(iid_acc)}
    print('Statistics:')
    print(stats)
    stats_path = os.path.join('result', f'compare_emnist_stats_{time.strftime("%Y-%m-%dT%H-%M-%S")}.json')
    with open(stats_path, 'w') as sf:
        json.dump(stats, sf, indent=2)
    print('Saved comparison stats to', stats_path)


if __name__ == '__main__':
    main()
