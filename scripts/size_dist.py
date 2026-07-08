"""lmdb 内の分子サイズ（原子数）分布を調べる。"""
from __future__ import annotations
import argparse, pickle
import lmdb, numpy as np


def sizes(path, max_scan=0):
    env = lmdb.open(path, subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False)
    with env.begin() as txn:
        meta = txn.get(b'__len__')
        n = int(meta.decode('ascii')) if meta else txn.stat()['entries']
    out = []
    with env.begin() as txn:
        lim = n if max_scan <= 0 else min(n, max_scan)
        for i in range(lim):
            v = txn.get(f'{i:09d}'.encode('ascii'))
            if v is None:
                continue
            d = pickle.loads(v)
            out.append(d['atom_type_idx'].shape[0])
    env.close()
    return np.array(out), n


def report(name, arr, total):
    print(f'\n== {name} ==  (走査 {len(arr)} / 全 {total})')
    if len(arr) == 0:
        return
    qs = [0, 5, 25, 50, 75, 90, 95, 99, 100]
    ps = np.percentile(arr, qs)
    print('  原子数 percentile:')
    for q, p in zip(qs, ps):
        print(f'    p{q:>3}: {p:6.0f}')
    print(f'  mean={arr.mean():.1f} min={arr.min()} max={arr.max()}')
    bins = [0, 20, 40, 60, 80, 100, 150, 200, 240, 10000]
    hist, _ = np.histogram(arr, bins=bins)
    print('  ヒストグラム:')
    for i in range(len(hist)):
        lo, hi = bins[i], bins[i + 1]
        frac = hist[i] / len(arr) * 100
        bar = '#' * int(frac / 2)
        print(f'    [{lo:>4},{hi:>5}): {hist[i]:>6} ({frac:5.1f}%) {bar}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--paths', nargs='+', required=True)
    p.add_argument('--max_scan', type=int, default=0,
                   help='各lmdbで走査する最大件数（0=全件）')
    args = p.parse_args()
    for path in args.paths:
        arr, total = sizes(path, args.max_scan)
        report(path, arr, total)
