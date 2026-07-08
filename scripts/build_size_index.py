"""
処理済み lmdb を 1 回スキャンし、各エントリの原子数を numpy int32 配列
（長さ = __len__）として `<src>.sizes.npy` に保存する。

大分子オーバーサンプリング（train.py --oversample_alpha）で、全 idx の
原子数を高速に参照するための事前計算インデックス。

  python scripts/build_size_index.py --src runs/overfit/tiny_large.lmdb

欠損エントリ（key が無い / None）は 0 を格納する。size_dist.py と同じ
lmdb 読み方（subdir=False, readonly, lock=False 等）を踏襲する。
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import lmdb
import numpy as np
from tqdm import tqdm

# cp932 コンソールでの UnicodeEncodeError 対策
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def build_sizes(src: str, max_scan: int = 0) -> np.ndarray:
    env = lmdb.open(src, subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False)
    with env.begin() as txn:
        meta = txn.get(b'__len__')
        n = int(meta.decode('ascii')) if meta else txn.stat()['entries']

    lim = n if max_scan <= 0 else min(n, max_scan)
    sizes = np.zeros(n, dtype=np.int32)
    n_missing = 0
    with env.begin() as txn:
        for i in tqdm(range(lim), desc='scan', dynamic_ncols=True):
            v = txn.get(f'{i:09d}'.encode('ascii'))
            if v is None:
                n_missing += 1
                continue
            d = pickle.loads(v)
            sizes[i] = int(d['atom_type_idx'].shape[0])
    env.close()

    if n_missing:
        print(f'  欠損エントリ: {n_missing} 件（原子数 0 として格納）')
    return sizes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='処理済み lmdb のパス')
    p.add_argument('--out', default=None,
                   help='出力 .npy パス（省略時 <src>.sizes.npy）')
    p.add_argument('--force', action='store_true',
                   help='既存の出力を上書き。無指定なら存在時はスキップして終了')
    p.add_argument('--max_scan', type=int, default=0,
                   help='走査する最大件数（0=全件、デバッグ用）')
    args = p.parse_args()

    out = Path(args.out) if args.out else Path(str(args.src) + '.sizes.npy')

    if out.exists() and not args.force:
        print(f'既に存在します（--force で上書き）: {out}')
        return

    print(f'src : {args.src}')
    sizes = build_sizes(args.src, args.max_scan)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, sizes)

    nz = sizes[sizes > 0]
    print(f'out : {out}  (長さ {len(sizes):,})')
    if len(nz) > 0:
        print(f'  原子数: min={nz.min()} max={nz.max()} '
              f'mean={nz.mean():.1f}  (>0 の件数 {len(nz):,})')


if __name__ == '__main__':
    main()
