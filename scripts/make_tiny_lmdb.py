"""
過学習テスト用に、既存の処理済み lmdb から先頭 N 分子だけを
コピーした極小 lmdb を作る。

  python scripts/make_tiny_lmdb.py --src data/val.lmdb --dst runs/overfit/tiny.lmdb --n 128

キー形式（{idx:09d} ascii → pickle）と __len__ をそのまま踏襲するので、
ConformerDataset がそのまま読める。max_atoms でフィルタされる分子は
スキップし、実際に格納した件数を __len__ に書く。
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import lmdb


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True)
    p.add_argument('--dst', required=True)
    p.add_argument('--n', type=int, default=128)
    p.add_argument('--max_atoms', type=int, default=240,
                   help='この原子数を超える分子はスキップ（学習側と一致させる）')
    p.add_argument('--min_atoms', type=int, default=0,
                   help='この原子数未満の分子はスキップ（大分子だけ抽出する用）')
    p.add_argument('--map_size_gb', type=float, default=2.0)
    args = p.parse_args()

    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    src_env = lmdb.open(args.src, subdir=False, readonly=True,
                        lock=False, readahead=False, meminit=False)
    with src_env.begin() as txn:
        meta = txn.get(b'__len__')
        src_len = int(meta.decode('ascii')) if meta else txn.stat()['entries']
    print(f'src   : {args.src}  ({src_len:,} 件)')

    dst_env = lmdb.open(str(dst), subdir=False,
                        map_size=int(args.map_size_gb * 1024 ** 3))

    kept = 0
    scanned = 0
    with src_env.begin() as rtxn, dst_env.begin(write=True) as wtxn:
        while kept < args.n and scanned < src_len:
            key = f'{scanned:09d}'.encode('ascii')
            val = rtxn.get(key)
            scanned += 1
            if val is None:
                continue
            d = pickle.loads(val)
            n_atoms = d['atom_type_idx'].shape[0]
            if n_atoms > args.max_atoms or n_atoms < args.min_atoms:
                continue
            new_key = f'{kept:09d}'.encode('ascii')
            wtxn.put(new_key, val)  # pickle バイト列をそのままコピー
            kept += 1
        wtxn.put(b'__len__', str(kept).encode('ascii'))

    src_env.close()
    dst_env.close()
    print(f'dst   : {dst}  ({kept} 件を格納, {scanned} 件走査)')


if __name__ == '__main__':
    main()
