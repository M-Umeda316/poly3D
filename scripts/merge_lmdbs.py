"""複数の処理済み lmdb を連番キーで1つに結合する。

用途: PG_train(リーク無し既存split) + core3(PEST/PSUL/PPHS, PG val と別クラスなので
リーク無し) を結合して大単位データ増 probe 用の学習 lmdb を作る。

各 lmdb はキー `f'{idx:09d}'`（値=pickle 済みレコード）+ `__len__`（件数）規約。
値は unpickle せず生バイトのまま連番で書き写す（高速・スキーマ非依存）。
"""
from __future__ import annotations

import argparse
import sys

import lmdb

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass


def _count(env) -> int:
    with env.begin() as txn:
        n = txn.get(b'__len__')
        if n is not None:
            return int(n.decode('ascii'))
        return txn.stat()['entries']


def merge(sources: list, out_path: str, map_size_gb: int) -> None:
    dst = lmdb.open(out_path, subdir=False, map_size=map_size_gb * (1024 ** 3),
                    lock=True, readahead=False, meminit=False)
    written = 0
    for src_path in sources:
        senv = lmdb.open(src_path, subdir=False, readonly=True, lock=False,
                         readahead=False, meminit=False, max_readers=256)
        n = _count(senv)
        copied = skipped = 0
        with senv.begin() as stxn, dst.begin(write=True) as dtxn:
            for i in range(n):
                v = stxn.get(f'{i:09d}'.encode('ascii'))
                if v is None:
                    skipped += 1
                    continue
                dtxn.put(f'{written:09d}'.encode('ascii'), v)
                written += 1
                copied += 1
        senv.close()
        print(f'{src_path}: {copied} 件コピー (skip {skipped}) -> 累計 {written}',
              flush=True)
    with dst.begin(write=True) as dtxn:
        dtxn.put(b'__len__', str(written).encode('ascii'))
    dst.sync()
    dst.close()
    print(f'DONE: {out_path} 合計 {written} 件', flush=True)


def parse_args():
    p = argparse.ArgumentParser(description='処理済み lmdb を連番キーで結合')
    p.add_argument('--sources', nargs='+', required=True, help='結合元 lmdb（順に連結）')
    p.add_argument('--out_path', required=True)
    p.add_argument('--map_size_gb', type=int, default=20)
    return p.parse_args()


if __name__ == '__main__':
    a = parse_args()
    merge(a.sources, a.out_path, a.map_size_gb)
