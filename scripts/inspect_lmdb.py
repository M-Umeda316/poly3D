"""
ASE lmdb ファイルの内部構造を調査するスクリプト
"""
import lmdb
import pickle
import sys
from pathlib import Path

def inspect_lmdb(path: str, n_samples: int = 3):
    env = lmdb.open(
        path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    with env.begin() as txn:
        # 全キーを確認
        cursor = txn.cursor()
        keys = []
        for key, _ in cursor.iternext_dup() if False else cursor.iternext():
            keys.append(key)

        print(f'総エントリ数: {len(keys)}')
        print(f'最初のキー10件: {keys[:10]}')
        print()

        # 最初のN件を詳しく見る
        for i, key in enumerate(keys[:n_samples]):
            val = txn.get(key)
            print(f'--- key={key} (len={len(val)}) ---')
            try:
                # zlib圧縮されている可能性を確認
                import zlib
                if val[:2] == b'x\x9c' or val[:2] == b'x\x01' or val[:2] == b'x\xda':
                    val = zlib.decompress(val)
                    print('  (zlib decompressed)')
                # JSON or pickle?
                import json
                import numpy as np
                try:
                    obj = json.loads(val)
                    print(f'  [JSON] type: {type(obj)}')
                    if isinstance(obj, dict):
                        print(f'  keys: {list(obj.keys())}')
                        for k, v in list(obj.items())[:15]:
                            print(f'    {k}: {type(v).__name__} = {str(v)[:150]}')
                except json.JSONDecodeError:
                    obj = pickle.loads(val)
                    print(f'  [pickle] type: {type(obj)}')
            except Exception as e:
                print(f'  pickle.loads failed: {e}')
                # 生データの先頭を見る
                print(f'  raw[:20]: {val[:20]}')
            print()

if __name__ == '__main__':
    # 最初のファイルを検査
    lmdb_path = r'D:\Dataset\OMol_base\OPoly26\val\data0000.aselmdb'
    if len(sys.argv) > 1:
        lmdb_path = sys.argv[1]

    print(f'Inspecting: {lmdb_path}\n')
    inspect_lmdb(lmdb_path)
