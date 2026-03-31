"""
ASE lmdb の data フィールドと全キー構造の詳細調査
"""
import lmdb
import zlib
import json
import sys

def inspect_lmdb_full(path: str, n_samples: int = 5):
    env = lmdb.open(path, subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        keys = [k for k, _ in cursor.iternext()]
        print(f'総エントリ数: {len(keys)}')

        for i, key in enumerate(keys[:n_samples]):
            val = txn.get(key)
            raw = zlib.decompress(val) if val[:2] in (b'x\x9c', b'x\x01', b'x\xda') else val
            obj = json.loads(raw)

            print(f'\n=== key={key} ===')
            # 原子数
            nums = obj['numbers']['__ndarray__'][2]
            n_atoms = obj['numbers']['__ndarray__'][0][0]
            print(f'  n_atoms: {n_atoms}')
            print(f'  atomic_numbers[:10]: {nums[:10]}')

            # data フィールドの中身を全部表示
            print(f'  data keys: {list(obj["data"].keys())}')
            for k, v in obj['data'].items():
                print(f'    data[{k}] = {str(v)[:200]}')

            # charge があるか確認
            print(f'  top-level keys: {list(obj.keys())}')

if __name__ == '__main__':
    path = r'D:\Dataset\OMol_base\OPoly26\val\data0000.aselmdb'
    inspect_lmdb_full(path)
