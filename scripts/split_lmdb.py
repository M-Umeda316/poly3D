"""処理済み lmdb を sid の接頭辞（PolyOmics なら uuid = ポリマー）単位で train/val 分割。

同一ポリマー（＝同一 SMILES の配座群）が train と val の両方に入る漏洩を防ぐため、
レコード index ではなく **sid の先頭トークン（':' 区切りの uuid）** 単位で振り分ける。
既定は uuid ハッシュで 1/val_every をホールドアウト（決定的）。

例:
  python scripts/split_lmdb.py --src data/polyomics_PG.lmdb \
      --train_out data/polyomics_PG_train.lmdb --val_out data/polyomics_PG_val.lmdb \
      --val_every 20
"""
import argparse
import pickle
import zlib

import lmdb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--train_out', required=True)
    ap.add_argument('--val_out', required=True)
    ap.add_argument('--val_every', type=int, default=20,
                    help='uuid の 1/val_every を val へ（決定的ハッシュ）')
    ap.add_argument('--train_map_gb', type=int, default=8)
    ap.add_argument('--val_map_gb', type=int, default=2)
    args = ap.parse_args()

    src = lmdb.open(args.src, subdir=False, readonly=True, lock=False, readahead=False)
    with src.begin() as txn:
        meta = txn.get(b'__len__')
        n = int(meta.decode()) if meta else txn.stat()['entries']

    tr = lmdb.open(args.train_out, map_size=args.train_map_gb * 1024**3, subdir=False, map_async=True)
    va = lmdb.open(args.val_out, map_size=args.val_map_gb * 1024**3, subdir=False, map_async=True)
    ttxn, vtxn = tr.begin(write=True), va.begin(write=True)
    nt = nv = 0
    uuids_val = set()
    uuids_tr = set()

    with src.begin() as stxn:
        for i in range(n):
            val = stxn.get(f'{i:09d}'.encode('ascii'))
            if val is None:
                continue
            d = pickle.loads(val)
            uuid = str(d.get('sid', '')).split(':')[0]
            # 決定的: uuid の CRC で振り分け（同一ポリマーは必ず同じ側）
            is_val = (zlib.crc32(uuid.encode()) % args.val_every == 0)
            if is_val:
                vtxn.put(f'{nv:09d}'.encode('ascii'), val)
                nv += 1
                uuids_val.add(uuid)
            else:
                ttxn.put(f'{nt:09d}'.encode('ascii'), val)
                nt += 1
                uuids_tr.add(uuid)
            if (nt + nv) % 20000 == 0:
                ttxn.commit(); ttxn = tr.begin(write=True)
                vtxn.commit(); vtxn = va.begin(write=True)

    ttxn.commit(); vtxn.commit()
    with tr.begin(write=True) as t:
        t.put(b'__len__', str(nt).encode())
    with va.begin(write=True) as t:
        t.put(b'__len__', str(nv).encode())
    tr.sync(); va.sync(); tr.close(); va.close(); src.close()
    leak = uuids_val & uuids_tr
    print(f'train={nt:,} rec / {len(uuids_tr):,} polymers → {args.train_out}')
    print(f'val  ={nv:,} rec / {len(uuids_val):,} polymers → {args.val_out}')
    print(f'polymer leakage (should be 0): {len(leak)}')


if __name__ == '__main__':
    main()
