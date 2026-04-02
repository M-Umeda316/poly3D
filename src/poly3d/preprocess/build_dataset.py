"""
OPoly26 前処理パイプライン

ASE lmdb → 処理済み lmdb（結合情報 + 特徴量 + RWPE 付き）

処理:
  1. XYZ 形式に変換 → RDKit DetermineBonds で結合推定
  2. 複数フラグメント → 最大フラグメントを採用
  3. model/features.mol_to_data() で特徴量（RWPE 含む）を計算
  4. lmdb に pickle 保存

実行例:
  python -m preprocess.build_dataset \\
      --src_dir D:/Dataset/OMol_base/OPoly26/val \\
      --out_path D:/Dataset/OMol_base/OPoly26/processed/val.lmdb \\
      --n_workers 8
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
from pathlib import Path
from typing import Optional, Tuple

import lmdb
import numpy as np
from tqdm import tqdm

# read_molecule() の戻り値型エイリアス: (atomic_nums, positions, charge, sid)
MolRecord = Tuple[np.ndarray, np.ndarray, int, str]


def _process_record(
    args: MolRecord,
) -> Optional[bytes]:
    """
    1 分子を処理して pickle bytes を返す。失敗時は None。
    spawn worker 内で実行されるためモジュールトップレベルに配置。
    """
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    from poly3d.model.features import mol_to_data

    atomic_nums, positions, charge, sid = args

    # 元素記号の取得: RDKit 周期表を使用（未知元素は None → スキップ）
    ptable = Chem.GetPeriodicTable()
    symbols = []
    for z in atomic_nums:
        try:
            sym = ptable.GetElementSymbol(int(z))
            symbols.append(sym)
        except Exception:
            return None  # 不明な原子番号はスキップ

    # XYZ 文字列を生成
    n = len(atomic_nums)
    xyz_lines = [str(n), '']
    for sym, (x, y, z) in zip(symbols, positions):
        xyz_lines.append(f'{sym}  {x:.6f}  {y:.6f}  {z:.6f}')
    xyz_block = '\n'.join(xyz_lines)

    try:
        mol = Chem.MolFromXYZBlock(xyz_block)
        if mol is None:
            return None
        rdDetermineBonds.DetermineBonds(mol, charge=charge, maxIterations=2000)
    except Exception:
        return None

    # 最大フラグメントを採用
    frags = Chem.GetMolFrags(mol, asMols=True)
    if not frags:
        return None
    mol = max(frags, key=lambda m: m.GetNumAtoms())

    if mol.GetNumConformers() == 0:
        return None

    try:
        data_dict = mol_to_data(mol, sid=sid, use_rwpe=True, use_lappe=False)
    except Exception:
        return None

    return pickle.dumps(data_dict)


# ── DatasetBuilder ─────────────────────────────────────────────────────────────

class DatasetBuilder:
    def __init__(
        self,
        src_dir: str | Path,
        out_path: str | Path,
        n_workers: int = 4,
        chunk_size: int = 64,
        map_size_gb: int = 50,
    ):
        self.src_dir = Path(src_dir)
        self.out_path = Path(out_path)
        self.n_workers = n_workers
        self.chunk_size = chunk_size
        self.map_size_gb = map_size_gb

    def run(self) -> None:
        from poly3d.preprocess.lmdb_reader import list_lmdb_files, iter_lmdb, read_molecule

        lmdb_files = list_lmdb_files(self.src_dir)
        if not lmdb_files:
            raise FileNotFoundError(f'{self.src_dir} に .aselmdb なし')

        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        env = lmdb.open(
            str(self.out_path),
            map_size=self.map_size_gb * 1024 ** 3,
            subdir=False, readonly=False, meminit=False, map_async=True,
        )

        total_ok = total_skip = 0

        def handle_payload(txn, payload):
            nonlocal total_ok, total_skip
            if payload is None:
                total_skip += 1
                return txn
            key = f'{total_ok:09d}'.encode('ascii')
            txn.put(key, payload)
            total_ok += 1
            if total_ok % 10_000 == 0:
                txn.commit()
                txn = env.begin(write=True)
            return txn

        txn = env.begin(write=True)

        for lmdb_file in tqdm(lmdb_files, desc='Files', unit='file'):
            if self.n_workers <= 1:
                for rec_idx, rec in enumerate(iter_lmdb(lmdb_file)):
                    args = read_molecule(rec)   # ここで本当の例外を見えるようにする
                    payload = _process_record(args)
                    txn = handle_payload(txn, payload)
            else:
                ctx = mp.get_context('spawn')
                with ctx.Pool(processes=self.n_workers) as pool:
                    args_iter = (read_molecule(rec) for rec in iter_lmdb(lmdb_file))
                    results = pool.imap_unordered(_process_record, args_iter, chunksize=self.chunk_size)
                    for payload in results:
                        txn = handle_payload(txn, payload)

        txn.commit()

        with env.begin(write=True) as txn:
            txn.put(b'__len__', str(total_ok).encode('ascii'))

        env.sync()
        env.close()
        print(f'完了: {total_ok:,} 件保存, {total_skip:,} 件スキップ → {self.out_path}')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='OPoly26 前処理')
    p.add_argument('--src_dir', type=str, required=True)
    p.add_argument('--out_path', type=str, required=True)
    p.add_argument('--n_workers', type=int, default=max(1, os.cpu_count() // 2))
    p.add_argument('--chunk_size', type=int, default=64)
    p.add_argument('--map_size_gb', type=int, default=50)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    DatasetBuilder(
        src_dir=args.src_dir,
        out_path=args.out_path,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
        map_size_gb=args.map_size_gb,
    ).run()
