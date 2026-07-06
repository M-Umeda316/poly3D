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
import functools
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


def _safe_mol_records(lmdb_file, iter_lmdb, read_molecule, stats: dict):
    """iter_lmdb → read_molecule を安全に橋渡しするジェネレータ。

    read_molecule() が形状/型不整合などで例外を投げても、そのレコード
    だけをスキップしてカウントし、ファイル全体の処理は継続する。
    このジェネレータはメインプロセス側でのみ評価される（imap_unordered
    は個々の yield 値のみを worker へ pickle 送信するため、本関数自体
    が multiprocessing 境界を越えることはない）。
    """
    for rec in iter_lmdb(lmdb_file, stats=stats):
        try:
            yield read_molecule(rec)
        except Exception:
            stats['read_error'] = stats.get('read_error', 0) + 1
            continue


def _process_record(
    args: MolRecord,
    precompute_topology: bool = False,
) -> Optional[bytes]:
    """
    1 分子を処理して pickle bytes を返す。失敗時は None。
    spawn worker 内で実行されるためモジュールトップレベルに配置。

    precompute_topology=True の場合、dist_mat/triplets/quartets
    （edge_index のみに依存する不変値）も pickle に含める。
    dataset.py 側は「保存されていれば使う・無ければ計算する」
    フォールバックのため、デフォルト False の既存 lmdb は
    従来どおり dataset.py 側で計算される（後方互換）。
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

    if precompute_topology:
        try:
            import torch
            from poly3d.model.pos_bias import compute_graph_distance
            from poly3d.model.geo_losses import build_angle_triplets, build_dihedral_quartets

            n = data_dict['atom_type_idx'].shape[0]
            edge_index_t = torch.from_numpy(data_dict['edge_index'])
            dist_mat = compute_graph_distance(edge_index_t, n, max_dist=4).to(torch.int8)
            triplets = build_angle_triplets(edge_index_t, n)
            quartets = build_dihedral_quartets(edge_index_t, n)
            data_dict['dist_mat'] = dist_mat.numpy()
            data_dict['triplets'] = triplets.numpy()
            data_dict['quartets'] = quartets.numpy()
        except Exception:
            # トポロジー事前計算に失敗しても分子自体は保存する
            # （dataset.py 側が起動時にフォールバック計算する）
            data_dict.pop('dist_mat', None)
            data_dict.pop('triplets', None)
            data_dict.pop('quartets', None)

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
        precompute_topology: bool = False,
    ):
        self.src_dir = Path(src_dir)
        self.out_path = Path(out_path)
        self.n_workers = n_workers
        self.chunk_size = chunk_size
        self.map_size_gb = map_size_gb
        # デフォルト False: 既存 lmdb との互換性を保つため、新規前処理でも
        # 明示的に指定しない限りトポロジー事前計算フィールドは付与しない。
        self.precompute_topology = precompute_topology

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

        total_ok = 0
        total_skip_process = 0  # _process_record() が None を返した件数
        # decode_error/missing_key(iter_lmdb) と read_error(read_molecule) は
        # stats に集約する（_safe_mol_records / iter_lmdb 経由）
        stats: dict = {}

        def handle_payload(txn, payload):
            nonlocal total_ok, total_skip_process
            if payload is None:
                total_skip_process += 1
                return txn
            key = f'{total_ok:09d}'.encode('ascii')
            txn.put(key, payload)
            total_ok += 1
            if total_ok % 10_000 == 0:
                txn.commit()
                txn = env.begin(write=True)
            return txn

        txn = env.begin(write=True)

        process_fn = functools.partial(
            _process_record, precompute_topology=self.precompute_topology
        )

        for lmdb_file in tqdm(lmdb_files, desc='Files', unit='file'):
            if self.n_workers <= 1:
                for rec_idx, molrec in enumerate(
                    _safe_mol_records(lmdb_file, iter_lmdb, read_molecule, stats)
                ):
                    payload = process_fn(molrec)
                    txn = handle_payload(txn, payload)
            else:
                ctx = mp.get_context('spawn')
                with ctx.Pool(processes=self.n_workers) as pool:
                    args_iter = _safe_mol_records(lmdb_file, iter_lmdb, read_molecule, stats)
                    results = pool.imap_unordered(process_fn, args_iter, chunksize=self.chunk_size)
                    for payload in results:
                        txn = handle_payload(txn, payload)

        txn.commit()

        with env.begin(write=True) as txn:
            txn.put(b'__len__', str(total_ok).encode('ascii'))

        env.sync()
        env.close()

        decode_error = stats.get('decode_error', 0)
        missing_key = stats.get('missing_key', 0)
        read_error = stats.get('read_error', 0)
        total_skip = decode_error + missing_key + read_error + total_skip_process
        print(
            f'完了: {total_ok:,} 件保存, {total_skip:,} 件スキップ '
            f'(JSONデコード失敗={decode_error:,}, キー欠落={missing_key:,}, '
            f'read_molecule失敗={read_error:,}, 特徴量抽出失敗={total_skip_process:,}) '
            f'→ {self.out_path}'
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='OPoly26 前処理')
    p.add_argument('--src_dir', type=str, required=True)
    p.add_argument('--out_path', type=str, required=True)
    p.add_argument('--n_workers', type=int, default=max(1, os.cpu_count() // 2))
    p.add_argument('--chunk_size', type=int, default=64)
    p.add_argument('--map_size_gb', type=int, default=50)
    p.add_argument(
        '--precompute_topology', action='store_true',
        help='dist_mat/triplets/quartets を前処理時に計算して lmdb に保存する'
             '（デフォルト無効。既存 lmdb との後方互換のため、指定しない限り付与しない）'
    )
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    DatasetBuilder(
        src_dir=args.src_dir,
        out_path=args.out_path,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
        map_size_gb=args.map_size_gb,
        precompute_topology=args.precompute_topology,
    ).run()
