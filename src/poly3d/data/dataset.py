"""
処理済み lmdb から PyTorch Geometric Data オブジェクトを返す Dataset。

パフォーマンス設計:
  __getitem__ でワーカープロセスが以下を事前計算（GPU スレッドと並列実行）:
    - dist_mat  : グラフ距離行列 (N, N) int8  — DiT pos_bias 用
    - triplets  : 結合角トリプレット (T, 3) int64  — VAE angle_loss 用
    - quartets  : 二面角カルテット (Q, 4) int64  — VAE dihedral_loss 用

  collate_fn でノードオフセットを付与してバッチ内でインデックスを統合。
  学習ループは build_angle_triplets / build_dihedral_quartets を呼ばない。
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from poly3d.model.pos_bias import compute_graph_distance
from poly3d.model.geo_losses import build_angle_triplets, build_dihedral_quartets


class ConformerDataset(Dataset):
    def __init__(
        self,
        lmdb_path: str | Path,
        max_atoms: Optional[int] = None,
        precompute_topology: bool = True,
    ):
        self.lmdb_path = str(lmdb_path)
        self.max_atoms = max_atoms
        self.precompute_topology = precompute_topology
        self._env: Optional[lmdb.Environment] = None

        env = self._open_env()
        with env.begin() as txn:
            meta = txn.get(b'__len__')
            self._len = int(meta.decode('ascii')) if meta else txn.stat()['entries']
        env.close()

    def _open_env(self) -> lmdb.Environment:
        return lmdb.open(
            self.lmdb_path, subdir=False, readonly=True,
            lock=False, readahead=False, meminit=False, max_readers=256,
        )

    def _get_env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = self._open_env()
        return self._env

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Optional[Data]:
        key = f'{idx:09d}'.encode('ascii')
        with self._get_env().begin() as txn:
            val = txn.get(key)
        if val is None:
            return None

        d = pickle.loads(val)
        n = d['atom_type_idx'].shape[0]

        if self.max_atoms is not None and n > self.max_atoms:
            return None

        edge_index_t = torch.from_numpy(d['edge_index'])
        kwargs = dict(
            atom_type_idx=torch.from_numpy(d['atom_type_idx'].astype(np.int64)),
            hyb_idx=torch.from_numpy(d['hyb_idx'].astype(np.int64)),
            atom_cont=torch.from_numpy(d['atom_cont']),
            bond_type_idx=torch.from_numpy(d['bond_type_idx'].astype(np.int64)),
            bond_cont=torch.from_numpy(d['bond_cont']),
            edge_index=edge_index_t,
            pos=torch.from_numpy(d['pos']),
            atomic_nums=torch.from_numpy(d['atomic_nums'].astype(np.int64)),
            sid=d.get('sid', ''),
            num_nodes=n,
        )
        if 'rwpe' in d:
            kwargs['rwpe'] = torch.from_numpy(d['rwpe'])
        if 'lappe' in d:
            kwargs['lappe'] = torch.from_numpy(d['lappe'])

        if self.precompute_topology:
            # ── ワーカープロセスで事前計算（GPU スレッドと並列実行）────────────
            # グラフ距離行列: int8 で省メモリ（値域 0-4）
            kwargs['dist_mat'] = compute_graph_distance(
                edge_index_t, n, max_dist=4
            ).to(torch.int8)

            # 結合角トリプレット・二面角カルテット（トポロジーのみ依存）
            # 単分子（N~50）なので Python ループも十分高速
            kwargs['triplets'] = build_angle_triplets(edge_index_t, n)    # (T, 3)
            kwargs['quartets'] = build_dihedral_quartets(edge_index_t, n) # (Q, 4)

        return Data(**kwargs)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


def collate_fn(batch: list) -> Optional[Batch]:
    valid = [d for d in batch if d is not None]
    if not valid:
        return None

    # ── トポロジーテンソルをノードオフセット付きで統合 ─────────────────────────
    # PyG の Batch.from_data_list に渡す前に取り出し、手動でオフセット処理する。
    # (dist_mat は 2D なので PyG が正しく扱えない。triplets/quartets は
    #  __inc__ を定義していないので同様に手動処理する。)

    all_dist_mats = []
    all_triplets  = []
    all_quartets  = []
    has_topology  = hasattr(valid[0], 'dist_mat')

    sizes = [d.num_nodes for d in valid]
    total_n = sum(sizes)
    offset = 0

    for d in valid:
        n = d.num_nodes
        if has_topology:
            # dist_mat: ブロック対角行列に埋め込む（異分子間 = 4 = far）
            all_dist_mats.append((offset, n, d.dist_mat))
            del d.dist_mat

            # triplets / quartets: ノードインデックスにオフセット加算
            if d.triplets.size(0) > 0:
                all_triplets.append(d.triplets + offset)
            if d.quartets.size(0) > 0:
                all_quartets.append(d.quartets + offset)
            del d.triplets
            del d.quartets
        offset += n

    pyg_batch = Batch.from_data_list(valid)

    if has_topology:
        # dist_mat: (total_N, total_N) int8 ブロック対角
        dist_mat = torch.full((total_n, total_n), 4, dtype=torch.int8)
        for (off, n, dm) in all_dist_mats:
            dist_mat[off:off + n, off:off + n] = dm
        pyg_batch.dist_mat = dist_mat

        # triplets / quartets
        pyg_batch.triplets = (
            torch.cat(all_triplets, dim=0) if all_triplets
            else torch.zeros((0, 3), dtype=torch.long)
        )
        pyg_batch.quartets = (
            torch.cat(all_quartets, dim=0) if all_quartets
            else torch.zeros((0, 4), dtype=torch.long)
        )

    return pyg_batch


def worker_init_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        ds = worker_info.dataset
        while hasattr(ds, 'dataset'):
            ds = ds.dataset
        if isinstance(ds, ConformerDataset):
            ds._env = None


def make_dataloader(
    lmdb_path: str | Path,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    max_atoms: Optional[int] = 300,
    pin_memory: bool = True,
    sampler: Optional[torch.utils.data.Sampler] = None,
    prefetch_factor: int = 4,
    precompute_topology: bool = True,
) -> torch.utils.data.DataLoader:
    dataset = ConformerDataset(
        lmdb_path, max_atoms=max_atoms, precompute_topology=precompute_topology
    )
    pf = prefetch_factor if num_workers > 0 else None
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=pf,
    )
