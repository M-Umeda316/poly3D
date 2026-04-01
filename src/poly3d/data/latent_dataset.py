"""
事前エンコード済み潜在変数を読み込む Dataset（DiT Stage 2 専用）。

precompute_latents.py で生成した LMDB を読み込む。
各エントリには以下が含まれる:
    z0        : np.float32 (N, latent_dim)  — VAE Encoder の mu
    cond      : np.float32 (N, cond_dim)   — ConditionalEncoder 出力
    e_cond    : np.float32 (E, edge_dim)   — エッジ条件ベクトル
    edge_index: np.int64   (2, E)
    num_nodes : int
    dist_mat  : np.int8    (N, N)   optional — グラフ距離行列

ConformerDataset の代わりにこれを使うと、DiT 学習時に
ConditionalEncoder + VAE Encoder の推論コストがゼロになる。
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data


class LatentDataset(Dataset):
    def __init__(self, lmdb_path: str | Path):
        self.lmdb_path = str(lmdb_path)
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
        n = d['num_nodes']

        kwargs = dict(
            z0=torch.from_numpy(d['z0']),              # (N, latent_dim)
            cond=torch.from_numpy(d['cond']),          # (N, cond_dim)
            e_cond=torch.from_numpy(d['e_cond']),      # (E, edge_dim)
            edge_index=torch.from_numpy(d['edge_index']),  # (2, E)
            num_nodes=n,
        )
        if 'dist_mat' in d:
            kwargs['dist_mat'] = torch.from_numpy(d['dist_mat'])   # (N, N) int8

        return Data(**kwargs)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


def latent_collate_fn(batch: list) -> Optional[Batch]:
    valid = [d for d in batch if d is not None]
    if not valid:
        return None

    # dist_mat をブロック対角行列として組み立て（ConformerDataset と同じ処理）
    has_dist = hasattr(valid[0], 'dist_mat')
    sizes = [d.num_nodes for d in valid]
    total_n = sum(sizes)
    offset = 0

    dist_blocks = []
    for d in valid:
        n = d.num_nodes
        if has_dist:
            dist_blocks.append((offset, n, d.dist_mat))
            del d.dist_mat
        offset += n

    pyg_batch = Batch.from_data_list(valid)

    if has_dist:
        dist_mat = torch.full((total_n, total_n), 4, dtype=torch.int8)
        for (off, n, dm) in dist_blocks:
            dist_mat[off:off + n, off:off + n] = dm
        pyg_batch.dist_mat = dist_mat

    return pyg_batch


def worker_init_fn_latent(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        ds = worker_info.dataset
        while hasattr(ds, 'dataset'):
            ds = ds.dataset
        if isinstance(ds, LatentDataset):
            ds._env = None


def make_latent_dataloader(
    lmdb_path: str | Path,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 8,
    pin_memory: bool = True,
    sampler=None,
    prefetch_factor: int = 4,
) -> torch.utils.data.DataLoader:
    dataset = LatentDataset(lmdb_path)
    pf = prefetch_factor if num_workers > 0 else None
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=latent_collate_fn,
        worker_init_fn=worker_init_fn_latent,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=pf,
    )
