"""
処理済み lmdb から PyTorch Geometric Data オブジェクトを返す Dataset。

各エントリは pickle 化された dict（model/features.py の mol_to_data 出力）:
    atom_type_idx : np.int32  (N,)
    hyb_idx       : np.int32  (N,)
    atom_cont     : np.float32 (N, ATOM_CONT_DIM)
    bond_type_idx : np.int32  (2E,)
    bond_cont     : np.float32 (2E, BOND_CONT_DIM)
    edge_index    : np.int64  (2, 2E)
    rwpe          : np.float32 (N, RWPE_DIM)   オプション
    lappe         : np.float32 (N, LAPPE_DIM)  オプション
    pos           : np.float32 (N, 3)
    atomic_nums   : np.int32  (N,)
    sid           : str
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


class ConformerDataset(Dataset):
    def __init__(
        self,
        lmdb_path: str | Path,
        max_atoms: Optional[int] = None,
    ):
        self.lmdb_path = str(lmdb_path)
        self.max_atoms = max_atoms
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

        kwargs = dict(
            atom_type_idx=torch.from_numpy(d['atom_type_idx'].astype(np.int64)),
            hyb_idx=torch.from_numpy(d['hyb_idx'].astype(np.int64)),
            atom_cont=torch.from_numpy(d['atom_cont']),
            bond_type_idx=torch.from_numpy(d['bond_type_idx'].astype(np.int64)),
            bond_cont=torch.from_numpy(d['bond_cont']),
            edge_index=torch.from_numpy(d['edge_index']),
            pos=torch.from_numpy(d['pos']),
            atomic_nums=torch.from_numpy(d['atomic_nums'].astype(np.int64)),
            sid=d.get('sid', ''),
            num_nodes=n,
        )
        if 'rwpe' in d:
            kwargs['rwpe'] = torch.from_numpy(d['rwpe'])
        if 'lappe' in d:
            kwargs['lappe'] = torch.from_numpy(d['lappe'])

        return Data(**kwargs)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


def collate_fn(batch: list) -> Optional[Batch]:
    valid = [d for d in batch if d is not None]
    if not valid:
        return None
    return Batch.from_data_list(valid)


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
) -> torch.utils.data.DataLoader:
    dataset = ConformerDataset(lmdb_path, max_atoms=max_atoms)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
