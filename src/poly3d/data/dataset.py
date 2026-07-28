"""
処理済み lmdb から PyTorch Geometric Data オブジェクトを返す Dataset。

パフォーマンス設計:
  __getitem__ でワーカープロセスが以下を事前計算（GPU スレッドと並列実行）:
    - dist_mat  : グラフ距離行列 (N, N) int8  — DiT pos_bias 用
    - triplets  : 結合角トリプレット (T, 3) int64  — VAE angle_loss 用
    - quartets  : 二面角カルテット (Q, 4) int64  — VAE dihedral_loss 用

  これらは edge_index のみに依存する不変値（学習で同一分子を毎エポック
  再読み込みしても結果は変わらない）。以下の 2 段構えで再計算を減らす:
    1. build_dataset.py が事前計算済みの値を pickle に含めていれば、
       それをそのまま使う（前処理を再実行させない後方互換フォールバック）。
    2. 含まれていない場合（既存の旧形式 lmdb）は計算し、ワーカーローカルの
       LRU キャッシュ（idx → テンソル）に格納する。DataLoader は
       persistent_workers=True で動作するため、このキャッシュは
       エポックをまたいで有効であり、2 エポック目以降は同一 idx の
       再計算をスキップできる。

  collate_fn でノードオフセットを付与してバッチ内でインデックスを統合。
  学習ループは build_angle_triplets / build_dihedral_quartets を呼ばない。
"""
from __future__ import annotations

import os
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from poly3d.model.pos_bias import compute_graph_distance, mds_init_coords
from poly3d.model.geo_losses import build_angle_triplets, build_dihedral_quartets


class ConformerDataset(Dataset):
    def __init__(
        self,
        lmdb_path: str | Path,
        max_atoms: Optional[int] = None,
        precompute_topology: bool = True,
        topology_cache_size: int = 4096,
        mds_init: bool = False,
        dit_latent_lmdb: Optional[str] = None,
    ):
        self.lmdb_path = str(lmdb_path)
        self.max_atoms = max_atoms
        self.precompute_topology = precompute_topology
        # mds_init: True のとき __getitem__ で MDS 大域足場（init_scaffold）を付与する。
        # デフォルト False で完全後方互換（足場を一切計算せず既存挙動を不変に保つ）。
        self.mds_init = mds_init
        # dit_latent_lmdb: 事前計算済みの DiT 潜在（precompute_dit_latents.py 生成）を
        # 読む LMDB のパス。設定時のみ __getitem__ で同 idx の潜在を付与する。
        # デフォルト None で完全後方互換（DiT 潜在を一切読まない）。
        self.dit_latent_lmdb = str(dit_latent_lmdb) if dit_latent_lmdb is not None else None
        # idx → (dist_mat, triplets, quartets) のワーカーローカル LRU キャッシュ。
        # persistent_workers=True 環境ではエポックをまたいで生存するため、
        # 2 エポック目以降は同一 idx の再計算をスキップできる。
        # サイズ上限を設けデータセット全体をメモリに載せないようにする
        # （0 以下でキャッシュ無効化）。
        self.topology_cache_size = topology_cache_size
        self._topo_cache: 'OrderedDict[int, tuple]' = OrderedDict()
        # idx → init_scaffold(n,3) のワーカーローカル LRU キャッシュ（_topo_cache と同機構）。
        self._scaffold_cache: 'OrderedDict[int, torch.Tensor]' = OrderedDict()
        self._env: Optional[lmdb.Environment] = None
        # DiT 潜在 LMDB env（worker ごとに遅延オープン。_env と同機構）。
        self._dit_env: Optional[lmdb.Environment] = None

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

    def _open_dit_env(self) -> lmdb.Environment:
        return lmdb.open(
            self.dit_latent_lmdb, subdir=False, readonly=True,
            lock=False, readahead=False, meminit=False, max_readers=256,
        )

    def _get_dit_env(self) -> lmdb.Environment:
        if self._dit_env is None:
            self._dit_env = self._open_dit_env()
        return self._dit_env

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
            dist_mat, triplets, quartets = self._get_topology(idx, d, edge_index_t, n)
            kwargs['dist_mat'] = dist_mat
            kwargs['triplets'] = triplets
            kwargs['quartets'] = quartets

        if self.mds_init:
            # MDS 大域足場（純トポロジー、座標非依存）。collate で pos と同様に縦連結。
            kwargs['init_scaffold'] = self._get_scaffold(idx, edge_index_t, n)

        if self.dit_latent_lmdb is not None:
            # 事前計算済み DiT 潜在（このレコード自身の原子順で保存されている）。
            # キーが無ければ付与しない（後方互換・学習側でフォールバック）。
            z_dit = self._get_dit_latent(idx, n)
            if z_dit is not None:
                kwargs['z_dit'] = z_dit

        return Data(**kwargs)

    def _get_dit_latent(self, idx: int, n: int) -> Optional[torch.Tensor]:
        """事前計算済み DiT 潜在 (N, latent_dim) float32 を取得する。

        precompute_dit_latents.py は同じ idx キー f'{idx:09d}' で
        (K, N, latent_dim) float32 numpy を pickle 保存している。K 個あれば
        1 個をランダムに選ぶ（同一レコードの複数サンプルからの多様化）。
        キーが無い（max_atoms 超過等でスキップされた）場合は None を返し、
        学習側は z_dit 無しのバッチとして従来経路にフォールバックする。

        保存時と同一 dataset・同一原子順なので、返るテンソルの行 i は
        pos の原子 i に対応する（リマップ不要・構造的に正しい）。
        """
        key = f'{idx:09d}'.encode('ascii')
        with self._get_dit_env().begin() as txn:
            val = txn.get(key)
        if val is None:
            return None
        arr = pickle.loads(val)   # (K, N, latent_dim) float32 numpy
        if arr.ndim != 3 or arr.shape[1] != n:
            # 想定外の形状（原子数不一致など）は安全側でスキップ＝フォールバック。
            return None
        k = int(torch.randint(arr.shape[0], (1,)).item()) if arr.shape[0] > 1 else 0
        return torch.from_numpy(np.ascontiguousarray(arr[k])).float()   # (N, latent_dim)

    def _get_topology(
        self, idx: int, d: dict, edge_index_t: torch.Tensor, n: int
    ) -> tuple:
        """
        dist_mat / triplets / quartets を取得する。

        優先順位:
          1. pickle 内に事前計算済みの値があればそれを使う
             （build_dataset.py --precompute_topology 済みの新形式 lmdb）。
          2. ワーカーローカル LRU キャッシュに hit したらそれを使う
             （persistent_workers=True 前提でエポックをまたいで有効）。
          3. どちらも無ければ計算し、キャッシュに格納する
             （既存の旧形式 lmdb はこの経路のまま従来どおり動作する）。
        """
        if 'dist_mat' in d and 'triplets' in d and 'quartets' in d:
            dist_mat = torch.from_numpy(d['dist_mat']).to(torch.int8)
            triplets = torch.from_numpy(d['triplets'].astype(np.int64))
            quartets = torch.from_numpy(d['quartets'].astype(np.int64))
            return dist_mat, triplets, quartets

        if self.topology_cache_size > 0 and idx in self._topo_cache:
            self._topo_cache.move_to_end(idx)
            dist_mat, triplets, quartets = self._topo_cache[idx]
            return dist_mat.clone(), triplets.clone(), quartets.clone()

        # ── ワーカープロセスで計算（GPU スレッドと並列実行）────────────────
        # グラフ距離行列: int8 で省メモリ（値域 0-4）
        dist_mat = compute_graph_distance(edge_index_t, n, max_dist=4).to(torch.int8)

        # 結合角トリプレット・二面角カルテット（トポロジーのみ依存）
        # 単分子（N~50）なので Python ループも十分高速
        triplets = build_angle_triplets(edge_index_t, n)    # (T, 3)
        quartets = build_dihedral_quartets(edge_index_t, n) # (Q, 4)

        if self.topology_cache_size > 0:
            self._topo_cache[idx] = (dist_mat, triplets, quartets)
            self._topo_cache.move_to_end(idx)
            if len(self._topo_cache) > self.topology_cache_size:
                self._topo_cache.popitem(last=False)

        return dist_mat, triplets, quartets

    def _get_scaffold(
        self, idx: int, edge_index_t: torch.Tensor, n: int
    ) -> torch.Tensor:
        """
        MDS 大域足場 init_scaffold (n, 3) float32 を取得する。

        _get_topology と同じワーカーローカル LRU キャッシュ機構を用いる
        （persistent_workers=True 前提でエポックをまたいで有効。過学習・固定分子
        では初回のみ計算し以降キャッシュ）。足場は座標非依存な純トポロジー量なので
        毎エポック同一 idx で結果は変わらない。
        """
        if self.topology_cache_size > 0 and idx in self._scaffold_cache:
            self._scaffold_cache.move_to_end(idx)
            return self._scaffold_cache[idx].clone()

        # ── ワーカープロセスで計算（GPU スレッドと並列実行）────────────────
        scaffold = mds_init_coords(edge_index_t, n)   # (n, 3) float32

        if self.topology_cache_size > 0:
            self._scaffold_cache[idx] = scaffold
            self._scaffold_cache.move_to_end(idx)
            if len(self._scaffold_cache) > self.topology_cache_size:
                self._scaffold_cache.popitem(last=False)
            return scaffold.clone()

        return scaffold

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
        if self._dit_env is not None:
            self._dit_env.close()
            self._dit_env = None


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
    all_scaffolds = []
    all_zdit      = []
    has_topology  = hasattr(valid[0], 'dist_mat')
    has_scaffold  = hasattr(valid[0], 'init_scaffold')
    # z_dit は「バッチ内の全レコードが持つ」ときだけ有効化する（最も単純で安全）。
    # 一部だけ持つ混在バッチでは z_dit を作らず、学習側は従来経路にフォールバックする。
    has_zdit      = all(hasattr(d, 'z_dit') for d in valid)

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
        if has_scaffold:
            # init_scaffold: pos と同様に縦連結（ノードオフセット不要の単純 cat）。
            # 分子順は valid の並び＝pos の並びと一致する。
            all_scaffolds.append(d.init_scaffold)
            del d.init_scaffold
        # z_dit は PyG の自動 collation に任せず手動処理する（混在バッチでの
        # from_data_list 例外を避けるため、持っているものは必ず取り出して消す）。
        if hasattr(d, 'z_dit'):
            if has_zdit:
                all_zdit.append(d.z_dit)
            del d.z_dit
        offset += n

    pyg_batch = Batch.from_data_list(valid)

    if has_scaffold:
        # (total_N, 3) float。pos と同じ行順（分子順・原子順）で連結される。
        pyg_batch.init_scaffold = torch.cat(all_scaffolds, dim=0)

    if has_zdit and all_zdit:
        # (total_N, latent_dim) float。pos と同じ block-diagonal ノード順（分子順・
        # 原子順）で連結される（各 z_dit の行順は保存時にそのレコードの pos と一致）。
        pyg_batch.z_dit = torch.cat(all_zdit, dim=0)
        assert pyg_batch.z_dit.size(0) == total_n, (
            f'z_dit ノード数不一致: {pyg_batch.z_dit.size(0)} != {total_n}')

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
            ds._dit_env = None


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
    subset_indices: Optional[list] = None,
    mds_init: bool = False,
    dit_latent_lmdb: Optional[str] = None,
) -> torch.utils.data.DataLoader:
    dataset = ConformerDataset(
        lmdb_path, max_atoms=max_atoms, precompute_topology=precompute_topology,
        mds_init=mds_init, dit_latent_lmdb=dit_latent_lmdb,
    )
    if subset_indices is not None:
        from torch.utils.data import Subset
        dataset = Subset(dataset, subset_indices)
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
