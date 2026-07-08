"""
Positional Biased Attention (グラフ距離ベース)

論文 Section 4.4: グラフ距離を 5 チャンネルの one-hot に変換し、
MLP でスカラーバイアスを計算。Transformer の attention logit に加算する。

5 チャンネル:
  0: 自分自身 (i == j)
  1: 結合 (グラフ距離 = 1)
  2: angle (グラフ距離 = 2)
  3: dihedral (グラフ距離 = 3)
  4: それ以上 (>= 4)

使用方法:
  bias_builder = GraphDistanceBias(n_heads=8)
  # バッチ内の各分子ごとに呼び出し or パック済み行列を渡す
  bias = bias_builder(dist_mat)  # (N, N, n_heads) → attention に加算
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


N_DIST_CHANNELS = 5  # 0:self, 1:bond, 2:angle, 3:dihedral, 4:far


def compute_graph_distance(
    edge_index: Tensor, num_nodes: int, max_dist: Optional[int] = 4
) -> Tensor:
    """
    BFS でグラフ距離行列を計算。

    scipy (C 実装) を使用し、純 Python 比 ~100 倍高速。
    DataLoader ワーカー内で呼び出すことで GPU スレッドのブロッキングを回避できる。

    Parameters
    ----------
    edge_index : (2, E) 有向エッジ
    num_nodes  : N
    max_dist   : それ以上の距離はすべて max_dist にクランプする。
                 `None` または `<= 0` を渡すとクランプせず完全ホップ距離を返す
                 （MDS init / 末端プロキシ用途で再利用する共通部品）。

    Returns
    -------
    dist : (N, N) int64  (自分自身=0)
           クランプ時: 到達不能=max_dist。
           非クランプ時: 到達不能=有限最大ホップ+1（連結成分が 1 つなら発生しない）。
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    device = edge_index.device
    N = num_nodes

    clamp = max_dist is not None and max_dist > 0

    if N == 0:
        return torch.zeros((0, 0), dtype=torch.long, device=device)

    ei = edge_index.cpu().numpy()
    rows, cols = ei[0], ei[1]

    if rows.size == 0:
        # エッジ無し: 非対角は全て到達不能。クランプ時は max_dist、
        # 非クランプ時は 1（有限最大ホップ 0 + 1）で埋める。
        fill = max_dist if clamp else 1
        dist_np = np.full((N, N), fill, dtype=np.int64)
        np.fill_diagonal(dist_np, 0)
    else:
        data = np.ones(rows.size, dtype=np.float32)
        adj = csr_matrix((data, (rows, cols)), shape=(N, N))
        # C 実装の BFS ベース最短経路: O(N(N+E)) で高速
        dist_f = shortest_path(adj, method='D', directed=False, unweighted=True)
        if clamp:
            dist_f = np.nan_to_num(dist_f, nan=float(max_dist), posinf=float(max_dist))
            dist_np = np.minimum(dist_f, max_dist).astype(np.int64)
        else:
            # 非クランプ: 到達不能(inf/nan)は「有限最大ホップ+1」で安全に処理し、
            # float ホップ距離をそのまま int 化する。
            finite = dist_f[np.isfinite(dist_f)]
            unreachable = float(finite.max()) + 1.0 if finite.size > 0 else 1.0
            dist_f = np.nan_to_num(dist_f, nan=unreachable, posinf=unreachable)
            dist_np = dist_f.astype(np.int64)
        np.fill_diagonal(dist_np, 0)

    return torch.from_numpy(dist_np.copy()).to(device)


def mds_init_coords(
    edge_index: Tensor, num_nodes: int, bond_scale: float = 1.5
) -> Tensor:
    """
    古典 MDS（Torgerson）でトポロジー由来の 3D 大域足場を生成する。

    デコーダの初期座標を「原子ごと独立の MLP」から「大域整合した足場＋小 MLP
    補正」に置換するための、座標非依存（純トポロジー）な初期スキャフォールド。

    アルゴリズム（分子ごと）:
      1. 完全ホップ距離 D(n×n) を取得（`compute_graph_distance(max_dist=None)` を
         再利用。shortest_path 等を再実装しない）。
      2. D2 = D², 二重中心化 B = -0.5·J·D2·J（J = I - 11ᵀ/n）。
      3. `torch.linalg.eigh(B)` の上位 3 正固有値 λ・固有ベクトル v から
         X = [v1·√λ1, v2·√λ2, v3·√λ3] ∈ R^{n×3}。
      4. ホップ→Å: 結合エッジ（edge_index）上の平均埋め込み距離が bond_scale(Å)
         になる一様スケール s を掛ける（結合ゼロ時 s=1）。

    エッジケース（設計 §3.2）:
      - 正固有値が 3 未満（線状/微小分子）: 不足次元は 0 のまま。
      - 正固有値 < 3 または n≤1: EGNN の d²=0 縮退回避のため std=1e-2 の
        微小乱数を全体へ加算する。
      - 到達不能ペア: compute_graph_distance 側が「有限最大ホップ+1」で処理済み。

    決定性（設計 §3.3）:
      - `eigh` は対称・決定的。微小乱数は **固定シードの `torch.Generator`** で
        生成し、同一入力なら必ず同一出力になるようにする（ワーカー再計算や LRU
        キャッシュとの整合、および `torch.manual_seed` 等のグローバル乱数状態への
        非依存を保証する）。`torch.randn`（グローバル状態依存）は使わない。
      - 足場の絶対姿勢（回転・並進・鏡映・固有ベクトル符号）は任意だが、デコーダ
        の損失（Kabsch RMSD / distmat 系）はすべて回転・並進不変なので問題ない。

    CPU で完結する（DataLoader ワーカーからの呼び出しを前提とする）。

    Parameters
    ----------
    edge_index : (2, E) 有向エッジ
    num_nodes  : n
    bond_scale : 結合原子間の平均埋め込み距離（Å）。デフォルト 1.5

    Returns
    -------
    x : (n, 3) float32  トポロジー由来の初期座標足場
    """
    device = edge_index.device
    n = int(num_nodes)

    # n≤1: 足場を作れない。零ベクトル（n==1 は微小乱数で非ゼロ化）を返す。
    if n <= 1:
        x = torch.zeros((n, 3), dtype=torch.float32, device=device)
        if n == 1:
            g = torch.Generator().manual_seed(0)
            x = x + torch.randn(n, 3, generator=g).to(device) * 1e-2
        return x

    # 完全ホップ距離（非クランプ）。既存共通部品を再利用（再実装禁止）。
    D = compute_graph_distance(edge_index, n, max_dist=None).to(torch.float64)

    # 二重中心化 B = -0.5·J·D2·J,  J = I - 11ᵀ/n
    # （torch.eye(n) - 1/n は対角 1-1/n・非対角 -1/n = I - (1/n)11ᵀ に一致）
    D2 = D * D
    J = torch.eye(n, dtype=torch.float64, device=D.device) - (1.0 / n)
    B = -0.5 * (J @ D2 @ J)
    B = 0.5 * (B + B.transpose(0, 1))   # 数値対称化（eigh の前提を明示保証）

    eigvals, eigvecs = torch.linalg.eigh(B)   # 昇順の実固有値・直交固有ベクトル
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # 上位から正固有値のみ採用し X=(n,3) を構成（不足次元は 0 のまま）
    x = torch.zeros((n, 3), dtype=torch.float64, device=D.device)
    n_pos = 0
    for k in range(min(3, n)):
        lam = eigvals[k]
        if lam <= 0:
            break
        x[:, k] = eigvecs[:, k] * torch.sqrt(lam)
        n_pos += 1

    x = x.to(torch.float32)

    # ホップ→Å スケール: 結合エッジ上の平均埋め込み距離を bond_scale に合わせる
    if edge_index.numel() > 0:
        ei = edge_index.to(x.device)
        bond_d = (x[ei[0]] - x[ei[1]]).norm(dim=-1)
        mean_bd = bond_d.mean()
        if mean_bd > 1e-8:
            x = x * (bond_scale / mean_bd)

    # 正固有値が 3 未満（線状/微小分子）: d²=0 縮退回避に決定的微小乱数を加算
    if n_pos < 3:
        g = torch.Generator().manual_seed(0)
        x = x + torch.randn(n, 3, generator=g) * 1e-2

    return x.to(device)


def dist_to_onehot(dist_mat: Tensor) -> Tensor:
    """
    (N, N) int → (N, N, 5) float one-hot

    0: self (dist==0)
    1: bond (dist==1)
    2: angle (dist==2)
    3: dihedral (dist==3)
    4: far (dist>=4)
    """
    return F.one_hot(dist_mat.clamp(max=4).long(), num_classes=N_DIST_CHANNELS).float()


class GraphDistanceBias(nn.Module):
    """
    グラフ距離 one-hot → MLP → attention スカラーバイアス

    Parameters
    ----------
    n_heads : attention ヘッド数
    """

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.mlp = nn.Sequential(
            nn.Linear(N_DIST_CHANNELS, n_heads * 2),
            nn.SiLU(),
            nn.Linear(n_heads * 2, n_heads),
        )

    def forward(self, dist_onehot: Tensor) -> Tensor:
        """
        Parameters
        ----------
        dist_onehot : (N, N, 5)

        Returns
        -------
        bias : (N, N, n_heads)  → attention logit (N, H, N, N) に加算する際は転置
        """
        return self.mlp(dist_onehot)   # (N, N, n_heads)
