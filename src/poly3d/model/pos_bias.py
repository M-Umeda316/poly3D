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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


N_DIST_CHANNELS = 5  # 0:self, 1:bond, 2:angle, 3:dihedral, 4:far


def compute_graph_distance(edge_index: Tensor, num_nodes: int, max_dist: int = 4) -> Tensor:
    """
    BFS でグラフ距離行列を計算。

    scipy (C 実装) を使用し、純 Python 比 ~100 倍高速。
    DataLoader ワーカー内で呼び出すことで GPU スレッドのブロッキングを回避できる。

    Parameters
    ----------
    edge_index : (2, E) 有向エッジ
    num_nodes  : N
    max_dist   : それ以上の距離はすべて max_dist として扱う

    Returns
    -------
    dist : (N, N) int64  (自分自身=0, 到達不能=max_dist)
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    device = edge_index.device
    N = num_nodes

    if N == 0:
        return torch.zeros((0, 0), dtype=torch.long, device=device)

    ei = edge_index.cpu().numpy()
    rows, cols = ei[0], ei[1]

    if rows.size == 0:
        dist_np = np.full((N, N), max_dist, dtype=np.int64)
        np.fill_diagonal(dist_np, 0)
    else:
        data = np.ones(rows.size, dtype=np.float32)
        adj = csr_matrix((data, (rows, cols)), shape=(N, N))
        # C 実装の BFS ベース最短経路: O(N(N+E)) で高速
        dist_f = shortest_path(adj, method='D', directed=False, unweighted=True)
        dist_f = np.nan_to_num(dist_f, nan=float(max_dist), posinf=float(max_dist))
        dist_np = np.minimum(dist_f, max_dist).astype(np.int64)
        np.fill_diagonal(dist_np, 0)

    return torch.from_numpy(dist_np.copy()).to(device)


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
