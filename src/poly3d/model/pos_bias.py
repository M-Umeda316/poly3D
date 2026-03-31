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
from torch import Tensor


N_DIST_CHANNELS = 5  # 0:self, 1:bond, 2:angle, 3:dihedral, 4:far


def compute_graph_distance(edge_index: Tensor, num_nodes: int, max_dist: int = 4) -> Tensor:
    """
    BFS でグラフ距離行列を計算。

    Parameters
    ----------
    edge_index : (2, E) 有向エッジ
    num_nodes  : N
    max_dist   : それ以上の距離はすべて max_dist として扱う

    Returns
    -------
    dist : (N, N) int32  (自分自身=0, 到達不能=max_dist)
    """
    device = edge_index.device
    # 隣接リストを CPU で構築
    src, dst = edge_index.cpu().tolist()[0], edge_index.cpu().tolist()[1]

    adj = [[] for _ in range(num_nodes)]
    for s, d in zip(src, dst):
        adj[s].append(d)

    dist = [[max_dist] * num_nodes for _ in range(num_nodes)]
    for start in range(num_nodes):
        dist[start][start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]; head += 1
            if dist[start][u] >= max_dist:
                continue
            for v in adj[u]:
                if dist[start][v] > dist[start][u] + 1:
                    dist[start][v] = dist[start][u] + 1
                    queue.append(v)

    return torch.tensor(dist, dtype=torch.long, device=device)  # (N, N)


def dist_to_onehot(dist_mat: Tensor) -> Tensor:
    """
    (N, N) int → (N, N, 5) float one-hot

    0: self (dist==0)
    1: bond (dist==1)
    2: angle (dist==2)
    3: dihedral (dist==3)
    4: far (dist>=4)
    """
    N = dist_mat.size(0)
    oh = torch.zeros(N, N, N_DIST_CHANNELS, device=dist_mat.device)
    oh[:, :, 0] = (dist_mat == 0).float()
    oh[:, :, 1] = (dist_mat == 1).float()
    oh[:, :, 2] = (dist_mat == 2).float()
    oh[:, :, 3] = (dist_mat == 3).float()
    oh[:, :, 4] = (dist_mat >= 4).float()
    return oh


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
