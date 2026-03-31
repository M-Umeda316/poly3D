"""
幾何学的補助損失: 結合角・二面角 MSE

angle_loss   : 3原子パス (i-j-k) の結合角 MSE
dihedral_loss: 4原子パス (i-j-k-l) の二面角 損失 (1 - cos(φ_pred - φ_gt))
"""
from __future__ import annotations

import torch
from torch import Tensor


def _angle_between(v1: Tensor, v2: Tensor, eps: float = 1e-8) -> Tensor:
    """v1, v2: (..., 3) → (...,) ラジアン"""
    n1 = v1.norm(dim=-1, keepdim=True).clamp(min=eps)
    n2 = v2.norm(dim=-1, keepdim=True).clamp(min=eps)
    cos = (v1 / n1 * (v2 / n2)).sum(dim=-1).clamp(-1 + eps, 1 - eps)
    return torch.acos(cos)


def _dihedral(p0: Tensor, p1: Tensor, p2: Tensor, p3: Tensor, eps: float = 1e-8) -> Tensor:
    """
    4点の二面角を計算。(..., 3) → (...,) ラジアン
    """
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2

    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)

    n1_norm = n1.norm(dim=-1, keepdim=True).clamp(min=eps)
    n2_norm = n2.norm(dim=-1, keepdim=True).clamp(min=eps)
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    cos_phi = (n1 * n2).sum(dim=-1).clamp(-1 + eps, 1 - eps)
    return torch.acos(cos_phi)


def build_angle_triplets(edge_index: Tensor, num_nodes: int) -> Tensor:
    """
    edge_index (2, E) から 3原子トリプレット (i, j, k) を構築。
    j を中心とする i-j-k ペア。

    Returns
    -------
    triplets : (T, 3) int64  [i, j, k]
    """
    src, dst = edge_index  # src→dst
    # j をキーにして、j に入るエッジ(src→j) と j から出るエッジ(j→dst) をマッチング
    triplets = []
    # adj: node → list of neighbors (src node)
    adj_in = [[] for _ in range(num_nodes)]
    for e_idx in range(src.size(0)):
        adj_in[dst[e_idx].item()].append(src[e_idx].item())

    for j in range(num_nodes):
        neighbors = adj_in[j]
        for ni in range(len(neighbors)):
            for nk in range(len(neighbors)):
                i, k = neighbors[ni], neighbors[nk]
                if i != k:
                    triplets.append([i, j, k])

    if not triplets:
        return torch.zeros((0, 3), dtype=torch.long, device=edge_index.device)
    return torch.tensor(triplets, dtype=torch.long, device=edge_index.device)


def build_dihedral_quartets(edge_index: Tensor, num_nodes: int) -> Tensor:
    """
    edge_index (2, E) から 4原子カルテット (i, j, k, l) を構築。
    j-k を軸とする i-j-k-l。

    Returns
    -------
    quartets : (Q, 4) int64  [i, j, k, l]
    """
    src, dst = edge_index
    # 隣接リスト構築
    adj = [[] for _ in range(num_nodes)]
    for e in range(src.size(0)):
        adj[src[e].item()].append(dst[e].item())

    # j-k エッジを軸として i∈N(j)\{k}, l∈N(k)\{j}
    quartets = []
    # 有向エッジセットから無向ペアを取得
    edge_set = set()
    for e in range(src.size(0)):
        j, k = src[e].item(), dst[e].item()
        if j < k:
            edge_set.add((j, k))

    for j, k in edge_set:
        for i in adj[j]:
            if i == k:
                continue
            for l in adj[k]:
                if l == j or l == i:
                    continue
                quartets.append([i, j, k, l])

    if not quartets:
        return torch.zeros((0, 4), dtype=torch.long, device=edge_index.device)
    return torch.tensor(quartets, dtype=torch.long, device=edge_index.device)


def angle_loss(pos_pred: Tensor, pos_gt: Tensor, triplets: Tensor) -> Tensor:
    """
    Parameters
    ----------
    pos_pred, pos_gt : (N, 3)
    triplets         : (T, 3) int64 [i, j, k]

    Returns
    -------
    scalar loss
    """
    if triplets.size(0) == 0:
        return pos_pred.new_zeros(())

    i, j, k = triplets[:, 0], triplets[:, 1], triplets[:, 2]

    v1_pred = pos_pred[i] - pos_pred[j]
    v2_pred = pos_pred[k] - pos_pred[j]
    theta_pred = _angle_between(v1_pred, v2_pred)

    with torch.no_grad():
        v1_gt = pos_gt[i] - pos_gt[j]
        v2_gt = pos_gt[k] - pos_gt[j]
        theta_gt = _angle_between(v1_gt, v2_gt)

    return torch.nn.functional.mse_loss(theta_pred, theta_gt)


def dihedral_loss(pos_pred: Tensor, pos_gt: Tensor, quartets: Tensor) -> Tensor:
    """
    Parameters
    ----------
    pos_pred, pos_gt : (N, 3)
    quartets         : (Q, 4) int64 [i, j, k, l]

    Returns
    -------
    scalar loss  (1 - cos(φ_pred - φ_gt) の平均)
    """
    if quartets.size(0) == 0:
        return pos_pred.new_zeros(())

    i, j, k, l = quartets[:, 0], quartets[:, 1], quartets[:, 2], quartets[:, 3]

    phi_pred = _dihedral(pos_pred[i], pos_pred[j], pos_pred[k], pos_pred[l])

    with torch.no_grad():
        phi_gt = _dihedral(pos_gt[i], pos_gt[j], pos_gt[k], pos_gt[l])

    return (1.0 - torch.cos(phi_pred - phi_gt)).mean()
