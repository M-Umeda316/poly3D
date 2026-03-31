"""
幾何学的補助損失: 結合角・二面角 MSE

angle_loss   : 3原子パス (i-j-k) の結合角 MSE
dihedral_loss: 4原子パス (i-j-k-l) の二面角 損失 (1 - cos(φ_pred - φ_gt))
"""
from __future__ import annotations

import torch
from torch import Tensor


def _angle_between(v1: Tensor, v2: Tensor, eps: float = 1e-8) -> Tensor:
    """v1, v2: (..., 3) → (...,) ラジアン
    acos の代わりに atan2 を使用（勾配が ±1 で発散しない）。
    """
    n1 = v1.norm(dim=-1, keepdim=True).clamp(min=eps)
    n2 = v2.norm(dim=-1, keepdim=True).clamp(min=eps)
    u1 = v1 / n1
    u2 = v2 / n2
    cross = torch.cross(u1, u2, dim=-1).norm(dim=-1).clamp(min=0.0)
    dot = (u1 * u2).sum(dim=-1)
    return torch.atan2(cross, dot)


def _dihedral(p0: Tensor, p1: Tensor, p2: Tensor, p3: Tensor, eps: float = 1e-8) -> Tensor:
    """
    4点の二面角を計算。(..., 3) → (...,) ラジアン
    atan2 ベースで数値安定（acos/cross-product 正規化の勾配爆発なし）。
    """
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2

    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)

    # b2 正規化ベクトルを使って atan2 の y 成分を計算
    b2_norm = b2 / b2.norm(dim=-1, keepdim=True).clamp(min=eps)
    m1 = torch.cross(n1, b2_norm, dim=-1)

    y = (m1 * n2).sum(dim=-1)
    x = (n1 * n2).sum(dim=-1)
    return torch.atan2(y, x)


def build_angle_triplets(edge_index: Tensor, num_nodes: int) -> Tensor:
    """
    edge_index (2, E) から 3原子トリプレット (i, j, k) を構築。
    j を中心とする i-j-k ペア（i ≠ k）。

    GPU ベクトル演算で実装。dst 値でグループ化し、
    同じ j を持つエッジのペアを (i, j, k) として取得する。

    Returns
    -------
    triplets : (T, 3) int64  [i, j, k]
    """
    src, dst = edge_index   # src→dst の有向エッジ
    device = edge_index.device

    if src.size(0) == 0:
        return torch.zeros((0, 3), dtype=torch.long, device=device)

    # dst でソートして、同じ j（=dst）を持つエッジのペアを組む
    order = dst.argsort(stable=True)
    src_s = src[order]   # j に入る src ノード（= i 候補）
    dst_s = dst[order]   # = j

    # 同じ j を持つ連続区間を境界で分割
    # boundary: dst_s[e] != dst_s[e-1] となる位置
    boundary = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device),
        (dst_s[1:] != dst_s[:-1]).nonzero(as_tuple=True)[0] + 1,
        torch.tensor([src_s.size(0)], dtype=torch.long, device=device),
    ])

    i_list, j_list, k_list = [], [], []
    for seg in range(boundary.size(0) - 1):
        s, e = boundary[seg].item(), boundary[seg + 1].item()
        if e - s < 2:
            continue
        nbrs = src_s[s:e]          # j の隣接ノード群
        j_val = dst_s[s].item()
        # 全ペア (i, k) で i ≠ k
        n = nbrs.size(0)
        idx = torch.arange(n, device=device)
        ii, kk = torch.meshgrid(idx, idx, indexing='ij')
        mask = ii != kk
        i_nodes = nbrs[ii[mask]]
        k_nodes = nbrs[kk[mask]]
        j_nodes = torch.full((mask.sum(),), j_val, dtype=torch.long, device=device)
        i_list.append(i_nodes)
        j_list.append(j_nodes)
        k_list.append(k_nodes)

    if not i_list:
        return torch.zeros((0, 3), dtype=torch.long, device=device)
    return torch.stack([
        torch.cat(i_list), torch.cat(j_list), torch.cat(k_list)
    ], dim=1)


def build_dihedral_quartets(edge_index: Tensor, num_nodes: int) -> Tensor:
    """
    edge_index (2, E) から 4原子カルテット (i, j, k, l) を構築。
    j-k を軸とする i-j-k-l。

    GPU ベクトル演算で実装。
    各無向エッジ (j,k) について i∈N(j)\\{k}、l∈N(k)\\{j,i} を列挙。

    Returns
    -------
    quartets : (Q, 4) int64  [i, j, k, l]
    """
    src, dst = edge_index
    device = edge_index.device

    if src.size(0) == 0:
        return torch.zeros((0, 4), dtype=torch.long, device=device)

    # 無向エッジ (j<k) のみ処理
    mask_jk = src < dst
    j_edges = src[mask_jk]   # (E',)
    k_edges = dst[mask_jk]   # (E',)

    # src でソートして隣接テーブルを高速参照
    order = src.argsort(stable=True)
    src_s = src[order]
    dst_s = dst[order]
    boundary = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device),
        (src_s[1:] != src_s[:-1]).nonzero(as_tuple=True)[0] + 1,
        torch.tensor([src_s.size(0)], dtype=torch.long, device=device),
    ])
    # node → (start, end) in sorted array
    node_start = torch.zeros(num_nodes, dtype=torch.long, device=device)
    node_end = torch.zeros(num_nodes, dtype=torch.long, device=device)
    for seg in range(boundary.size(0) - 1):
        s, e = boundary[seg].item(), boundary[seg + 1].item()
        node_val = src_s[s].item()
        node_start[node_val] = s
        node_end[node_val] = e

    i_list, j_list, k_list, l_list = [], [], [], []
    for idx in range(j_edges.size(0)):
        j = j_edges[idx].item()
        k = k_edges[idx].item()

        # i ∈ N(j) \ {k}
        nbrs_j = dst_s[node_start[j]:node_end[j]]
        i_nodes = nbrs_j[nbrs_j != k]
        if i_nodes.size(0) == 0:
            continue

        # l ∈ N(k) \ {j}
        nbrs_k = dst_s[node_start[k]:node_end[k]]
        l_nodes = nbrs_k[nbrs_k != j]
        if l_nodes.size(0) == 0:
            continue

        # 全組み合わせ (i, l)、ただし l ≠ i
        ii, ll = torch.meshgrid(
            torch.arange(i_nodes.size(0), device=device),
            torch.arange(l_nodes.size(0), device=device),
            indexing='ij',
        )
        i_exp = i_nodes[ii.flatten()]
        l_exp = l_nodes[ll.flatten()]
        mask = i_exp != l_exp
        i_exp, l_exp = i_exp[mask], l_exp[mask]
        if i_exp.size(0) == 0:
            continue

        n = i_exp.size(0)
        i_list.append(i_exp)
        j_list.append(torch.full((n,), j, dtype=torch.long, device=device))
        k_list.append(torch.full((n,), k, dtype=torch.long, device=device))
        l_list.append(l_exp)

    if not i_list:
        return torch.zeros((0, 4), dtype=torch.long, device=device)
    return torch.stack([
        torch.cat(i_list), torch.cat(j_list),
        torch.cat(k_list), torch.cat(l_list),
    ], dim=1)


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
    theta_pred = torch.nan_to_num(_angle_between(v1_pred, v2_pred), nan=0.0)

    with torch.no_grad():
        v1_gt = pos_gt[i] - pos_gt[j]
        v2_gt = pos_gt[k] - pos_gt[j]
        theta_gt = torch.nan_to_num(_angle_between(v1_gt, v2_gt), nan=0.0)

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

    phi_pred = torch.nan_to_num(
        _dihedral(pos_pred[i], pos_pred[j], pos_pred[k], pos_pred[l]), nan=0.0
    )

    with torch.no_grad():
        phi_gt = torch.nan_to_num(
            _dihedral(pos_gt[i], pos_gt[j], pos_gt[k], pos_gt[l]), nan=0.0
        )

    return (1.0 - torch.cos(phi_pred - phi_gt)).mean()
