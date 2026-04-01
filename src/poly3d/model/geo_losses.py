"""
幾何学的補助損失: 結合角・二面角 MSE

angle_loss   : 3原子パス (i-j-k) の結合角 MSE
dihedral_loss: 4原子パス (i-j-k-l) の二面角 損失 (1 - cos(φ_pred - φ_gt))

build_angle_triplets / build_dihedral_quartets は Pythonループなし・
.item() 呼び出しなし の完全ベクトル化実装。
DataLoader ワーカーで per-molecule 事前計算して collate_fn でオフセット付き
concat することで、学習ループの GPU ホットパスから除去することも可能。
"""
from __future__ import annotations

import torch
from torch import Tensor


def _angle_between(v1: Tensor, v2: Tensor, eps: float = 1e-8) -> Tensor:
    """v1, v2: (..., 3) → (...,) ラジアン
    atan2 ベース（acos より勾配安定）。
    """
    n1 = v1.norm(dim=-1, keepdim=True).clamp(min=eps)
    n2 = v2.norm(dim=-1, keepdim=True).clamp(min=eps)
    u1 = v1 / n1
    u2 = v2 / n2
    cross = torch.cross(u1, u2, dim=-1).norm(dim=-1).clamp(min=0.0)
    dot = (u1 * u2).sum(dim=-1)
    return torch.atan2(cross, dot)


def _dihedral(p0: Tensor, p1: Tensor, p2: Tensor, p3: Tensor, eps: float = 1e-8) -> Tensor:
    """4点の二面角。(..., 3) → (...,) ラジアン。atan2 ベースで勾配安定。"""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2

    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)

    b2_norm = b2 / b2.norm(dim=-1, keepdim=True).clamp(min=eps)
    m1 = torch.cross(n1, b2_norm, dim=-1)

    y = (m1 * n2).sum(dim=-1)
    x = (n1 * n2).sum(dim=-1)
    return torch.atan2(y, x)


def build_angle_triplets(edge_index: Tensor, num_nodes: int) -> Tensor:
    """
    edge_index (2, E) から 3原子トリプレット (i, j, k) を構築。
    j を中心とする i-j-k ペア（i ≠ k）。

    完全ベクトル化実装: Python ループなし、.item() 呼び出しなし。
    バッチ全体で一度に処理するため GPU-CPU 同期なし。

    Returns
    -------
    triplets : (T, 3) int64  [i, j, k]
    """
    src, dst = edge_index
    device = edge_index.device

    if src.size(0) == 0:
        return torch.zeros((0, 3), dtype=torch.long, device=device)

    # dst でソートしてグループ化
    order = dst.argsort(stable=True)
    src_s = src[order]
    dst_s = dst[order]

    # 各 dst 値（= j）のグループサイズを取得
    _, counts = torch.unique_consecutive(dst_s, return_counts=True)
    ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])   # (G+1,)

    # c >= 2 のグループのみ対象
    valid = counts >= 2
    if not valid.any():
        return torch.zeros((0, 3), dtype=torch.long, device=device)

    v_counts = counts[valid]                # (G',)
    v_starts = ptr[:-1][valid]             # グループ先頭位置
    v_j      = dst_s[v_starts]            # j 値
    G        = v_counts.size(0)

    # 各グループで c² ペアを生成 → 対角（a==b）除去で c*(c-1) ペアを得る
    n_sq   = v_counts * v_counts
    n_pairs = v_counts * (v_counts - 1)
    total_sq = n_sq.sum()

    if total_sq == 0:
        return torch.zeros((0, 3), dtype=torch.long, device=device)

    # 各 c² ペアのグループインデックスと親グループの c
    g_idx = torch.repeat_interleave(torch.arange(G, device=device), n_sq)
    c_g   = torch.repeat_interleave(v_counts, n_sq)

    # グループ内ローカル通し番号 (0..c²-1)
    sq_starts = torch.cat([n_sq.new_zeros(1), n_sq.cumsum(0)[:-1]])
    local = torch.arange(total_sq, device=device) - torch.repeat_interleave(sq_starts, n_sq)

    local_a = local // c_g    # 行 (0..c-1)
    local_b = local %  c_g    # 列 (0..c-1)

    # 対角フィルタ（i ≠ k）
    mask    = local_a != local_b
    g_pair  = g_idx[mask]
    la      = local_a[mask]
    lb      = local_b[mask]

    # グローバルエッジインデックスへ変換
    g_start = torch.repeat_interleave(v_starts, n_pairs)
    i_nodes = src_s[g_start + la]
    j_nodes = v_j[g_pair]
    k_nodes = src_s[g_start + lb]

    return torch.stack([i_nodes, j_nodes, k_nodes], dim=1)


def build_dihedral_quartets(edge_index: Tensor, num_nodes: int) -> Tensor:
    """
    edge_index (2, E) から 4原子カルテット (i, j, k, l) を構築。
    j-k を軸とする i-j-k-l。

    完全ベクトル化実装: Python ループなし、.item() 呼び出しなし。

    Phase 1: 各無向エッジ (j,k) に対して i ∈ N(j)\\{k} を展開
    Phase 2: 各 (e,i) ペアに対して l ∈ N(k)\\{j,i} を展開

    Returns
    -------
    quartets : (Q, 4) int64  [i, j, k, l]
    """
    src, dst = edge_index
    device = edge_index.device

    if src.size(0) == 0:
        return torch.zeros((0, 4), dtype=torch.long, device=device)

    # CSR 隣接テーブルをベクトル演算で構築
    order = src.argsort(stable=True)
    src_s = src[order]
    dst_s = dst[order]

    deg = torch.zeros(num_nodes, dtype=torch.long, device=device)
    deg.scatter_add_(0, src, torch.ones(src.size(0), dtype=torch.long, device=device))
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(0)])   # (num_nodes+1,)

    # 無向エッジ (j < k) のみ
    jk_mask = src < dst
    j_vec = src[jk_mask]   # (E',)
    k_vec = dst[jk_mask]
    E_prime = j_vec.size(0)
    if E_prime == 0:
        return torch.zeros((0, 4), dtype=torch.long, device=device)

    # ── Phase 1: i ∈ N(j)\{k} の展開 ────────────────────────────────────────
    j_start = ptr[j_vec]
    j_deg   = deg[j_vec]   # j の次数（k を含む）

    total_i_raw = j_deg.sum().item()
    if total_i_raw == 0:
        return torch.zeros((0, 4), dtype=torch.long, device=device)

    e_for_i_raw = torch.repeat_interleave(torch.arange(E_prime, device=device), j_deg)

    # j の全隣接ノードを展開（セグメント arange パターン）
    j_cs = torch.cat([j_deg.new_zeros(1), j_deg.cumsum(0)[:-1]])
    local_i = torch.arange(total_i_raw, device=device) - torch.repeat_interleave(j_cs, j_deg)
    i_raw = dst_s[torch.repeat_interleave(j_start, j_deg) + local_i]

    # k を除外
    k_for_i_raw = k_vec[e_for_i_raw]
    keep_i = i_raw != k_for_i_raw
    e_for_i = e_for_i_raw[keep_i]
    i_valid = i_raw[keep_i]

    if i_valid.size(0) == 0:
        return torch.zeros((0, 4), dtype=torch.long, device=device)

    # ── Phase 2: l ∈ N(k)\{j,i} の展開 ──────────────────────────────────────
    k_for_ei = k_vec[e_for_i]
    j_for_ei = j_vec[e_for_i]

    k_start_ei = ptr[k_for_ei]
    k_deg_ei   = deg[k_for_ei]

    total_l = k_deg_ei.sum().item()
    if total_l == 0:
        return torch.zeros((0, 4), dtype=torch.long, device=device)

    n_ei     = i_valid.size(0)
    ei_for_l = torch.repeat_interleave(torch.arange(n_ei, device=device), k_deg_ei)

    # k の全隣接ノードを展開
    k_cs     = torch.cat([k_deg_ei.new_zeros(1), k_deg_ei.cumsum(0)[:-1]])
    local_l  = torch.arange(total_l, device=device) - torch.repeat_interleave(k_cs, k_deg_ei)
    l_raw    = dst_s[torch.repeat_interleave(k_start_ei, k_deg_ei) + local_l]

    j_for_l  = j_for_ei[ei_for_l]
    i_for_l  = i_valid[ei_for_l]
    k_for_l  = k_for_ei[ei_for_l]

    # j と i を除外
    keep_l = (l_raw != j_for_l) & (l_raw != i_for_l)

    if not keep_l.any():
        return torch.zeros((0, 4), dtype=torch.long, device=device)

    return torch.stack([
        i_for_l[keep_l],
        j_for_l[keep_l],
        k_for_l[keep_l],
        l_raw[keep_l],
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
