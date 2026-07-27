"""
幾何学的補助損失: 結合角・二面角 MSE

angle_loss   : 3原子パス (i-j-k) の結合角 MSE
dihedral_loss: 4原子パス (i-j-k-l) の二面角 損失 (1 - cos(φ_pred - φ_gt))

build_angle_triplets / build_dihedral_quartets は Pythonループなし・
.item() 呼び出しなし の完全ベクトル化実装。
DataLoader ワーカーで per-molecule 事前計算して collate_fn でオフセット付き
concat することで、学習ループの GPU ホットパスから除去することも可能。

座標損失（回転・並進不変）:
  kabsch_rmsd_loss : 最適回転アライメント後の MSE（Kabsch アルゴリズム）
  dist_matrix_loss : 全原子ペア距離行列の MSE
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter, scatter_mean


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


def wrap_to_pi(angle: Tensor) -> Tensor:
    """
    角度（差）を (-π, π] に折り返す。

    atan2(sin, cos) ベースで分岐なし・勾配安定。剰余演算 (x + π) % 2π - π と
    数値的に等価だが、境界での勾配不連続を避けられる。

    二面角差 φ_pred - φ_gt は [-2π, 2π] の範囲を取りうるため、
    角度距離を正しく評価するには (-π, π] への折り返しが必要。
    """
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def build_angle_triplets(edge_index: Tensor, num_nodes: int) -> Tensor:
    """
    edge_index (2, E) から 3原子トリプレット (i, j, k) を構築。
    j を中心とする i-j-k ペア（i ≠ k）。

    θ(i,j,k) == θ(k,j,i) のため、i < k の片方向のみを残して重複計算を半減
    させている（angle_loss の平均値は不変: 値が等しい重複ペアを除いても
    単純平均・scatter_mean のいずれも同じ平均値になる）。

    完全ベクトル化実装: Python ループなし。
    可変長 arange 生成のため .item() を 1 回使用するが、
    DataLoader ワーカー（CPU）上で呼ぶため GPU-CPU 同期コストなし。

    Returns
    -------
    triplets : (T, 3) int64  [i, j, k]  (i < k の片方向のみ)
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

    # i < k の片方向のみ残す（i == k は既にフィルタ済み。θ(i,j,k)=θ(k,j,i) の
    # 重複を排除し計算量を半減させる）
    keep = i_nodes < k_nodes
    return torch.stack([i_nodes[keep], j_nodes[keep], k_nodes[keep]], dim=1)


def build_dihedral_quartets(edge_index: Tensor, num_nodes: int) -> Tensor:
    """
    edge_index (2, E) から 4原子カルテット (i, j, k, l) を構築。
    j-k を軸とする i-j-k-l。

    完全ベクトル化実装: Python ループなし。
    可変長テンソル生成のため .item() を 2 回使用するが、
    DataLoader ワーカー（CPU）上で呼ぶため GPU-CPU 同期コストなし。

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


def angle_loss(
    pos_pred: Tensor,
    pos_gt: Tensor,
    triplets: Tensor,
    batch: Optional[Tensor] = None,
) -> Tensor:
    """
    Parameters
    ----------
    pos_pred, pos_gt : (N, 3)
    triplets         : (T, 3) int64 [i, j, k]
    batch            : (N,) int  分子インデックス。指定時は「トリプレットごとの
                       二乗誤差 → 中心原子 j の所属分子で scatter_mean → 分子間 mean」
                       の 2 段階正規化を行う（座標損失と同じ正規化方式）。
                       None の場合は従来どおり全トリプレット一括 mean。

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

    sq_err = (theta_pred - theta_gt).pow(2)   # (T,)

    if batch is None:
        return sq_err.mean()

    mol_id = batch[j]                          # トリプレットが属する分子（中心原子 j で決定）
    mol_mse = scatter_mean(sq_err, mol_id, dim=0)
    return mol_mse.mean()


def dihedral_loss(
    pos_pred: Tensor,
    pos_gt: Tensor,
    quartets: Tensor,
    batch: Optional[Tensor] = None,
) -> Tensor:
    """
    Parameters
    ----------
    pos_pred, pos_gt : (N, 3)
    quartets         : (Q, 4) int64 [i, j, k, l]
    batch            : (N,) int  分子インデックス。指定時は「カルテットごとの
                       (1 - cos(Δφ)) → 軸原子 j の所属分子で scatter_mean → 分子間 mean」
                       の 2 段階正規化を行う。None の場合は従来どおり全カルテット一括 mean。

    Returns
    -------
    scalar loss  (1 - cos(Δφ) の平均、Δφ は (-π, π] に折り返した二面角差)

    Notes
    -----
    角度差 Δφ = φ_pred - φ_gt を wrap_to_pi で (-π, π] に折り返してから
    評価する。1 - cos(・) 自体は周期関数のため折り返しても値・勾配は不変だが、
    折り返しにより Δφ が明示的に角度距離となり、他の角度指標（度数 MAE 等）と
    一貫する。1 - cos は [0, 2] に有界なため、大域的な折り畳みのズレで一部の
    二面角が大きく外れても損失が発散せず、局所幾何の学習が阻害されにくい。
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

    delta = wrap_to_pi(phi_pred - phi_gt)   # (-π, π] に折り返し
    elem_loss = 1.0 - torch.cos(delta)      # (Q,)

    if batch is None:
        return elem_loss.mean()

    mol_id = batch[j]                        # カルテットが属する分子（軸原子 j で決定）
    mol_mse = scatter_mean(elem_loss, mol_id, dim=0)
    return mol_mse.mean()


# ── 回転・並進不変な座標損失 ─────────────────────────────────────────────────────

def kabsch_rmsd_loss(pos_pred: Tensor, pos_gt: Tensor, batch: Tensor) -> Tensor:
    """
    Kabsch アルゴリズムによる回転・並進不変な座標損失。

    分子ごとに最適回転行列 R を SVD で求め、pos_pred を R で回転させてから
    pos_gt との MSE を計算する。勾配は pos_pred を通じて流れる。

    R は pos_pred に対する最適回転（argmin）であり、envelope theorem により
    R を detach しても Kabsch 損失の正確な勾配が得られる。SVD 逆伝播の
    特異値縮退による不安定性も回避される。

    scatter + バッチ SVD による完全ベクトル化: Python ループなし。

    Parameters
    ----------
    pos_pred : (N, 3)  予測座標
    pos_gt   : (N, 3)  正解座標
    batch    : (N,) int  バッチ内の分子インデックス（0, 0, ..., 1, 1, ...）

    Returns
    -------
    scalar loss  (分子ごとの MSE の平均)
    """
    # 重心除去（ベクトル化）
    P_mean = scatter_mean(pos_pred, batch, dim=0)   # (B, 3)
    Q_mean = scatter_mean(pos_gt, batch, dim=0)     # (B, 3)
    P = pos_pred - P_mean[batch]                     # (N, 3)
    Q_c = pos_gt - Q_mean[batch]                     # (N, 3)

    # 相関行列 H_m = Σ_{i∈m} P_i ⊗ Q_i を scatter_add で計算
    outer = P.unsqueeze(2) * Q_c.unsqueeze(1)        # (N, 3, 3)
    H = scatter(outer, batch, dim=0, reduce='add')   # (B, 3, 3)

    # SVD は fp32 で実行（AMP bf16/fp16 時の数値安定性確保）
    # 特異値縮退対策として微小摂動を加算
    H_f32 = H.float() + 1e-6 * torch.eye(3, device=H.device).unsqueeze(0)
    U, _S, Vh = torch.linalg.svd(H_f32)

    # reflection 補正 + 最適回転行列
    # autocast スコープ内では matmul が bf16 にキャストされ det が失敗するため
    # 明示的に fp32 を強制する
    with torch.no_grad(), torch.autocast(device_type=H.device.type, enabled=False):
        VhT_UT = Vh.mT @ U.mT                        # (B, 3, 3)  fp32
        d = torch.det(VhT_UT).sign()                  # (B,)
        D = torch.zeros_like(H_f32)
        D[:, 0, 0] = 1.0
        D[:, 1, 1] = 1.0
        D[:, 2, 2] = d
        R = (Vh.mT @ D @ U.mT).to(pos_pred.dtype)    # (B, 3, 3) → 元の dtype

    # 回転適用: P_rot[i] = R[batch[i]] @ P[i]
    P_rot = torch.einsum('nij,nj->ni', R[batch], P)  # (N, 3)

    # 分子ごとの MSE の平均
    per_atom_mse = (P_rot - Q_c).pow(2).mean(dim=-1)     # (N,)
    mol_mse = scatter_mean(per_atom_mse, batch, dim=0)    # (B,)
    return mol_mse.mean()


def _pad_per_molecule(pos_pred: Tensor, pos_gt: Tensor, batch: Tensor):
    """
    (N, 3) の pos_pred/pos_gt を分子ごとに 0-パディングして
    (B, max_n, 3) の密テンソルに変換する。

    `batch.unique()` を用いた Python ループ（分子数 B 回の暗黙的
    GPU→CPU 同期）を避けるため、分子数 B と最大原子数 max_n を得る
    2 回の `.item()` 呼び出しのみに同期を限定し、残りは完全にベクトル化する。

    Returns
    -------
    padded_pred, padded_gt : (B, max_n, 3)
    valid_mask             : (B, max_n) bool  パディングでない実原子位置
    """
    device = pos_pred.device
    N = pos_pred.size(0)

    B = int(batch.max().item()) + 1 if N > 0 else 0          # 同期 1 回
    if B == 0:
        return (
            pos_pred.new_zeros((0, 0, 3)),
            pos_gt.new_zeros((0, 0, 3)),
            torch.zeros((0, 0), dtype=torch.bool, device=device),
        )

    # 分子内でのローカル位置（0..count-1）をベクトル化して求める。
    # batch は必ずしもソート済みでなくても良いよう argsort 経由で計算する。
    order = torch.argsort(batch, stable=True)
    batch_sorted = batch[order]
    counts = torch.bincount(batch_sorted, minlength=B)         # (B,)
    ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])   # (B+1,)
    pos_in_sorted = torch.arange(N, device=device) - torch.repeat_interleave(ptr[:-1], counts)
    local_idx = torch.empty(N, dtype=torch.long, device=device)
    local_idx[order] = pos_in_sorted

    max_n = int(counts.max().item())                            # 同期 1 回

    flat_idx = batch * max_n + local_idx                        # (N,)

    padded_pred = torch.zeros((B * max_n, 3), dtype=pos_pred.dtype, device=device)
    padded_pred = padded_pred.scatter(0, flat_idx.unsqueeze(-1).expand(-1, 3), pos_pred)
    padded_pred = padded_pred.view(B, max_n, 3)

    with torch.no_grad():
        padded_gt = torch.zeros((B * max_n, 3), dtype=pos_gt.dtype, device=device)
        padded_gt = padded_gt.scatter(0, flat_idx.unsqueeze(-1).expand(-1, 3), pos_gt)
        padded_gt = padded_gt.view(B, max_n, 3)

    valid_mask = torch.arange(max_n, device=device).unsqueeze(0) < counts.unsqueeze(1)  # (B, max_n)

    return padded_pred, padded_gt, valid_mask


def dist_matrix_loss(pos_pred: Tensor, pos_gt: Tensor, batch: Tensor) -> Tensor:
    """
    全原子ペア距離行列の MSE による回転・並進不変な座標損失。

    kabsch_rmsd_loss の代替手段として用意。切り替えは vae_loss の
    pos_loss_type 引数で行う。

    計算量は O(n²) / molecule。大きな分子では kabsch_rmsd_loss の方が
    効率的だが、こちらは SVD なしでシンプルに回転不変性を確保できる。

    分子ごとに `batch.unique()` で Python ループする実装は分子数 B 回の
    暗黙的 GPU→CPU 同期を引き起こすため、分子を最大原子数へ 0-パディング
    してから batched cdist で一括計算する（`_pad_per_molecule`）。
    O(Σn)² の全原子ペア cdist（異分子間ペアも含む）を作らず、
    O(B · max_n²) のブロック対角相当の計算に抑える。

    Parameters
    ----------
    pos_pred : (N, 3)  予測座標
    pos_gt   : (N, 3)  正解座標
    batch    : (N,) int  バッチ内の分子インデックス

    Returns
    -------
    scalar loss  (分子ごとの距離行列 MSE の平均)
    """
    if pos_pred.size(0) == 0:
        return pos_pred.new_zeros(())

    device_type = pos_pred.device.type if pos_pred.device.type != 'cpu' else 'cpu'
    padded_pred, padded_gt, valid_mask = _pad_per_molecule(pos_pred, pos_gt, batch)  # (B, max_n, 3), (B, max_n)

    valid_pair = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)   # (B, max_n, max_n)

    # cdist の二乗計算が bf16 で精度劣化するため fp32 を強制
    with torch.autocast(device_type=device_type, enabled=False):
        d_pred = torch.cdist(padded_pred.float(), padded_pred.float())   # (B, max_n, max_n)
        with torch.no_grad():
            d_gt = torch.cdist(padded_gt.float(), padded_gt.float())

        sq_err = (d_pred - d_gt).pow(2) * valid_pair                      # パディング領域は 0
        mol_sum = sq_err.sum(dim=(1, 2))                                  # (B,)
        mol_count = valid_mask.sum(dim=1).float().pow(2).clamp(min=1.0)   # n_m² （元実装の n×n 平均に一致）
        mol_mse = mol_sum / mol_count

    return mol_mse.mean()


def local_distance_loss(
    pos_pred: Tensor,
    pos_gt: Tensor,
    batch: Tensor,
    cutoff: float = 5.0,
    eps: float = 1e-8,
) -> Tensor:
    """
    局所距離損失（大域的な折り畳みに頑健な座標損失）。

    GT で空間的に近い原子ペア（GT 距離 < cutoff Å）に限定して距離 MSE を計算する。
    遠距離ペアを一切ペナルティに含めないため、主鎖二面角のわずかなズレで分子全体の
    折り畳み（大域構造）が変わっても、局所幾何が正しければ損失は小さいままとなる。

    Kabsch RMSD は「1 本の回転可能結合のズレ → 分子の半分が反転 → RMSD 爆発」
    という大域折り畳みへの脆弱性を持つが、本損失はその影響を受けない。局所的な
    結合・角度・近接パッキングの再現性を評価・学習するのに適する。

    分子ごとに `batch.unique()` で Python ループする実装は分子数 B 回の
    暗黙的 GPU→CPU 同期を引き起こすため、分子を最大原子数へ 0-パディング
    してから batched cdist で一括計算する（`_pad_per_molecule`）。
    近接ペアが 1 つも無い分子は元実装同様に平均対象から除外する
    （0 を加算せず、その分子自体をスキップする）。

    Parameters
    ----------
    pos_pred : (N, 3)  予測座標
    pos_gt   : (N, 3)  正解座標
    batch    : (N,) int  バッチ内の分子インデックス
    cutoff   : float  近接ペアとみなす GT 距離の閾値（Å）

    Returns
    -------
    scalar loss  (分子ごとの近接ペア距離 MSE の平均)
    """
    if pos_pred.size(0) == 0:
        return pos_pred.new_zeros(())

    device_type = pos_pred.device.type if pos_pred.device.type != 'cpu' else 'cpu'
    padded_pred, padded_gt, valid_mask = _pad_per_molecule(pos_pred, pos_gt, batch)  # (B, max_n, 3), (B, max_n)

    valid_pair = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)   # (B, max_n, max_n)

    with torch.autocast(device_type=device_type, enabled=False):
        d_pred = torch.cdist(padded_pred.float(), padded_pred.float())   # (B, max_n, max_n)
        with torch.no_grad():
            d_gt = torch.cdist(padded_gt.float(), padded_gt.float())
            # GT で近接する（かつ自己ペアでない、パディングでない）ペアのみを対象
            local = valid_pair & (d_gt < cutoff) & (d_gt > eps)

        sq_err = (d_pred - d_gt).pow(2) * local
        mol_count = local.sum(dim=(1, 2))                 # (B,) 分子ごとの近接ペア数
        has_local = mol_count > 0
        if not has_local.any():
            return pos_pred.new_zeros(())

        mol_sum = sq_err.sum(dim=(1, 2))
        mol_mse = mol_sum[has_local] / mol_count[has_local].float()

    return mol_mse.mean()


def longrange_distance_loss(
    pos_pred: Tensor,
    pos_gt: Tensor,
    dist_mat: Tensor,
    batch: Tensor,
    ptr: Optional[Tensor] = None,
    min_graph_dist: int = 4,
    max_pairs: int = 256,
    huber_delta: float = 1.0,
) -> Tensor:
    """
    大域（long-range）距離損失（滑らかな大域教師信号）。

    グラフ距離（結合ホップ数）が `min_graph_dist` 以上の「遠隔ペア」を対象に、
    予測座標とGT座標のペア間ユークリッド距離の差を Huber (smooth L1) で教師化する。
    距離は回転・並進不変であり、Kabsch RMSD が torsion-flip（1本のねじれのズレで
    分子の半分が反転）で不連続に爆発するのを避け、大域構造へ滑らかな勾配を与える。

    全 (N, N) ペアは使わず、分子ごとに最大 `max_pairs` ペアを **棄却サンプリング**
    で選ぶことで O(B · max_pairs) に計算量を抑える（設計書 §5 の long-range distmat）。
    各分子内でローカル原子インデックスを一様サンプルし、`dist_mat` のブロック対角
    構造から真のグラフ距離を引いて `min_graph_dist` 未満のペアを棄却する。

    分子ごとに正規化（local_distance_loss と同じく分子単位で平均 → 分子間平均）。
    サンプリングは同一分子内のローカルインデックス（共通オフセット `ptr` 由来）で
    行うため、`dist_mat` の分子間 off-block（= far 埋め）を誤って拾うことはない。

    Parameters
    ----------
    pos_pred : (N, 3)  予測座標（勾配はここを通る）
    pos_gt   : (N, 3)  正解座標
    dist_mat : (N, N) int  ブロック対角グラフ距離行列（分子内 0-4, 4=far, clamp 済み）
    batch    : (N,) int  バッチ内の分子インデックス（連続・昇順を仮定; PyG Batch 準拠）
    ptr      : (B+1,) int  分子境界。None の場合は batch から復元する
    min_graph_dist : int  遠隔ペアとみなすグラフ距離の下限（例 4=far）
    max_pairs      : int  1 分子あたりの最大サンプリングペア数
    huber_delta    : float  Huber 損失の遷移点 δ

    Returns
    -------
    scalar loss  (分子ごとの遠隔ペア Huber 距離損失の平均)

    Notes
    -----
    サンプリングには torch のグローバル RNG を用いる。再現性が要る場合は呼び出し側で
    seed を固定すること。eval では seed 固定をしない限りサンプルが毎回ばらつくため、
    値は近傍で揺らぐ（大域傾向の把握には十分だが厳密な決定論ではない点に注意）。
    対象ペアが 1 つも無い（または原子数 < 2 の）分子は 0 寄与でスキップする。
    """
    N = pos_pred.size(0)
    if N == 0:
        return pos_pred.new_zeros(())

    device = pos_pred.device

    # 分子境界 ptr（None なら batch から復元。分子は連続・昇順に並ぶ前提）
    if ptr is None:
        counts = torch.bincount(batch)                              # (B,)
        ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])    # (B+1,)
    else:
        counts = ptr[1:] - ptr[:-1]                                 # (B,)

    B = counts.size(0)
    if B == 0:
        return pos_pred.new_zeros(())

    starts = ptr[:-1]                          # (B,) 各分子の先頭ノード
    n_mol = counts.to(torch.float32)          # (B,) 各分子の原子数

    # ── 分子ごとに max_pairs ペアをローカル一様サンプリング（棄却サンプリング）──
    # ローカルインデックス ia, ib ∈ [0, n_m)。同一分子内なので分子間ペアは生じない。
    rand_a = torch.rand(B, max_pairs, device=device)
    rand_b = torch.rand(B, max_pairs, device=device)
    ia = (rand_a * n_mol.unsqueeze(1)).long()   # floor → [0, n_m-1]
    ib = (rand_b * n_mol.unsqueeze(1)).long()
    # rand が 1.0 に極めて近い場合の丸め対策で上限クランプ（n_m>=1 のときのみ有効）
    max_local = (counts - 1).clamp(min=0).unsqueeze(1)              # (B,1)
    ia = torch.minimum(ia, max_local)
    ib = torch.minimum(ib, max_local)

    # グローバルノードインデックスへ変換
    gi = starts.unsqueeze(1) + ia               # (B, max_pairs)
    gj = starts.unsqueeze(1) + ib
    gi = gi.clamp(max=N - 1)                     # 安全のため範囲内に収める
    gj = gj.clamp(max=N - 1)

    # 真のグラフ距離をブロック対角行列から取得（同一分子内なので off-block を拾わない）
    gd = dist_mat[gi.reshape(-1), gj.reshape(-1)].view(B, max_pairs)

    # 有効ペア: グラフ距離 >= 閾値 かつ 自己ペアでない かつ 原子数 >= 2
    valid = (gd >= min_graph_dist) & (ia != ib) & (counts.unsqueeze(1) >= 2)

    # ── 距離差の Huber（fp32 強制。cdist 同様 bf16/fp16 の精度劣化を避ける）──
    device_type = device.type if device.type != 'cpu' else 'cpu'
    with torch.autocast(device_type=device_type, enabled=False):
        gi_flat = gi.reshape(-1)
        gj_flat = gj.reshape(-1)
        d_pred = (pos_pred.float()[gi_flat] - pos_pred.float()[gj_flat]).norm(dim=-1).view(B, max_pairs)
        with torch.no_grad():
            d_gt = (pos_gt.float()[gi_flat] - pos_gt.float()[gj_flat]).norm(dim=-1).view(B, max_pairs)

        huber = F.huber_loss(d_pred, d_gt, delta=huber_delta, reduction='none')   # (B, max_pairs)
        huber = huber * valid                                                     # 無効ペアは 0

        mol_count = valid.sum(dim=1)                                              # (B,)
        has_pair = mol_count > 0
        if not has_pair.any():
            return pos_pred.new_zeros(())

        mol_sum = huber.sum(dim=1)
        mol_mean = mol_sum[has_pair] / mol_count[has_pair].float()

    return mol_mean.mean()


def clash_loss(
    pos_pred: Tensor,
    dist_mat: Tensor,
    rvdw: Tensor,
    batch: Tensor,
    ptr: Optional[Tensor] = None,
    min_graph_dist: int = 3,
    clash_factor: float = 0.6,
    max_pairs: int = 512,
    exact: bool = False,
) -> Tensor:
    """
    立体衝突（steric clash）ペナルティ = eval の妥当性ゲートの clash 判定を鏡写しにした損失。

    妥当性ゲート（eval_ensemble.validity_of_conformer）は
        「真に非結合なペア（グラフ距離 >= 3）で dist < (rvdw_i + rvdw_j) * clash_factor」
    を 1 つでも持つ配座を fail とする。本損失はその閾値を下回った量（食い込み）だけを
    ヒンジで罰する **ガードレール型** の損失で、閾値以上に離れているペアには一切勾配を
    与えない（GT 参照配座はすべてゲートを通過＝食い込み 0 なので、再構築目標と衝突しない）。

    `longrange_distance_loss` と同じく、全 (N,N) ペアは使わず分子ごとに `max_pairs` ペアを
    棄却サンプリングし、`dist_mat`（ブロック対角・分子内 0-4）からグラフ距離を引いて
    `min_graph_dist` 未満のペア（結合・1-3 幾何隣接）を除外する。分子内ローカル
    インデックスで抽出するため分子間 off-block（far=4）は拾わない。

    Parameters
    ----------
    pos_pred : (N, 3)  予測座標（勾配はここを通る）。pos_gt は使わない（絶対的な立体制約）
    dist_mat : (N, N) int  ブロック対角グラフ距離行列（分子内 0-4, 4=far, clamp 済み）
    rvdw     : (N,) float  原子ごとの van der Waals 半径（Å）。Z→GetRvdw で eval と一致させる
    batch    : (N,) int  バッチ内の分子インデックス（連続・昇順; PyG Batch 準拠）
    ptr      : (B+1,) int  分子境界。None の場合は batch から復元する
    min_graph_dist : int  clash 対象とみなすグラフ距離の下限（eval と合わせ 3）
    clash_factor   : float  閾値係数（eval と合わせ 0.6）
    max_pairs      : int  1 分子あたりの最大サンプリングペア数（exact=False のみ使用）
    exact          : bool  True なら分子ごとの全ユニークペア (i<j) を厳密列挙し、サンプリングを
                     一切行わない（見逃しゼロ）。デフォルト False は従来どおりの棄却サンプリング
                     版で、既存の学習コードパスは一切変更しない完全後方互換。

    Returns
    -------
    scalar loss  (分子ごとの平均 clash エネルギー relu(thr - d)^2 の分子間平均)
                 clash が無ければ 0。対象ペアが無い分子は 0 寄与でスキップ。
    """
    N = pos_pred.size(0)
    if N == 0:
        return pos_pred.new_zeros(())

    device = pos_pred.device

    if ptr is None:
        counts = torch.bincount(batch)                              # (B,)
        ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])    # (B+1,)
    else:
        counts = ptr[1:] - ptr[:-1]                                 # (B,)

    B = counts.size(0)
    if B == 0:
        return pos_pred.new_zeros(())

    starts = ptr[:-1]                          # (B,)

    if exact:
        # ── 厳密モード: 分子ごとに全ユニークペア (i<j) を列挙（サンプリングなし）──
        # 分子ごとに `torch.triu_indices` で局所ペアを作り、分子オフセットを足して
        # グローバルインデックスへ変換する。分子数 B 回のみ Python ループ
        # （後処理・評価用途で低頻度呼び出しのため許容。原子数も数百程度で O(N²) は軽い）。
        gi_list = []
        gj_list = []
        mol_id_list = []
        for m in range(B):
            n_m = int(counts[m].item())
            if n_m < 2:
                continue
            local_i, local_j = torch.triu_indices(n_m, n_m, offset=1, device=device)
            gi_list.append(starts[m] + local_i)
            gj_list.append(starts[m] + local_j)
            mol_id_list.append(torch.full((local_i.size(0),), m, dtype=torch.long, device=device))

        if len(gi_list) == 0:
            return pos_pred.new_zeros(())

        gi_flat = torch.cat(gi_list)
        gj_flat = torch.cat(gj_list)
        mol_id = torch.cat(mol_id_list)

        gd = dist_mat[gi_flat, gj_flat]                     # (P,) 全ペアの真のグラフ距離
        valid = gd >= min_graph_dist                        # (P,) 真に非結合なペアのみ

        device_type = device.type if device.type != 'cpu' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            d_pred = (pos_pred.float()[gi_flat] - pos_pred.float()[gj_flat]).norm(dim=-1)   # (P,)
            rv = rvdw.float()
            with torch.no_grad():
                thr = (rv[gi_flat] + rv[gj_flat]) * clash_factor                             # (P,)

            overlap = (thr - d_pred).clamp(min=0.0)
            pen = (overlap * overlap) * valid                # 無効ペアは 0

            mol_count = scatter(valid.long(), mol_id, dim=0, dim_size=B, reduce='sum')       # (B,)
            has_pair = mol_count > 0
            if not has_pair.any():
                return pos_pred.new_zeros(())

            mol_sum = scatter(pen, mol_id, dim=0, dim_size=B, reduce='sum')                  # (B,)
            mol_mean = mol_sum[has_pair] / mol_count[has_pair].float()

        return mol_mean.mean()

    n_mol = counts.to(torch.float32)          # (B,)

    # 分子ごとに max_pairs ペアをローカル一様サンプリング（棄却サンプリング）
    rand_a = torch.rand(B, max_pairs, device=device)
    rand_b = torch.rand(B, max_pairs, device=device)
    ia = (rand_a * n_mol.unsqueeze(1)).long()
    ib = (rand_b * n_mol.unsqueeze(1)).long()
    max_local = (counts - 1).clamp(min=0).unsqueeze(1)
    ia = torch.minimum(ia, max_local)
    ib = torch.minimum(ib, max_local)

    gi = (starts.unsqueeze(1) + ia).clamp(max=N - 1)   # (B, max_pairs) グローバル index
    gj = (starts.unsqueeze(1) + ib).clamp(max=N - 1)

    # 真のグラフ距離（同一分子内なので off-block を拾わない）
    gd = dist_mat[gi.reshape(-1), gj.reshape(-1)].view(B, max_pairs)

    # 有効ペア: グラフ距離 >= 閾値（真の非結合）かつ 自己ペアでない かつ 原子数 >= 2
    valid = (gd >= min_graph_dist) & (ia != ib) & (counts.unsqueeze(1) >= 2)

    # 距離と閾値（fp32 強制。bf16/fp16 の距離精度劣化を避ける）
    device_type = device.type if device.type != 'cpu' else 'cpu'
    with torch.autocast(device_type=device_type, enabled=False):
        gi_flat = gi.reshape(-1)
        gj_flat = gj.reshape(-1)
        d_pred = (pos_pred.float()[gi_flat] - pos_pred.float()[gj_flat]).norm(dim=-1).view(B, max_pairs)
        # 閾値 thr = (rvdw_i + rvdw_j) * clash_factor（勾配不要の定数）
        rv = rvdw.float()
        with torch.no_grad():
            thr = (rv[gi_flat] + rv[gj_flat]).view(B, max_pairs) * clash_factor

        # ヒンジ: 閾値を下回った食い込み量のみ罰する（それ以外は 0 勾配）
        overlap = (thr - d_pred).clamp(min=0.0)          # (B, max_pairs)
        pen = (overlap * overlap) * valid                # 無効ペアは 0

        mol_count = valid.sum(dim=1)                      # (B,) 有効ペア数
        has_pair = mol_count > 0
        if not has_pair.any():
            return pos_pred.new_zeros(())

        mol_mean = pen.sum(dim=1)[has_pair] / mol_count[has_pair].float()

    return mol_mean.mean()


def bond_range_loss(
    pos_pred: Tensor,
    edge_index: Tensor,
    batch: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    bond_lo: float = 0.7,
    bond_hi: float = 2.6,
) -> Tensor:
    """
    結合長レンジ・ヒンジ損失 = eval の妥当性ゲートの結合長サニティ判定を鏡写しにした
    **GT 不要** のガードレール損失。

    妥当性ゲート（eval_ensemble.validity_of_conformer）は結合原子ペア距離が
    [bond_lo, bond_hi] Å の範囲を外れる配座を fail とする。本損失はその範囲を
    外れた量だけを両側ヒンジで罰し、範囲内のペアには一切勾配を与えない
    （clash_loss と対をなす、絶対的な幾何制約のガードレール）。

    clash_loss と同じく fp32 を強制する（bf16/fp16 での距離精度劣化を避ける）。

    Parameters
    ----------
    pos_pred   : (N, 3)  予測座標（勾配はここを通る）。GT は使わない
    edge_index : (2, E) int  結合エッジ（有向両方向でも可。対称なヒンジのため問題ない）
    batch      : (N,) int  バッチ内の分子インデックス。指定時は「エッジごとの罰則 →
                 エッジ端点（src）の所属分子で scatter_mean → 分子間 mean」の
                 2 段階正規化を行う（bond_length_loss と同じ正規化方式）。
                 None の場合は全エッジ一括 mean。
    ptr        : 未使用（他のガードレール損失とのシグネチャ整合のために受理するのみ）
    bond_lo    : float  結合長下限（Å）。eval と合わせ 0.7
    bond_hi    : float  結合長上限（Å）。eval と合わせ 2.6

    Returns
    -------
    scalar loss  (relu(bond_lo - d)^2 + relu(d - bond_hi)^2 の平均)
                 全結合が範囲内なら 0。
    """
    del ptr  # シグネチャ整合のためのみ受理（現状未使用）

    if edge_index.size(1) == 0:
        return pos_pred.new_zeros(())

    src, dst = edge_index

    device_type = pos_pred.device.type if pos_pred.device.type != 'cpu' else 'cpu'
    with torch.autocast(device_type=device_type, enabled=False):
        d = (pos_pred.float()[src] - pos_pred.float()[dst]).norm(dim=-1)   # (E,)

        under = (bond_lo - d).clamp(min=0.0)
        over = (d - bond_hi).clamp(min=0.0)
        pen = under * under + over * over            # (E,)

        if batch is None:
            return pen.mean()

        mol_id = batch[src]
        mol_mean = scatter_mean(pen, mol_id, dim=0)
        return mol_mean.mean()
