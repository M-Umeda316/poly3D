"""
VAE 学習損失

L = w_pos * Lpos + w_bond * Lbond + w_angle * Langle + w_dih * Ldihedral + beta * Lkl

各損失:
  Lpos      : 回転・並進不変な座標損失（pos_loss_type で切り替え可）
                'kabsch'  : Kabsch RMSD（デフォルト）
                'distmat' : 全原子ペア距離行列 MSE
  Lbond     : 結合長 MSE
  Langle    : 結合角 MSE
  Ldihedral : 二面角損失 (1 - cos(φ_pred - φ_gt))
  Lkl       : KL(q(Z|X) || N(0,I))

beta は学習初期に 0 から 1 に warm-up する (KL annealing)。
"""
from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_mean

from poly3d.model.geo_losses import (
    angle_loss, dihedral_loss,
    build_angle_triplets, build_dihedral_quartets,
    kabsch_rmsd_loss, dist_matrix_loss, local_distance_loss,
    longrange_distance_loss, clash_loss,
)


def bond_length_loss(
    pos_pred: Tensor,
    pos_gt: Tensor,
    edge_index: Tensor,
    batch: Optional[Tensor] = None,
) -> Tensor:
    """
    結合で繋がった原子ペアの距離 MSE。

    batch を指定すると、「エッジごとの二乗誤差 → エッジ端点（src）の
    所属分子で scatter_mean → 分子間 mean」の 2 段階正規化を行う
    （座標損失と同じ正規化方式）。エッジは分子内のみに存在するため
    src / dst どちらの batch を使っても結果は同じ。
    batch が None の場合は従来どおり全エッジ一括 mean（後方互換）。
    """
    if edge_index.size(1) == 0:
        return pos_pred.new_zeros(())
    src, dst = edge_index
    d_pred = (pos_pred[src] - pos_pred[dst]).norm(dim=-1)
    with torch.no_grad():
        d_gt = (pos_gt[src] - pos_gt[dst]).norm(dim=-1)

    sq_err = (d_pred - d_gt).pow(2)   # (E,)

    if batch is None:
        return sq_err.mean()

    mol_id = batch[src]
    mol_mse = scatter_mean(sq_err, mol_id, dim=0)
    return mol_mse.mean()


def vae_loss(
    pos_pred: Tensor,
    pos_gt: Tensor,
    mu: Tensor,
    logvar: Tensor,
    edge_index: Tensor,
    num_nodes: int,
    beta: float = 1.0,
    w_pos: float = 1.0,
    w_bond: float = 1.0,
    w_angle: float = 0.5,
    w_dihedral: float = 0.1,
    # 事前計算済みトポロジー（None の場合はここで計算）
    triplets: Optional[Tensor] = None,
    quartets: Optional[Tensor] = None,
    # 座標損失の種類:
    #   'kabsch'        : Kabsch RMSD（回転不変、大域構造も評価）
    #   'distmat'       : 全ペア距離行列 MSE（回転不変）
    #   'local_distmat' : 近接ペア（GT < local_cutoff Å）のみの距離 MSE
    #                     大域的な折り畳みに頑健
    #   'multiscale_distmat' : local_distmat + long-range distmat（大域の滑らかな教師信号）
    #                     L_pos = w_local * local + w_global * longrange
    pos_loss_type: Literal['kabsch', 'distmat', 'local_distmat', 'multiscale_distmat'] = 'kabsch',
    local_cutoff: float = 5.0,
    batch: Optional[Tensor] = None,
    # multiscale_distmat 用（他の pos_loss_type では未使用）
    dist_mat: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    w_local: float = 1.0,
    w_global: float = 1.0,
    longrange_min_graph_dist: int = 4,
    longrange_max_pairs: int = 256,
    longrange_huber_delta: float = 1.0,
    # clash（立体衝突）ガードレール損失。w_clash=0 で完全無効（後方互換）
    w_clash: float = 0.0,
    rvdw: Optional[Tensor] = None,
    clash_factor: float = 0.6,
    clash_min_graph_dist: int = 3,
    clash_max_pairs: int = 512,
) -> Tuple[Tensor, dict]:
    """
    Parameters
    ----------
    pos_pred      : (N, 3) 予測座標
    pos_gt        : (N, 3) 正解座標
    mu, logvar    : (N, latent_dim)
    edge_index    : (2, E)
    num_nodes     : N
    beta          : KL 損失の重み（warm-up で変化させる）
    pos_loss_type : 座標損失の種類（'kabsch' or 'distmat'）
    batch         : (N,) 分子インデックス。None の場合は単一分子として扱う

    Returns
    -------
    total_loss : scalar
    loss_dict  : 各損失の値（float）
    """
    # batch が None の場合は全原子を1分子として扱う
    if batch is None:
        batch = pos_pred.new_zeros(num_nodes, dtype=torch.long)

    # 座標損失（回転・並進不変）
    # multiscale の内訳ログ用（該当時のみ埋める）
    l_pos_local: Optional[Tensor] = None
    l_pos_global: Optional[Tensor] = None
    if pos_loss_type == 'distmat':
        l_pos = dist_matrix_loss(pos_pred, pos_gt, batch)
    elif pos_loss_type == 'local_distmat':
        l_pos = local_distance_loss(pos_pred, pos_gt, batch, cutoff=local_cutoff)
    elif pos_loss_type == 'multiscale_distmat':
        # 局所（近接ペア距離）＋ 大域（遠隔ペア距離 Huber, サンプリング版）
        l_pos_local = local_distance_loss(pos_pred, pos_gt, batch, cutoff=local_cutoff)
        if dist_mat is not None:
            l_pos_global = longrange_distance_loss(
                pos_pred, pos_gt, dist_mat, batch, ptr=ptr,
                min_graph_dist=longrange_min_graph_dist,
                max_pairs=longrange_max_pairs,
                huber_delta=longrange_huber_delta,
            )
        else:
            # dist_mat 未供給時は大域項をスキップ（局所のみで後方互換的に動作）
            l_pos_global = pos_pred.new_zeros(())
        l_pos = w_local * l_pos_local + w_global * l_pos_global
    else:
        l_pos = kabsch_rmsd_loss(pos_pred, pos_gt, batch)

    # 結合長
    l_bond = bond_length_loss(pos_pred, pos_gt, edge_index, batch)

    # 結合角
    if triplets is None:
        triplets = build_angle_triplets(edge_index, num_nodes)
    l_angle = angle_loss(pos_pred, pos_gt, triplets, batch)

    # 二面角
    if quartets is None:
        quartets = build_dihedral_quartets(edge_index, num_nodes)
    l_dihedral = dihedral_loss(pos_pred, pos_gt, quartets, batch)

    # KL（logvar は VAEEncoder で clamp 済みだが念のため再 clamp）
    logvar_safe = logvar.clamp(-10.0, 10.0)
    l_kl = (-0.5 * (1 + logvar_safe - mu.pow(2) - logvar_safe.exp())).mean()

    # clash（立体衝突）ガードレール: グラフ距離>=3 のペアが vdW 閾値に食い込んだ量を罰する。
    # w_clash=0 / dist_mat 無 / rvdw 無 のいずれかで無効（後方互換）。
    l_clash: Optional[Tensor] = None
    if w_clash > 0.0 and dist_mat is not None and rvdw is not None:
        l_clash = clash_loss(
            pos_pred, dist_mat, rvdw, batch, ptr=ptr,
            min_graph_dist=clash_min_graph_dist,
            clash_factor=clash_factor,
            max_pairs=clash_max_pairs,
        )

    total = w_pos * l_pos + w_bond * l_bond + w_angle * l_angle + w_dihedral * l_dihedral
    if l_clash is not None:
        total = total + w_clash * l_clash
    # beta=0 のとき 0*NaN=NaN になるのを避けるため明示的にガード
    if beta > 0.0:
        total = total + beta * l_kl

    # detach のみ（.item() は呼ばない → GPU-CPU sync を回避）
    # 呼び出し側で必要時に .item() を呼ぶ
    loss_dict = {
        'total': total.detach(),
        'pos': l_pos.detach(),
        'bond': l_bond.detach(),
        'angle': l_angle.detach(),
        'dihedral': l_dihedral.detach(),
        'kl': l_kl.detach(),
    }
    # multiscale 経路では内訳（局所・大域）も記録
    if l_pos_local is not None:
        loss_dict['pos_local'] = l_pos_local.detach()
        loss_dict['pos_global'] = l_pos_global.detach()
    if l_clash is not None:
        loss_dict['clash'] = l_clash.detach()
    return total, loss_dict
