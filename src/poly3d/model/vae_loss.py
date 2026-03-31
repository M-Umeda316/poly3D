"""
VAE 学習損失

L = w_pos * Lpos + w_bond * Lbond + w_angle * Langle + w_dih * Ldihedral + beta * Lkl

各損失:
  Lpos      : 座標 MSE
  Lbond     : 結合長 MSE
  Langle    : 結合角 MSE
  Ldihedral : 二面角損失 (1 - cos(φ_pred - φ_gt))
  Lkl       : KL(q(Z|X) || N(0,I))

beta は学習初期に 0 から 1 に warm-up する (KL annealing)。
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from poly3d.model.geo_losses import (
    angle_loss, dihedral_loss,
    build_angle_triplets, build_dihedral_quartets,
)


def bond_length_loss(pos_pred: Tensor, pos_gt: Tensor, edge_index: Tensor) -> Tensor:
    """結合で繋がった原子ペアの距離 MSE"""
    if edge_index.size(1) == 0:
        return pos_pred.new_zeros(())
    src, dst = edge_index
    d_pred = (pos_pred[src] - pos_pred[dst]).norm(dim=-1)
    with torch.no_grad():
        d_gt = (pos_gt[src] - pos_gt[dst]).norm(dim=-1)
    return F.mse_loss(d_pred, d_gt)


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
) -> Tuple[Tensor, dict]:
    """
    Parameters
    ----------
    pos_pred  : (N, 3) 予測座標
    pos_gt    : (N, 3) 正解座標
    mu, logvar: (N, latent_dim)
    edge_index: (2, E)
    num_nodes : N
    beta      : KL 損失の重み（warm-up で変化させる）

    Returns
    -------
    total_loss : scalar
    loss_dict  : 各損失の値（float）
    """
    # 座標 MSE
    l_pos = F.mse_loss(pos_pred, pos_gt)

    # 結合長
    l_bond = bond_length_loss(pos_pred, pos_gt, edge_index)

    # 結合角
    if triplets is None:
        triplets = build_angle_triplets(edge_index, num_nodes)
    l_angle = angle_loss(pos_pred, pos_gt, triplets)

    # 二面角
    if quartets is None:
        quartets = build_dihedral_quartets(edge_index, num_nodes)
    l_dihedral = dihedral_loss(pos_pred, pos_gt, quartets)

    # KL（logvar は VAEEncoder で clamp 済みだが念のため再 clamp）
    logvar_safe = logvar.clamp(-10.0, 10.0)
    l_kl = (-0.5 * (1 + logvar_safe - mu.pow(2) - logvar_safe.exp())).mean()

    total = w_pos * l_pos + w_bond * l_bond + w_angle * l_angle + w_dihedral * l_dihedral
    # beta=0 のとき 0*NaN=NaN になるのを避けるため明示的にガード
    if beta > 0.0:
        total = total + beta * l_kl

    loss_dict = {
        'total': total.item(),
        'pos': l_pos.item(),
        'bond': l_bond.item(),
        'angle': l_angle.item(),
        'dihedral': l_dihedral.item(),
        'kl': l_kl.item(),
    }
    return total, loss_dict
