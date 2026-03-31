"""
Flow Matching スケジューラ + 損失

論文: Gaussian flow matching
  Zt = (1-t) * Z0 + t * Z1
  velocity target: v* = Z1 - Z0
  loss: 1/(1-t)^2 * ||Z1 - Z1_pred||^2  (Z1 を予測するパラメータ化)

  t ~ U(0, t_max)  t_max=0.9 で高ノイズ端のサンプルを切る。

サンプリング: Euler ODE
  dZ/dt = (Z1_pred - Zt) / (1 - t)
  Zt-dt = Zt - dt * (Zt - Z1_pred) / (1 - t)

パフォーマンスメモ:
  attn_bias (グラフ距離 BFS) は loss() / sample() の先頭で 1 回だけ計算し、
  複数の forward 呼び出しに使い回す。
    - loss(): self-cond 2 パス → 1 回の BFS に削減
    - sample(): n_steps ステップ → 1 回の BFS に削減
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def _unwrap_model(model: nn.Module) -> nn.Module:
    """DDP ラップを外して元のモジュールを返す。"""
    return model.module if hasattr(model, 'module') else model


class FlowMatching(nn.Module):
    """
    Flow matching の学習・サンプリング用ラッパー。

    Parameters
    ----------
    model      : LatentDiT  (zt, cond, t, batch, ...) → z1_pred
    t_max      : t のサンプリング上限（論文: 0.9）
    p_selfcond : self-conditioning 確率
    """

    def __init__(
        self,
        model: nn.Module,
        t_max: float = 0.9,
        p_selfcond: float = 0.5,
    ):
        super().__init__()
        self.model = model
        self.t_max = t_max
        self.p_selfcond = p_selfcond

    def loss(
        self,
        z0: Tensor,          # (N, latent_dim)  VAE encoder 出力 (mu)
        cond: Tensor,        # (N, cond_dim)    ConditionalEncoder Ci
        batch: Tensor,       # (N,) int
        edge_index: Optional[Tensor] = None,  # (2, E)
    ) -> Tuple[Tensor, dict]:
        """
        Returns
        -------
        loss     : scalar
        loss_dict: {'flow': float}
        """
        B = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        N = z0.size(0)
        device = z0.device

        # t ~ U(0, t_max) を分子ごとにサンプル
        t = torch.rand(B, device=device) * self.t_max   # (B,)

        # Z1 ~ N(0, I)
        z1 = torch.randn_like(z0)

        # Zt = (1-t) * Z0 + t * Z1
        t_node = t[batch]
        zt = (1.0 - t_node.unsqueeze(-1)) * z0 + t_node.unsqueeze(-1) * z1

        # BFS・マスク計算を 1 回だけ実行（DDP 外のローカルモジュールで計算）
        raw = _unwrap_model(self.model)
        attn_bias = raw.precompute_attn_inputs(edge_index, batch, N)

        # Self-conditioning: no_grad パス
        # DDP を介さず raw module を直接呼ぶことで不要な gradient sync を回避
        if self.training and torch.rand(1).item() < self.p_selfcond:
            with torch.no_grad():
                z_sc = raw(zt, cond, t, batch, z_sc=None, attn_bias=attn_bias).detach()
        else:
            z_sc = None

        # 本番パス: self.model 経由で呼ぶことで DDP gradient sync が正しく発動
        z1_pred = self.model(zt, cond, t, batch, z_sc=z_sc, attn_bias=attn_bias)

        # 損失: 1/(1-t)^2 * ||Z1 - Z1_pred||^2 の平均
        weight = 1.0 / (1.0 - t_node).pow(2).clamp(min=1e-4)
        flow_loss = (weight * (z1 - z1_pred).pow(2).mean(dim=-1)).mean()

        return flow_loss, {'flow': flow_loss.item()}

    @torch.no_grad()
    def sample(
        self,
        n_atoms: int,
        cond: Tensor,        # (N, cond_dim)
        batch: Tensor,       # (N,)
        n_steps: int = 100,
        edge_index: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Euler ODE サンプリング。Z1 ~ N(0,I) → Z0。

        attn_bias（BFS）をループ外で 1 回だけ計算し、全ステップで使い回す。
        Self-conditioning: 前ステップの z1_pred を次ステップの z_sc に渡す。
        """
        if device is None:
            device = cond.device

        B = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        raw = _unwrap_model(self.model)
        latent_dim = raw.latent_dim

        zt = torch.randn(n_atoms, latent_dim, device=device)

        # BFS は ODE ループ外で 1 回だけ
        attn_bias = raw.precompute_attn_inputs(edge_index, batch, n_atoms)

        # t: 1.0 → 0.0
        ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

        z_sc = None   # self-conditioning: 前ステップの予測
        for i in range(n_steps):
            t_val = ts[i]
            t = t_val.expand(B)
            dt = ts[i] - ts[i + 1]   # 正の値

            z1_pred = raw(zt, cond, t, batch, z_sc=z_sc, attn_bias=attn_bias)
            z_sc = z1_pred   # 次ステップ用に保存

            # Euler step: Zt_{t-dt} = Zt - dt * (Zt - Z1_pred) / (1 - t)
            denom = (1.0 - t_val).clamp(min=1e-4)
            zt = zt - dt * (zt - z1_pred) / denom

        return zt
