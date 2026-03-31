"""
Structural VAE

Encoder E: (Ci, pos, edge_index, e_cond) → (μ, log_σ)  各原子ごとの潜在変数
Decoder D: (Z, Ci, edge_index, e_cond) → pos_pred

潜在変数 Zi = μi + exp(log_σi) * ε  (reparameterization)
潜在次元: latent_dim (デフォルト 16)

どちらも EGNN ベースなので SE(3)-equivariant。
Encoder は実座標 pos を入力に取り、潜在空間 Z を出力。
Decoder は Z を初期座標として EGNN に通し、座標を復元する。
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter

from poly3d.model.egnn import EGNN


class VAEEncoder(nn.Module):
    """
    E(Ci, pos, edge_index, e_cond) → (μ, log_σ)  各 (N, latent_dim)

    Parameters
    ----------
    cond_dim   : Ci の次元 (ConditionalEncoder の hidden_dim)
    edge_dim   : e_cond の次元
    hidden_dim : EGNN 内部次元
    latent_dim : 潜在変数の次元
    n_layers   : EGNN 層数
    """

    def __init__(
        self,
        cond_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 16,
        n_layers: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Ci を EGNN の入力次元に投影
        self.input_proj = nn.Linear(cond_dim, hidden_dim)

        self.egnn = EGNN(
            in_node_dim=hidden_dim,
            hidden_dim=hidden_dim,
            edge_feat_dim=edge_dim,
            n_layers=n_layers,
        )

        # EGNN 出力 → μ, log_σ
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        cond: Tensor,        # (N, cond_dim)
        pos: Tensor,         # (N, 3)
        edge_index: Tensor,  # (2, E)
        e_cond: Tensor,      # (E, edge_dim)
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        mu     : (N, latent_dim)
        logvar : (N, latent_dim)
        """
        h0 = self.input_proj(cond)   # (N, hidden_dim)
        h, _ = self.egnn(h0, pos, edge_index, e_cond, batch)
        mu = self.mu_head(h)
        # exp(logvar) が float32 でオーバーフローしないよう clamp
        # exp(10) ≈ 22026（十分大きな分散）、exp(-10) ≈ 4.5e-5（ほぼ決定論的）
        logvar = self.logvar_head(h).clamp(-10.0, 10.0)
        return mu, logvar


class VAEDecoder(nn.Module):
    """
    D(Z, Ci, edge_index, e_cond) → pos_pred

    Z は潜在変数 (N, latent_dim)。
    Ci との concat を EGNN に入力し、座標を復元する。

    初期座標: [Z; Ci] → MLP → (N, 3) でノードごとにランダムに異なる
    初期値を生成。全ゼロだと EGNN の距離入力 d²=0 で区別不能になるため。
    """

    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 4,
    ):
        super().__init__()

        in_dim = latent_dim + cond_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # 初期座標推定 MLP: [Z; Ci] → 3
        self.init_pos = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        # 最終層を小さい乱数で初期化。
        # zeros だと x0 が全ノード同一（=0）になり EGNN の d²=0 で区別不能。
        # 勾配も ∂(cross_product)/∂x ~ 1/eps = 1e8 になり爆発する。
        # std=0.01 程度の小さな値で非ゼロな初期座標を保証する。
        nn.init.normal_(self.init_pos[-1].weight, std=0.01)
        nn.init.zeros_(self.init_pos[-1].bias)

        self.egnn = EGNN(
            in_node_dim=hidden_dim,
            hidden_dim=hidden_dim,
            edge_feat_dim=edge_dim,
            n_layers=n_layers,
        )

    def forward(
        self,
        z: Tensor,           # (N, latent_dim)
        cond: Tensor,        # (N, cond_dim)
        edge_index: Tensor,  # (2, E)
        e_cond: Tensor,      # (E, edge_dim)
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Returns
        -------
        pos_pred : (N, 3)
        """
        feat = torch.cat([z, cond], dim=-1)
        h0 = self.input_proj(feat)

        # 初期座標をノード特徴量から推定（ノードごとに異なる値）
        x0 = self.init_pos(feat)

        # 重心ゼロ化（並進不変性）
        if batch is None:
            x0 = x0 - x0.mean(dim=0, keepdim=True)
        else:
            from torch_scatter import scatter_mean
            mean = scatter_mean(x0, batch, dim=0)
            x0 = x0 - mean[batch]

        _, pos_pred = self.egnn(h0, x0, edge_index, e_cond, batch)

        # 出力も重心ゼロに正規化
        if batch is None:
            pos_pred = pos_pred - pos_pred.mean(dim=0, keepdim=True)
        else:
            from torch_scatter import scatter_mean as _sm
            mean = _sm(pos_pred, batch, dim=0)
            pos_pred = pos_pred - mean[batch]

        return pos_pred


class StructuralVAE(nn.Module):
    """
    Encoder + Decoder のラッパー。

    学習時: encode → reparameterize → decode → 損失計算
    推論時: sample Z → decode
    """

    def __init__(
        self,
        cond_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 16,
        enc_layers: int = 4,
        dec_layers: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = VAEEncoder(
            cond_dim=cond_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_layers=enc_layers,
        )
        self.decoder = VAEDecoder(
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            n_layers=dec_layers,
        )

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def encode(
        self,
        cond: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        e_cond: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns
        -------
        z      : (N, latent_dim)
        mu     : (N, latent_dim)
        logvar : (N, latent_dim)
        """
        mu, logvar = self.encoder(cond, pos, edge_index, e_cond, batch)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(
        self,
        z: Tensor,
        cond: Tensor,
        edge_index: Tensor,
        e_cond: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        return self.decoder(z, cond, edge_index, e_cond, batch)

    def forward(
        self,
        cond: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        e_cond: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns
        -------
        pos_pred : (N, 3)
        mu       : (N, latent_dim)
        logvar   : (N, latent_dim)
        """
        z, mu, logvar = self.encode(cond, pos, edge_index, e_cond, batch)
        pos_pred = self.decode(z, cond, edge_index, e_cond, batch)
        return pos_pred, mu, logvar

    def kl_loss(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """KL(q(Z|X) || N(0,I)) の平均"""
        return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()

    @torch.no_grad()
    def sample_z(self, n_atoms: int, device: torch.device) -> Tensor:
        return torch.randn(n_atoms, self.latent_dim, device=device)
