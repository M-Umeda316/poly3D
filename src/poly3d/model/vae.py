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
from torch_scatter import scatter_mean

from poly3d.model.egnn import EGNN, EGNNLayer, EGTLayer


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
        egt_every: int = 0,
        n_heads: int = 8,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.egt_every = egt_every

        # 入力投影（従来 EGNN 内部に統合されていたものを encoder 側に移設）。
        # EGT を層間に挟むため層スタックを ModuleList で直接構築する。
        # egt_every=0 なら全層 EGNNLayer（完全後方互換）。
        self.input_proj = nn.Linear(cond_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.layers = nn.ModuleList()
        self.layer_is_egt: list[bool] = []
        for i in range(n_layers):
            use_egt = egt_every > 0 and ((i + 1) % egt_every == 0)
            if use_egt:
                self.layers.append(EGTLayer(
                    hidden_dim=hidden_dim,
                    edge_feat_dim=edge_dim,
                    n_heads=n_heads,
                ))
            else:
                self.layers.append(EGNNLayer(
                    hidden_dim=hidden_dim,
                    edge_feat_dim=edge_dim,
                ))
            self.layer_is_egt.append(use_egt)

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
        dist_mat: Optional[Tensor] = None,  # (total_N, total_N) グラフ距離（任意）
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        mu     : (N, latent_dim)
        logvar : (N, latent_dim)

        エンコーダは実座標 pos を入力座標に取り、EGNN/EGT で h を更新する。
        座標 x の更新は使わず（潜在は h のみから作る）h の最終値を μ/logσ に写す。
        dist_mat は EGTLayer の b_graph に供給（None なら 3D 距離バイアスのみ）。
        """
        h = self.input_norm(self.input_proj(cond))   # (N, hidden_dim)
        x = pos
        for layer, is_egt in zip(self.layers, self.layer_is_egt):
            if is_egt:
                h, x = layer(h, x, edge_index, e_cond, batch, dist_mat)
            else:
                h, x = layer(h, x, edge_index, e_cond, batch)
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

    層構成: EGNNLayer（局所）と EGTLayer（局所＋大域アテンション）を
    egt_every 層ごとに交互配置する。egt_every=0 なら全層 EGNNLayer（完全後方互換）。
    例: n_layers=4, egt_every=2 → [EGNN, EGT, EGNN, EGT]。
    """

    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 4,
        egt_every: int = 0,
        n_heads: int = 8,
    ):
        super().__init__()

        in_dim = latent_dim + cond_dim
        self.egt_every = egt_every

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

        # 入力投影（従来 EGNN 内部に統合されていたものを decoder 側に移設）。
        # EGT を層間に挟むため層スタックを ModuleList で直接構築する。
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # EGNNLayer / EGTLayer を egt_every ごとに交互配置
        self.layers = nn.ModuleList()
        self.layer_is_egt: list[bool] = []
        for i in range(n_layers):
            use_egt = egt_every > 0 and ((i + 1) % egt_every == 0)
            if use_egt:
                self.layers.append(EGTLayer(
                    hidden_dim=hidden_dim,
                    edge_feat_dim=edge_dim,
                    n_heads=n_heads,
                ))
            else:
                self.layers.append(EGNNLayer(
                    hidden_dim=hidden_dim,
                    edge_feat_dim=edge_dim,
                ))
            self.layer_is_egt.append(use_egt)

    def forward(
        self,
        z: Tensor,           # (N, latent_dim)
        cond: Tensor,        # (N, cond_dim)
        edge_index: Tensor,  # (2, E)
        e_cond: Tensor,      # (E, edge_dim)
        batch: Optional[Tensor] = None,
        dist_mat: Optional[Tensor] = None,  # (total_N, total_N) グラフ距離（任意）
        init_scaffold: Optional[Tensor] = None,  # (N, 3) MDS 大域足場（任意）
    ) -> Tensor:
        """
        Returns
        -------
        pos_pred : (N, 3)

        dist_mat は batch のブロック対角グラフ距離行列（(total_N, total_N) int8 想定）。
        EGTLayer の b_graph（グラフ距離アテンションバイアス）に供給する。None なら
        EGT は 3D 距離バイアスのみで動作（完全後方互換）。

        init_scaffold は MDS 由来の大域足場 (N, 3)。与えられれば初期座標を
        「足場 + 小 MLP 補正」に置換する。None なら従来どおり MLP のみ（完全後方互換）。
        """
        feat = torch.cat([z, cond], dim=-1)   # (N, latent_dim + cond_dim)

        # 初期座標をノード特徴量から推定（ノードごとに異なる値）。
        # init_scaffold があれば大域足場に per-atom MLP 補正を加算する
        # （base = 足場、なければ 0 → x0 = init_pos(feat) と完全一致で後方互換）。
        mlp_pos = self.init_pos(feat)
        x0 = mlp_pos if init_scaffold is None else init_scaffold + mlp_pos

        # 重心ゼロ化（並進不変性）
        if batch is None:
            x0 = x0 - x0.mean(dim=0, keepdim=True)
        else:
            mean = scatter_mean(x0, batch, dim=0)
            x0 = x0 - mean[batch]

        h = self.input_norm(self.input_proj(feat))   # (N, hidden_dim)
        x = x0

        for layer, is_egt in zip(self.layers, self.layer_is_egt):
            if is_egt:
                h, x = layer(h, x, edge_index, e_cond, batch, dist_mat)
            else:
                h, x = layer(h, x, edge_index, e_cond, batch)

        pos_pred = x

        # 出力も重心ゼロに正規化
        if batch is None:
            pos_pred = pos_pred - pos_pred.mean(dim=0, keepdim=True)
        else:
            mean = scatter_mean(pos_pred, batch, dim=0)
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
        egt_every: int = 0,
        enc_egt_every: int = 0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.egt_every = egt_every
        self.enc_egt_every = enc_egt_every

        self.encoder = VAEEncoder(
            cond_dim=cond_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_layers=enc_layers,
            egt_every=enc_egt_every,
        )
        self.decoder = VAEDecoder(
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            n_layers=dec_layers,
            egt_every=egt_every,
        )

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(
        self,
        cond: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        e_cond: Tensor,
        batch: Optional[Tensor] = None,
        dist_mat: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns
        -------
        z      : (N, latent_dim)
        mu     : (N, latent_dim)
        logvar : (N, latent_dim)
        """
        mu, logvar = self.encoder(cond, pos, edge_index, e_cond, batch, dist_mat)
        # 評価（eval）時は決定論デコード（z = mu）で乱数ノイズを排除し、
        # val loss のばらつき＝チェックポイント選択ノイズを防ぐ。
        # 学習（train）時は従来どおり reparameterize で確率的サンプリング。
        # KL 損失用に mu/logvar は常に返す。
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        return z, mu, logvar

    def decode(
        self,
        z: Tensor,
        cond: Tensor,
        edge_index: Tensor,
        e_cond: Tensor,
        batch: Optional[Tensor] = None,
        dist_mat: Optional[Tensor] = None,
        init_scaffold: Optional[Tensor] = None,
    ) -> Tensor:
        return self.decoder(z, cond, edge_index, e_cond, batch, dist_mat, init_scaffold)

    def forward(
        self,
        cond: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        e_cond: Tensor,
        batch: Optional[Tensor] = None,
        dist_mat: Optional[Tensor] = None,
        init_scaffold: Optional[Tensor] = None,
        robust_noise_std: float = 0.0,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        """
        Returns
        -------
        pos_pred    : (N, 3)          posterior 潜在 z のデコード（通常 recon）
        mu          : (N, latent_dim)
        logvar      : (N, latent_dim)
        pos_robust  : (N, 3) or None  off-manifold 潜在 z' のデコード（後述）

        eval 時は encode が決定論デコード（z = mu）を返すため、forward も
        自動的に決定論的になる（同一入力→同一 pos_pred）。train 時は確率的。

        dist_mat（batch のブロック対角グラフ距離行列, 任意）は decoder の EGTLayer
        の b_graph バイアスに配線される。デフォルト None で完全後方互換。

        init_scaffold（MDS 大域足場 (N, 3), 任意）は decoder の初期座標足場に配線。
        デフォルト None で従来どおり MLP のみの初期座標（完全後方互換）。

        robust_noise_std（off-manifold ロバスト化, 任意）:
        学習時かつ robust_noise_std>0 のとき、posterior 潜在 z に等方ガウス雑音を
        加えた off-manifold 潜在 `z' = z + robust_noise_std * randn` を追加でデコードし
        pos_robust として返す。呼び出し側はこの出力に GT 不要のガードレール損失
        （clash + bond_range）だけを課し、「多少ずれた潜在でもデコーダが壊れない」
        ことを学習させる（DiT が生成する off-manifold 潜在への頑健化）。
        デフォルト 0.0・eval 時は pos_robust=None で完全後方互換。
        """
        z, mu, logvar = self.encode(cond, pos, edge_index, e_cond, batch, dist_mat)
        pos_pred = self.decode(z, cond, edge_index, e_cond, batch, dist_mat, init_scaffold)

        # off-manifold 潜在のデコード（学習時のみ・GT 不要のガードレール損失用）
        pos_robust: Optional[Tensor] = None
        if self.training and robust_noise_std > 0.0:
            z_rob = z + robust_noise_std * torch.randn_like(z)
            pos_robust = self.decode(
                z_rob, cond, edge_index, e_cond, batch, dist_mat, init_scaffold
            )
        return pos_pred, mu, logvar, pos_robust

