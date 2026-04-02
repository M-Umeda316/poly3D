"""
EGNN (Equivariant Graph Neural Network) 実装

参考: Satorras et al., "E(n) Equivariant Graph Neural Networks" (ICML 2021)

SE(3)-同変なメッセージパッシングを実装する。
  - ノード特徴量 h: 任意の不変量スカラー
  - 座標 x: 回転・反射・並進に対して同変
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter


_DIST_SQ_MAX: float = 100.0   # d² クランプ上限（√100 = 10 Å 相当）


def _silu() -> nn.SiLU:
    return nn.SiLU()


class EGNNLayer(nn.Module):
    """
    1 層分の EGNN。

    メッセージ計算:
        m_ij = φ_e(h_i, h_j, d_ij², e_ij)
    座標更新（並進不変・回転同変）:
        Δx_i = Σ_j (x_i - x_j) * φ_x(m_ij) / max(|N_i|, 1)
    ノード更新（残差接続あり）:
        h_i' = h_i + φ_h(h_i, Σ_j m_ij)
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_feat_dim: int,
        act: nn.Module | None = None,
        norm_coord: bool = True,
        residual: bool = True,
        coord_clamp: float = 10.0,
    ):
        super().__init__()
        self.norm_coord = norm_coord
        self.residual = residual
        self.coord_clamp = coord_clamp

        act = act or _silu()

        # エッジ MLP: (h_i, h_j, d², e_ij) → m_ij
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + edge_feat_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
        )

        # 座標重み MLP: m_ij → scalar
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            act,
            nn.Linear(hidden_dim // 2, 1),
        )

        # ノード MLP: (h_i, Σm_ij) → h_i'
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
        )

        # エッジメッセージ・ノード特徴量爆発を防ぐ LayerNorm（bf16 対策）
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # 座標 MLP の最終層を 0 に近い値で初期化（学習初期の爆発を防ぐ）
        nn.init.zeros_(self.coord_mlp[-1].weight)
        nn.init.zeros_(self.coord_mlp[-1].bias)

    def forward(
        self,
        h: Tensor,           # (N, hidden_dim)
        x: Tensor,           # (N, 3)
        edge_index: Tensor,  # (2, E)
        edge_attr: Tensor,   # (E, edge_feat_dim)
        batch: Optional[Tensor] = None,  # (N,) バッチ割り当て（未使用だが API 統一用）
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        h_new : (N, hidden_dim)  更新後ノード特徴量
        x_new : (N, 3)           更新後座標（SE(3)-同変）
        """
        src, dst = edge_index   # src→dst のエッジ

        # ─── エッジメッセージ ───
        diff = x[src] - x[dst]                     # (E, 3)
        dist_sq = (diff.pow(2)).sum(dim=-1, keepdim=True).clamp(max=_DIST_SQ_MAX)  # (E, 1)

        edge_in = torch.cat([h[src], h[dst], dist_sq, edge_attr], dim=-1)
        m = self.edge_norm(self.edge_mlp(edge_in))  # (E, hidden_dim)

        # ─── 座標更新 ───
        coord_w = self.coord_mlp(m)                 # (E, 1)
        if self.coord_clamp > 0:
            coord_w = coord_w.clamp(-self.coord_clamp, self.coord_clamp)

        coord_delta = scatter(
            diff * coord_w, src, dim=0,
            dim_size=x.size(0), reduce='sum',
        )                                           # (N, 3)

        if self.norm_coord:
            # 近傍数で割って更新量をスケーリング（bincount で allocation 節約）
            n_neighbors = torch.bincount(
                src, minlength=x.size(0),
            ).float().clamp(min=1.0).unsqueeze(-1)  # (N, 1)
            coord_delta = coord_delta / n_neighbors

        # scatter 後に再クランプ（diff * coord_w の積が大きい場合に備える）
        if self.coord_clamp > 0:
            coord_delta = coord_delta.clamp(-self.coord_clamp, self.coord_clamp)

        x_new = x + coord_delta                     # (N, 3)

        # ─── ノード更新 ───
        aggr_m = scatter(
            m, dst, dim=0,
            dim_size=h.size(0), reduce='sum',
        )                                           # (N, hidden_dim)

        node_in = torch.cat([h, aggr_m], dim=-1)   # (N, 2*hidden_dim)
        h_delta = self.node_mlp(node_in)            # (N, hidden_dim)

        if self.residual:
            h_new = self.node_norm(h + h_delta)
        else:
            h_new = self.node_norm(h_delta)

        return h_new, x_new


class EGNN(nn.Module):
    """
    複数層の EGNN スタック。

    入力:
        h0   : (N, in_node_dim)  原子特徴量（時刻埋め込み込み）
        x0   : (N, 3)            ノイズ付き座標
        edge_index: (2, E)
        edge_attr : (E, edge_feat_dim)
        batch     : (N,)

    出力:
        h    : (N, hidden_dim)   最終ノード特徴量
        x    : (N, 3)            更新後座標（予測 x₀ として使用）
    """

    def __init__(
        self,
        in_node_dim: int,
        hidden_dim: int,
        edge_feat_dim: int,
        n_layers: int = 6,
        norm_coord: bool = True,
        residual: bool = True,
        coord_clamp: float = 10.0,
    ):
        super().__init__()

        self.input_proj = nn.Linear(in_node_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)   # input_proj 後の bf16 オーバーフロー防止

        self.layers = nn.ModuleList([
            EGNNLayer(
                hidden_dim=hidden_dim,
                edge_feat_dim=edge_feat_dim,
                norm_coord=norm_coord,
                residual=residual,
                coord_clamp=coord_clamp,
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        h0: Tensor,          # (N, in_node_dim)
        x0: Tensor,          # (N, 3)
        edge_index: Tensor,  # (2, E)
        edge_attr: Tensor,   # (E, edge_feat_dim)
        batch: Optional[Tensor] = None,  # (N,)
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        h : (N, hidden_dim)  最終ノード特徴量
        x : (N, 3)           更新後座標
        """
        h = self.input_norm(self.input_proj(h0))   # (N, hidden_dim)
        x = x0                                    # (N, 3)

        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr, batch)

        return h, x
