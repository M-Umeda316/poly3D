"""
Conditional Encoder

分子グラフのトポロジー情報（3D 座標なし）をエンコードする。
diffusion の各ステップで再計算するのではなく、1 回だけ実行して
conditioning ベクトルとして denoiser に渡す。

入力:
  - atom_type_idx : (N,) int  → nn.Embedding で学習可能な埋め込み
  - hyb_idx       : (N,) int  → nn.Embedding
  - atom_cont     : (N, ATOM_CONT_DIM)  連続値特徴量
  - bond_type_idx : (E,) int  → nn.Embedding
  - bond_cont     : (E, BOND_CONT_DIM) 連続値特徴量
  - rwpe          : (N, RWPE_DIM)  Random Walk PE
  - edge_index    : (2, E)

出力:
  - h_cond : (N, hidden_dim)  ノードレベルの条件付け特徴量
  - e_cond : (E, edge_dim)    エッジレベルの条件付け特徴量
  - cond   : (N, hidden_dim)  グローバル + ローカル融合ベクトル Ci（VAE/DiT に渡す）

アーキテクチャ:
  1. Atom embedding + hybridization embedding + cont 投影 + RWPE 投影 → h_init
  2. Bond embedding + cont 投影 → e_init
  3. MPNN layers で h_cond を計算（位置更新なし、トポロジーのみ）
  4. e_cond は各エッジに対して (h_i, h_j, e_init) から計算
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter, scatter_mean

from poly3d.model.features import (
    ATOM_TYPE_VOCAB, HYBRIDIZATION_VOCAB, BOND_TYPE_VOCAB,
    ATOM_CONT_DIM, BOND_CONT_DIM, RWPE_DIM, LAPPE_DIM,
)


class MPNNLayer(nn.Module):
    """
    位置更新なしのメッセージパッシング層（トポロジーエンコード用）。

    m_ij = φ_e(h_i, h_j, e_ij)
    h_i' = h_i + φ_h(h_i, Σ_j m_ij)
    e_ij' = e_ij + φ_edge(m_ij)   # エッジ特徴量も更新
    """

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        act = nn.SiLU()

        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, edge_dim),
            act,
            nn.Linear(edge_dim, edge_dim),
        )

        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)

    def forward(
        self,
        h: Tensor,           # (N, hidden_dim)
        e: Tensor,           # (E, edge_dim)
        edge_index: Tensor,  # (2, E)
    ) -> Tuple[Tensor, Tensor]:
        src, dst = edge_index

        # メッセージ
        m = self.msg_mlp(torch.cat([h[src], h[dst], e], dim=-1))   # (E, hidden_dim)

        # ノード更新（残差）
        aggr = scatter(m, dst, dim=0, dim_size=h.size(0), reduce='sum')
        h_new = self.node_norm(h + self.node_mlp(torch.cat([h, aggr], dim=-1)))

        # エッジ更新（残差）
        e_new = self.edge_norm(e + self.edge_mlp(torch.cat([m, e], dim=-1)))

        return h_new, e_new


class ConditionalEncoder(nn.Module):
    """
    分子グラフトポロジー → conditioning ベクトル。

    Parameters
    ----------
    hidden_dim      : ノード隠れ次元数
    edge_dim        : エッジ隠れ次元数
    n_layers        : MPNN 層数
    atom_emb_dim    : 原子タイプ embedding 次元
    hyb_emb_dim     : 混成軌道 embedding 次元
    bond_emb_dim    : 結合タイプ embedding 次元
    use_rwpe        : RWPE を使うか
    use_lappe       : LapPE を使うか
    lappe_dim       : LapPE の次元（use_lappe=True 時のみ有効）
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        edge_dim: int = 64,
        n_layers: int = 4,
        atom_emb_dim: int = 32,
        hyb_emb_dim: int = 16,
        bond_emb_dim: int = 16,
        use_rwpe: bool = True,
        use_lappe: bool = False,
    ):
        super().__init__()
        self.use_rwpe = use_rwpe
        self.use_lappe = use_lappe

        # ── Embedding テーブル ─────────────────────────────────────────────
        self.atom_emb = nn.Embedding(ATOM_TYPE_VOCAB, atom_emb_dim)
        self.hyb_emb = nn.Embedding(HYBRIDIZATION_VOCAB, hyb_emb_dim)
        self.bond_emb = nn.Embedding(BOND_TYPE_VOCAB, bond_emb_dim)

        # ── 入力次元の計算 ─────────────────────────────────────────────────
        node_in_dim = atom_emb_dim + hyb_emb_dim + ATOM_CONT_DIM
        if use_rwpe:
            node_in_dim += RWPE_DIM
        if use_lappe:
            node_in_dim += LAPPE_DIM

        edge_in_dim = bond_emb_dim + BOND_CONT_DIM

        # ── 入力投影 ───────────────────────────────────────────────────────
        self.node_proj = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_in_dim, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),
        )

        # ── MPNN 層 ────────────────────────────────────────────────────────
        self.layers = nn.ModuleList([
            MPNNLayer(hidden_dim, edge_dim)
            for _ in range(n_layers)
        ])

        # ── Global pooling → Ci ────────────────────────────────────────────
        # g = scatter_mean(h, batch) → Ci = MLP([h; g_expand])
        self.global_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        atom_type_idx: Tensor,    # (N,) int
        hyb_idx: Tensor,          # (N,) int
        atom_cont: Tensor,        # (N, ATOM_CONT_DIM)
        bond_type_idx: Tensor,    # (E,) int
        bond_cont: Tensor,        # (E, BOND_CONT_DIM)
        edge_index: Tensor,       # (2, E)
        rwpe: Optional[Tensor] = None,   # (N, RWPE_DIM)
        lappe: Optional[Tensor] = None,  # (N, LAPPE_DIM)
        batch: Optional[Tensor] = None,  # (N,) バッチ割り当て（None → 全部0）
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns
        -------
        h_cond : (N, hidden_dim)
        e_cond : (E, edge_dim)
        cond   : (N, hidden_dim)  Ci = MLP([h; g_i])
        """
        # ── ノード初期特徴量 ───────────────────────────────────────────────
        N = atom_type_idx.size(0)
        dev = atom_type_idx.device
        node_parts = [
            self.atom_emb(atom_type_idx),   # (N, atom_emb_dim)
            self.hyb_emb(hyb_idx),          # (N, hyb_emb_dim)
            atom_cont,                       # (N, ATOM_CONT_DIM)
        ]
        if self.use_rwpe:
            # rwpe が None の場合（データに含まれていないとき）はゼロで埋める
            node_parts.append(rwpe if rwpe is not None
                              else torch.zeros(N, RWPE_DIM, device=dev))
        if self.use_lappe:
            # lappe が None の場合も同様にゼロで埋める
            node_parts.append(lappe if lappe is not None
                              else torch.zeros(N, LAPPE_DIM, device=dev))

        h = self.node_proj(torch.cat(node_parts, dim=-1))   # (N, hidden_dim)

        # ── エッジ初期特徴量 ───────────────────────────────────────────────
        edge_parts = [
            self.bond_emb(bond_type_idx),   # (E, bond_emb_dim)
            bond_cont,                       # (E, BOND_CONT_DIM)
        ]
        e = self.edge_proj(torch.cat(edge_parts, dim=-1))   # (E, edge_dim)

        # ── MPNN 層 ────────────────────────────────────────────────────────
        for layer in self.layers:
            h, e = layer(h, e, edge_index)

        # ── Global pooling → Ci ────────────────────────────────────────────
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        # num_graphs: batch index から GPU-CPU sync なしで推定
        num_graphs = batch[-1] + 1 if batch.numel() > 0 else 1
        g = scatter(h, batch, dim=0, dim_size=num_graphs, reduce='sum')  # (B, hidden_dim)
        g_expand = g[batch]                  # (N, hidden_dim)
        cond = self.global_proj(torch.cat([h, g_expand], dim=-1))  # (N, hidden_dim)

        return h, e, cond
