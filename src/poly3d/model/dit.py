"""
Latent Diffusion Transformer (DiT)

潜在空間 Zt (N, latent_dim) に対して flow matching を実行する Transformer。

論文の ADiT ベース:
  - 各ノードを 1 トークンとして扱う
  - グラフ距離ベースの positional biased attention
  - Self-conditioning (p=0.5 で前回の予測を入力に)
  - 条件付け Ci を concat して入力
  - **batch 内の分子間 attention を block-diagonal mask で遮断**

入力:
  Zt    : (N, latent_dim)  ノイズ付き潜在変数
  cond  : (N, cond_dim)   ConditionalEncoder の Ci
  t     : (B,) float       時刻
  batch : (N,) int         各ノードのバッチ番号
  z_sc  : (N, latent_dim) self-conditioning（None → ゼロ）

出力:
  z1_pred : (N, latent_dim)  Z1 の予測（velocity）
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from poly3d.model.pos_bias import GraphDistanceBias, compute_graph_distance, dist_to_onehot


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """t: (B,) float → (B, dim)"""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / (half - 1)
        )
        args = t[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


def _build_block_diagonal_mask(batch: Tensor) -> Tensor:
    """
    batch (N,) → (N, N) bool mask。同じ分子内のノードペアのみ True。
    """
    return batch.unsqueeze(0) == batch.unsqueeze(1)   # (N, N)


def _build_batch_pos_bias(
    edge_index: Tensor, batch: Tensor, num_nodes: int,
    bias_module: GraphDistanceBias,
) -> Tensor:
    """
    バッチ内の各分子ごとにグラフ距離 → one-hot → bias を計算し、
    block-diagonal な (N, N, n_heads) テンソルに詰める。
    異なる分子間は 0 (attention mask で -inf にされるので問題ない)。
    """
    device = edge_index.device
    B = int(batch.max().item()) + 1
    n_heads = bias_module.n_heads
    bias = torch.zeros(num_nodes, num_nodes, n_heads, device=device)

    for b in range(B):
        node_mask = (batch == b)
        node_indices = node_mask.nonzero(as_tuple=True)[0]
        n_b = node_indices.size(0)
        if n_b == 0:
            continue

        # この分子に属するエッジを抽出してローカルインデックスに変換
        offset = node_indices[0].item()
        src, dst = edge_index
        edge_mask = node_mask[src] & node_mask[dst]
        local_ei = edge_index[:, edge_mask] - offset  # (2, E_b)

        dist_mat = compute_graph_distance(local_ei, n_b)
        oh = dist_to_onehot(dist_mat)           # (n_b, n_b, 5)
        local_bias = bias_module(oh)             # (n_b, n_b, n_heads)

        # block-diagonal に配置
        idx = node_indices
        bias[idx.unsqueeze(1), idx.unsqueeze(0)] = local_bias

    return bias


class DiTBlock(nn.Module):
    """
    1 層の DiT ブロック。

    Pre-LN → Multi-Head Attention (+ positional bias + block-diagonal mask) → FFN
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        ffn_dim = hidden_dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        x         : (N, hidden_dim)
        attn_mask : (N, N) bool — True = attend, False = block
        bias      : (N, N, n_heads) — positional bias
        """
        N = x.size(0)
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (N, H, D)

        # (H, N, N) attention logit
        scale = math.sqrt(self.head_dim)
        attn = torch.einsum('ihd,jhd->hij', q, k) / scale  # (H, N, N)

        # Positional bias
        if bias is not None:
            attn = attn + bias.permute(2, 0, 1)   # (H, N, N)

        # Block-diagonal mask: 異なる分子のペアは -inf
        if attn_mask is not None:
            attn = attn.masked_fill(~attn_mask.unsqueeze(0), float('-inf'))

        attn = self.attn_drop(F.softmax(attn, dim=-1))

        # NaN 防止: 全マスクされた行は softmax が NaN になるので 0 に
        attn = attn.nan_to_num(0.0)

        out = torch.einsum('hij,jhd->ihd', attn, v).reshape(N, self.hidden_dim)
        x = x + self.out_proj(out)

        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class LatentDiT(nn.Module):
    """
    Latent Diffusion Transformer。

    PyG Batch のフラットなノード列を受け取り、
    batch テンソルに基づいて block-diagonal attention を適用する。

    Parameters
    ----------
    latent_dim  : 潜在変数次元
    cond_dim    : Ci の次元
    time_dim    : time embedding 次元
    hidden_dim  : Transformer 隠れ次元
    n_heads     : attention ヘッド数
    n_layers    : DiT ブロック数
    ffn_mult    : FFN 拡大係数
    use_pos_bias: グラフ距離 positional bias を使うか
    """

    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        time_dim: int = 64,
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        use_pos_bias: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.use_pos_bias = use_pos_bias

        # Time embedding
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # 入力投影: [Zt; Ci; t_emb; z_sc] → hidden_dim
        in_dim = latent_dim + cond_dim + time_dim + latent_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Transformer ブロック
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, n_heads, ffn_mult, dropout)
            for _ in range(n_layers)
        ])

        # Positional bias
        self.pos_bias = GraphDistanceBias(n_heads) if use_pos_bias else None

        # 出力
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        zt: Tensor,            # (N, latent_dim)
        cond: Tensor,          # (N, cond_dim)
        t: Tensor,             # (B,) float  0〜1
        batch: Tensor,         # (N,) int
        edge_index: Optional[Tensor] = None,  # (2, E) pos_bias 用
        z_sc: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Returns
        -------
        z1_pred : (N, latent_dim)
        """
        N = zt.size(0)

        # Time embedding: t (B,) → (N, time_dim)
        t_emb_b = self.time_emb(t)     # (B, time_dim)
        t_emb = t_emb_b[batch]         # (N, time_dim)

        if z_sc is None:
            z_sc = torch.zeros_like(zt)

        # 入力結合
        x = self.input_proj(torch.cat([zt, cond, t_emb, z_sc], dim=-1))

        # Block-diagonal attention mask
        attn_mask = _build_block_diagonal_mask(batch)  # (N, N) bool

        # Positional bias (分子ごとに計算して block-diagonal に詰める)
        bias = None
        if self.use_pos_bias and self.pos_bias is not None and edge_index is not None:
            bias = _build_batch_pos_bias(edge_index, batch, N, self.pos_bias)

        # Transformer
        for block in self.blocks:
            x = block(x, attn_mask, bias)

        z1_pred = self.out_proj(self.norm_out(x))
        return z1_pred

    def forward_with_selfcond(
        self,
        zt: Tensor,
        cond: Tensor,
        t: Tensor,
        batch: Tensor,
        edge_index: Optional[Tensor] = None,
        p_selfcond: float = 0.5,
    ) -> Tensor:
        """
        学習時の self-conditioning (2パス)。
        p_selfcond の確率で1パス目の予測を2パス目に渡す。
        """
        if self.training and torch.rand(1).item() < p_selfcond:
            with torch.no_grad():
                z_sc = self.forward(zt, cond, t, batch, edge_index, z_sc=None).detach()
        else:
            z_sc = None

        return self.forward(zt, cond, t, batch, edge_index, z_sc)
