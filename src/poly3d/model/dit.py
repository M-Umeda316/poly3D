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
  attn_bias : (H, N, N) float  precompute_attn_inputs() の出力（省略時は内部計算）

出力:
  z1_pred : (N, latent_dim)  Z1 の予測（velocity）

パフォーマンスメモ:
  attn_bias は BFS（CPU Python）を含むため計算コストが高い。
  FlowMatching.loss / sample から precompute_attn_inputs() を 1 バッチ 1 回だけ
  呼び出し、その結果を attn_bias として渡すこと（学習時 2× 削減、サンプリング時 100× 削減）。
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
    """batch (N,) → (N, N) bool mask。同じ分子内のノードペアのみ True。"""
    return batch.unsqueeze(0) == batch.unsqueeze(1)   # (N, N)


def _build_batch_pos_bias(
    edge_index: Tensor, batch: Tensor, num_nodes: int,
    bias_module: GraphDistanceBias,
) -> Tensor:
    """
    バッチ内の各分子ごとにグラフ距離 → one-hot → bias を計算し、
    block-diagonal な (N, N, n_heads) テンソルに詰める。
    異なる分子間は 0（attn_bias で -1e9 に上書きされるので問題ない）。
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

        offset = node_indices[0].item()
        src, dst = edge_index
        edge_mask = node_mask[src] & node_mask[dst]
        local_ei = edge_index[:, edge_mask] - offset   # (2, E_b)

        dist_mat = compute_graph_distance(local_ei, n_b)
        oh = dist_to_onehot(dist_mat)                  # (n_b, n_b, 5)
        local_bias = bias_module(oh)                    # (n_b, n_b, n_heads)

        idx = node_indices
        bias[idx.unsqueeze(1), idx.unsqueeze(0)] = local_bias

    return bias   # (N, N, n_heads)


class DiTBlock(nn.Module):
    """
    1 層の DiT ブロック。

    Pre-LN → F.scaled_dot_product_attention (Flash Attention) → FFN

    attn_bias は (H, N, N) float。
    - 同分子内: pos_bias 値（use_pos_bias=False なら 0）
    - 異分子間: -1e9（softmax で実質ゼロになりクロス分子 attention を遮断）
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
        self.dropout = dropout

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

    def forward(
        self,
        x: Tensor,
        attn_bias: Optional[Tensor] = None,   # (H, N, N) float
    ) -> Tensor:
        """
        x         : (N, hidden_dim)
        attn_bias : (H, N, N) float — pos_bias + block-diagonal mask を統合済み
        """
        N = x.size(0)
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)   # (N, H, D)

        # F.scaled_dot_product_attention: (B, H, N, D) 形式に変換
        q = q.permute(1, 0, 2).unsqueeze(0)   # (1, H, N, D)
        k = k.permute(1, 0, 2).unsqueeze(0)
        v = v.permute(1, 0, 2).unsqueeze(0)

        # attn_bias: (H, N, N) → (1, H, N, N)
        sdpa_mask = attn_bias.unsqueeze(0) if attn_bias is not None else None

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )   # (1, H, N, D)

        out = out.squeeze(0).permute(1, 0, 2).reshape(N, self.hidden_dim)
        x = x + self.out_proj(out)
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
        self.n_heads = n_heads
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

    def precompute_attn_inputs(
        self,
        edge_index: Optional[Tensor],
        batch: Tensor,
        num_nodes: int,
        dist_mat: Optional[Tensor] = None,
    ) -> Tensor:
        """
        グラフ距離 BFS とブロック対角マスクを 1 回計算し、
        DiTBlock に渡す (H, N, N) float の combined attention bias を返す。

          同分子内  : pos_bias 値 (use_pos_bias=False なら 0)
          異分子間  : -1e9  (softmax で実質ゼロ → cross-molecule attention を遮断)

        Parameters
        ----------
        dist_mat : (total_N, total_N) int8 optional
            DataLoader ワーカーで事前計算済みのブロック対角距離行列。
            提供された場合は BFS をスキップして高速パスを使用。
            None の場合はオンザフライで BFS 計算（低速パス）。
        """
        device = batch.device
        attn_mask = _build_block_diagonal_mask(batch)   # (N, N) bool

        if self.pos_bias is not None:
            if dist_mat is not None:
                # 高速パス: ワーカーで事前計算済みの距離行列を使用（BFS スキップ）
                oh = dist_to_onehot(dist_mat.long().to(device))       # (N, N, 5)
                pos_bias_val = self.pos_bias(oh)                       # (N, N, H)
                combined = pos_bias_val.permute(2, 0, 1).contiguous() # (H, N, N)
            elif edge_index is not None:
                # 低速パス: オンザフライ BFS（dist_mat 未提供時の後方互換）
                pos_bias_val = _build_batch_pos_bias(
                    edge_index, batch, num_nodes, self.pos_bias
                )                                                       # (N, N, H)
                combined = pos_bias_val.permute(2, 0, 1).contiguous()  # (H, N, N)
            else:
                combined = torch.zeros(self.n_heads, num_nodes, num_nodes, device=device)
        else:
            combined = torch.zeros(self.n_heads, num_nodes, num_nodes, device=device)

        # 異分子間を -1e9 でマスク
        combined = combined.masked_fill(~attn_mask.unsqueeze(0), -1e9)
        return combined   # (H, N, N) float

    def forward(
        self,
        zt: Tensor,            # (N, latent_dim)
        cond: Tensor,          # (N, cond_dim)
        t: Tensor,             # (B,) float  0〜1
        batch: Tensor,         # (N,) int
        edge_index: Optional[Tensor] = None,   # pos_bias 計算用（attn_bias 未提供時のみ使用）
        z_sc: Optional[Tensor] = None,         # self-conditioning
        attn_bias: Optional[Tensor] = None,    # (H, N, N) precomputed — 毎回計算を避けるため
    ) -> Tensor:
        """
        Returns
        -------
        z1_pred : (N, latent_dim)
        """
        N = zt.size(0)

        # Time embedding: t (B,) → (N, time_dim)
        t_emb = self.time_emb(t)[batch]   # (N, time_dim)

        if z_sc is None:
            z_sc = torch.zeros_like(zt)

        # 入力結合
        x = self.input_proj(torch.cat([zt, cond, t_emb, z_sc], dim=-1))

        # attn_bias が未提供なら内部で計算（backward compat）
        if attn_bias is None:
            attn_bias = self.precompute_attn_inputs(edge_index, batch, N, dist_mat=None)

        # Transformer
        for block in self.blocks:
            x = block(x, attn_bias)

        return self.out_proj(self.norm_out(x))
