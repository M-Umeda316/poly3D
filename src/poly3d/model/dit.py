"""
Latent Diffusion Transformer (DiT)

潜在空間 Zt (N, latent_dim) に対して flow matching を実行する Transformer。

論文の ADiT ベース:
  - 各ノードを 1 トークンとして扱う
  - グラフ距離ベースの positional biased attention
  - Self-conditioning (p=0.5 で前回の予測を入力に)
  - 条件付け Ci を concat して入力
  - **batch 内の分子間 attention を block-diagonal（分子内限定）に構造的に遮断**

入力:
  Zt    : (N, latent_dim)  ノイズ付き潜在変数
  cond  : (N, cond_dim)   ConditionalEncoder の Ci
  t     : (B,) float       時刻
  batch : (N,) int         各ノードのバッチ番号
  z_sc  : (N, latent_dim) self-conditioning（None → ゼロ）
  attn_bias : precompute_attn_inputs() の出力（パディング形式コンテキスト、省略時は内部計算）

出力:
  z1_pred : (N, latent_dim)  Z1 の予測（velocity）

パフォーマンスメモ:
  従来は全分子を長さ N の 1 シーケンスに連結し (H, N, N) の密 attention を計算していたため、
  計算・メモリが (Σnᵢ)² に膨れていた。本実装は分子ごとにパディングした (B, max_n, hidden) で
  attention を計算するため O(B·max_n²) に削減される（分子間 attention は別バッチ要素として
  構造的に遮断され、-1e9 の明示マスクは padding key のみに限定される）。

  attn_bias（グラフ距離 BFS）は CPU Python を含むため計算コストが高い。
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


class DiTBlock(nn.Module):
    """
    1 層の DiT ブロック。

    Pre-LN → F.scaled_dot_product_attention (Flash Attention) → FFN

    attn_ctx は precompute_attn_inputs() が返すパディング形式のコンテキスト dict:
      - 'gather_idx' : (B, max_n) long   各 (mol, local_pos) → flat ノード index
      - 'pad_mask'   : (B, max_n) bool   有効ノード = True
      - 'attn_mask'  : (B, H, max_n, max_n) もしくは (B, 1, 1, max_n) float
                       同分子内 = pos_bias 値（use_pos_bias=False なら 0）、
                       padding key = -1e9（softmax で実質ゼロ）

    分子間はそれぞれ別のバッチ要素として扱われるため、cross-molecule attention は
    構造的に発生しない（明示的な -1e9 マスクは padding key のみ）。
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
        attn_ctx: Optional[dict] = None,
    ) -> Tensor:
        """
        x        : (N, hidden_dim)
        attn_ctx : precompute_attn_inputs() の戻り値（パディング形式）。
                   None の場合は全ノードを 1 分子とみなした密 attention（後方互換）。
        """
        N = x.size(0)
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)   # (N, H, D)

        if attn_ctx is None:
            # フォールバック: 全ノードを 1 分子として密 attention
            q = q.permute(1, 0, 2).unsqueeze(0)   # (1, H, N, D)
            k = k.permute(1, 0, 2).unsqueeze(0)
            v = v.permute(1, 0, 2).unsqueeze(0)
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
            )   # (1, H, N, D)
            out = out.squeeze(0).permute(1, 0, 2).reshape(N, self.hidden_dim)
        else:
            gather_idx = attn_ctx['gather_idx']   # (B, max_n)
            pad_mask = attn_ctx['pad_mask']       # (B, max_n)
            attn_mask = attn_ctx['attn_mask']     # (B, H, max_n, max_n) or (B, 1, 1, max_n)
            B, max_n = gather_idx.shape

            # flat (N, H, D) → padded (B, max_n, H, D) → (B, H, max_n, D)
            # padding スロットは gather_idx=0（node 0）を指すが、query 行は破棄され、
            # key/value 列は attn_mask の -1e9 で無効化されるため汚染しない。
            qp = q[gather_idx].permute(0, 2, 1, 3)   # (B, H, max_n, D)
            kp = k[gather_idx].permute(0, 2, 1, 3)
            vp = v[gather_idx].permute(0, 2, 1, 3)

            out = F.scaled_dot_product_attention(
                qp, kp, vp,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )   # (B, H, max_n, D)

            out = out.permute(0, 2, 1, 3).reshape(B, max_n, self.hidden_dim)   # (B, max_n, hidden)
            # padded → flat (N, hidden) に戻す
            result = torch.zeros(N, self.hidden_dim, device=x.device, dtype=out.dtype)
            result[gather_idx[pad_mask]] = out[pad_mask]
            out = result

        x = x + self.out_proj(out)
        x = x + self.ffn(self.norm2(x))
        return x


class LatentDiT(nn.Module):
    """
    Latent Diffusion Transformer。

    PyG Batch のフラットなノード列を受け取り、
    batch テンソルに基づいて分子ごとにパディングした block-diagonal attention を適用する。

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
    ) -> dict:
        """
        分子ごとにパディングした attention コンテキストを 1 回計算して返す。

        戻り値 dict:
          'gather_idx' : (B, max_n) long   各 (mol, local_pos) → flat ノード index
                         （padding スロットは 0）
          'pad_mask'   : (B, max_n) bool   有効ノード = True
          'attn_mask'  : (B, H, max_n, max_n) もしくは (B, 1, 1, max_n) float
                         同分子内  : pos_bias 値（use_pos_bias=False なら 0）
                         padding key: -1e9（softmax で実質ゼロ）

        分子間は別のバッチ要素として構造的に遮断されるため、cross-molecule の
        -1e9 マスクは不要（padding key のみに限定）。

        Parameters
        ----------
        dist_mat : (total_N, total_N) int8 optional
            DataLoader ワーカーで事前計算済みのブロック対角距離行列。
            提供された場合は BFS をスキップして高速パスを使用。
            None の場合はオンザフライで BFS 計算（低速パス）。
        """
        device = batch.device
        H = self.n_heads

        if batch.numel() > 0:
            B = int(batch.max().item()) + 1
        else:
            B = 1

        # 各分子のノード index リスト（flat）
        node_lists = [(batch == b).nonzero(as_tuple=True)[0] for b in range(B)]
        counts = [nl.size(0) for nl in node_lists]
        max_n = max(counts) if counts else 0
        max_n = max(max_n, 1)

        gather_idx = torch.zeros(B, max_n, dtype=torch.long, device=device)
        pad_mask = torch.zeros(B, max_n, dtype=torch.bool, device=device)
        for b, nl in enumerate(node_lists):
            n_b = nl.size(0)
            if n_b > 0:
                gather_idx[b, :n_b] = nl
                pad_mask[b, :n_b] = True

        # padding key を無効化するバイアス: (B, max_n) → 有効=0, padding=-1e9
        pad_bias = torch.zeros(B, max_n, device=device).masked_fill(~pad_mask, -1e9)

        if self.pos_bias is not None:
            pos_bias_pad = torch.zeros(B, H, max_n, max_n, device=device)
            if dist_mat is not None:
                # 高速パス: 事前計算済み距離行列から分子ブロックを切り出す
                dm = dist_mat.long().to(device)
                for b, nl in enumerate(node_lists):
                    n_b = nl.size(0)
                    if n_b == 0:
                        continue
                    sub = dm[nl][:, nl]                  # (n_b, n_b)
                    oh = dist_to_onehot(sub)             # (n_b, n_b, 5)
                    lb = self.pos_bias(oh)               # (n_b, n_b, H)
                    pos_bias_pad[b, :, :n_b, :n_b] = lb.permute(2, 0, 1)
            elif edge_index is not None:
                # 低速パス: 分子ごとにオンザフライ BFS
                src, dst = edge_index
                for b, nl in enumerate(node_lists):
                    n_b = nl.size(0)
                    if n_b == 0:
                        continue
                    node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
                    node_mask[nl] = True
                    edge_mask = node_mask[src] & node_mask[dst]
                    # global ノード index → local (0..n_b-1) に remap
                    remap = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
                    remap[nl] = torch.arange(n_b, device=device)
                    local_ei = remap[edge_index[:, edge_mask]]   # (2, E_b)
                    dist_mat_b = compute_graph_distance(local_ei, n_b)
                    oh = dist_to_onehot(dist_mat_b)
                    lb = self.pos_bias(oh)
                    pos_bias_pad[b, :, :n_b, :n_b] = lb.permute(2, 0, 1)
            # pos_bias + padding マスクを統合
            attn_mask = pos_bias_pad + pad_bias.view(B, 1, 1, max_n)
        else:
            # pos_bias 無し: padding key マスクのみ（ヘッド・query 方向へブロードキャスト）
            attn_mask = pad_bias.view(B, 1, 1, max_n)

        return {
            'gather_idx': gather_idx,
            'pad_mask': pad_mask,
            'attn_mask': attn_mask,
        }

    def forward(
        self,
        zt: Tensor,            # (N, latent_dim)
        cond: Tensor,          # (N, cond_dim)
        t: Tensor,             # (B,) float  0〜1
        batch: Tensor,         # (N,) int
        edge_index: Optional[Tensor] = None,   # pos_bias 計算用（attn_bias 未提供時のみ使用）
        z_sc: Optional[Tensor] = None,         # self-conditioning
        attn_bias: Optional[dict] = None,      # precomputed attention context（毎回計算を避けるため）
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
