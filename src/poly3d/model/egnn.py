"""
EGNN (Equivariant Graph Neural Network) 実装

参考: Satorras et al., "E(n) Equivariant Graph Neural Networks" (ICML 2021)

SE(3)-同変なメッセージパッシングを実装する。
  - ノード特徴量 h: 任意の不変量スカラー
  - 座標 x: 回転・反射・並進に対して同変
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter

from poly3d.model.pos_bias import GraphDistanceBias, dist_to_onehot


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

        # scatter 後にノードごとのベクトルノルムでスケールダウン
        # （成分ごと clamp は回転同変性を破壊するため使用不可。
        #  一様スケール R(αv)=αR(v) はベクトルに対して回転同変。）
        if self.coord_clamp > 0:
            norm = coord_delta.norm(dim=-1, keepdim=True)        # (N, 1)
            scale = (self.coord_clamp / norm.clamp(min=1e-8)).clamp(max=1.0)
            coord_delta = coord_delta * scale

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


class EGTLayer(nn.Module):
    """
    Equivariant Graph Transformer レイヤ（設計書 §3.2）。

    2 経路構成:
      1. 局所チャンネル: 既存 EGNNLayer をそのまま内部に持ち、結合エッジ沿いに
         h, x を更新（局所幾何 = bond/angle 担当）。
      2. 大域チャンネル: 分子内の全原子ペアに対する同変アテンション。
         受容野 = 分子全体（1 ブロックで大域通信）。

    大域チャンネルのアテンション重み（すべて不変量から計算）:
        logit_ij = (W_q h_i)·(W_k h_j)/√d
                   + b_graph(graph_dist_ij)   # GraphDistanceBias 流用（dist_mat 任意）
                   + g(d_ij²)                  # 3D 距離二乗のヘッド別バイアス
        a_ij = softmax_j(logit_ij)             # 分子間はブロック対角で遮断

    スカラー更新（不変・大域情報ルーティング）:
        h_i ← LayerNorm( h_i + W_o Σ_j a_ij (W_v h_j) )

    座標更新（E(3) 同変・大域協調）:
        Δx_i = Σ_j (mean_h a_ij) · φ_x(m_ij) · (x_i − x_j)
        x_i ← x_i + clamp_scale(Δx_i)          # ノルムベース一様スケール（EGNNLayer と同方式）

    a_ij（不変）× φ_x（不変スカラー）× (x_i−x_j)（同変）→ Δx_i は同変。

    分子間 attention は batch から作るブロック対角パディング（DiT の
    precompute_attn_inputs 式）で構造的に遮断する（別分子ペアは softmax から除外）。

    Parameters
    ----------
    hidden_dim    : ノード特徴量次元（n_heads で割り切れること）
    edge_feat_dim : 局所チャンネル（EGNNLayer）のエッジ特徴量次元
    n_heads       : 大域アテンションのヘッド数
    coord_clamp   : 座標更新のノルム上限（一様スケール）
    coord_msg_dim : 座標メッセージ φ_x の内部次元（None → hidden_dim//2）
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_feat_dim: int,
        n_heads: int = 8,
        act: nn.Module | None = None,
        norm_coord: bool = True,
        residual: bool = True,
        coord_clamp: float = 10.0,
        coord_msg_dim: Optional[int] = None,
    ):
        super().__init__()
        assert hidden_dim % n_heads == 0, \
            f'hidden_dim({hidden_dim}) は n_heads({n_heads}) で割り切れる必要があります'
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.coord_clamp = coord_clamp

        act = act or _silu()

        # ─── 局所チャンネル（既存 EGNNLayer を温存して内部利用） ───
        self.local = EGNNLayer(
            hidden_dim=hidden_dim,
            edge_feat_dim=edge_feat_dim,
            act=_silu(),
            norm_coord=norm_coord,
            residual=residual,
            coord_clamp=coord_clamp,
        )

        # ─── 大域チャンネル: Q / K / V / 出力投影 ───
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_out = nn.Linear(hidden_dim, hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)

        # b_graph（グラフ距離バイアス）: GraphDistanceBias を流用（dist_mat 提供時のみ使用）
        self.graph_bias = GraphDistanceBias(n_heads)

        # g(d²): 3D 距離二乗 → ヘッドごとのアテンションバイアス（層ごとに動的再計算）
        self.dist_bias = nn.Sequential(
            nn.Linear(1, hidden_dim),
            _silu(),
            nn.Linear(hidden_dim, n_heads),
        )

        # 座標重み φ_x（不変スカラー）。
        # pairwise C 次元テンソル (B, max_n, max_n, C) の材料化は破綻的コストのため撤廃。
        # 座標更新 Δx_i = Σ_j w_ij (x_i − x_j) は w_ij が不変量なら E(3) 同変が保たれる。
        # w_ij = mean_h(a_ij) · g(d_ij²) とし、g は d² スカラーのみを受ける軽量 MLP。
        # coord_msg_dim は g の隠れ次元として流用（None → max(hidden//2,16)）。
        C = coord_msg_dim if coord_msg_dim is not None else max(hidden_dim // 2, 16)
        self.coord_g = nn.Sequential(
            nn.Linear(1, C),
            _silu(),
            nn.Linear(C, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # 座標重み最終層を 0 初期化（学習初期の座標爆発を防ぐ）
        nn.init.zeros_(self.coord_g[-1].weight)
        nn.init.zeros_(self.coord_g[-1].bias)

    def forward(
        self,
        h: Tensor,           # (N, hidden_dim)
        x: Tensor,           # (N, 3)
        edge_index: Tensor,  # (2, E)
        edge_attr: Tensor,   # (E, edge_feat_dim)
        batch: Optional[Tensor] = None,  # (N,) バッチ割り当て
        dist_mat: Optional[Tensor] = None,  # (total_N, total_N) グラフ距離（任意）
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        h_new : (N, hidden_dim)  更新後ノード特徴量（回転不変）
        x_new : (N, 3)           更新後座標（E(3) 同変）

        dist_mat が None の場合は b_graph 項を省略し、3D 距離バイアス g(d²) と
        batch ブロック対角マスクのみで動作する（後方互換）。
        """
        # ─── 1. 局所チャンネル（結合エッジ沿い） ───
        h, x = self.local(h, x, edge_index, edge_attr, batch)

        # ─── 2. 大域チャンネル（分子内全ペア・パディング式） ───
        N = h.size(0)
        device = h.device
        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=device)
        B = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        # 分子ごとのノード index → パディング（DiT precompute_attn_inputs 式）
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

        # Q/K/V を (B, H, max_n, d) にパック
        q = self.q_proj(h).reshape(N, self.n_heads, self.head_dim)
        k = self.k_proj(h).reshape(N, self.n_heads, self.head_dim)
        v = self.v_proj(h).reshape(N, self.n_heads, self.head_dim)
        qp = q[gather_idx].permute(0, 2, 1, 3)   # (B, H, max_n, d)
        kp = k[gather_idx].permute(0, 2, 1, 3)
        vp = v[gather_idx].permute(0, 2, 1, 3)

        xp = x[gather_idx]                        # (B, max_n, 3)

        # content logit（不変）
        logit = torch.matmul(qp, kp.transpose(-1, -2)) / math.sqrt(self.head_dim)  # (B,H,max_n,max_n)

        # d_ij²（回転・並進不変）
        diff = xp[:, :, None, :] - xp[:, None, :, :]           # (B, max_n, max_n, 3) = x_i - x_j
        dist_sq = diff.pow(2).sum(-1).clamp(max=_DIST_SQ_MAX)  # (B, max_n, max_n)

        # g(d²) バイアス（ヘッドごと）
        g_bias = self.dist_bias(dist_sq.unsqueeze(-1))         # (B, max_n, max_n, H)
        logit = logit + g_bias.permute(0, 3, 1, 2)

        # b_graph（任意入力）: dist_mat があればグラフ距離バイアスを加算
        if dist_mat is not None:
            dm = dist_mat.long().to(device)
            gb = torch.zeros(B, self.n_heads, max_n, max_n, device=device, dtype=logit.dtype)
            for b, nl in enumerate(node_lists):
                n_b = nl.size(0)
                if n_b == 0:
                    continue
                sub = dm[nl][:, nl]                 # (n_b, n_b)  分子ブロックを切り出す
                oh = dist_to_onehot(sub)            # (n_b, n_b, 5)
                lb = self.graph_bias(oh)            # (n_b, n_b, H)
                gb[b, :, :n_b, :n_b] = lb.permute(2, 0, 1).to(gb.dtype)
            logit = logit + gb

        # padding key を softmax から除外（分子間遮断はブロック構造で自動達成）
        pad_bias = torch.zeros(B, max_n, device=device, dtype=logit.dtype).masked_fill(~pad_mask, -1e9)
        logit = logit + pad_bias.view(B, 1, 1, max_n)

        a = torch.softmax(logit, dim=-1)   # (B, H, max_n, max_n)

        # ─── スカラー更新（不変） ───
        attn_out = torch.matmul(a, vp)     # (B, H, max_n, d)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, max_n, self.hidden_dim)
        attn_out = self.attn_out(attn_out)  # (B, max_n, hidden)
        h_delta = torch.zeros(N, self.hidden_dim, device=device, dtype=attn_out.dtype)
        h_delta[gather_idx[pad_mask]] = attn_out[pad_mask]
        h_new = self.node_norm(h + h_delta)

        # ─── 座標更新（同変） ───
        # w_ij = mean_h(a_ij) · g(d_ij²)。g は d² スカラーのみを受ける軽量 MLP。
        # pairwise C 次元テンソル (B,max_n,max_n,C) の材料化を避け、
        # (B,max_n,max_n) スカラーのみで完結する（大分子でも軽量）。
        # w_ij は不変量なので Δx_i = Σ_j w_ij (x_i−x_j) は E(3) 同変を保つ。
        a_mean = a.mean(dim=1)             # (B, max_n, max_n)  ヘッド平均のアテンション
        g = self.coord_g(dist_sq.unsqueeze(-1)).squeeze(-1)   # (B, max_n, max_n)  φ_x(d²)（不変スカラー）
        w = a_mean * g                     # (B, max_n, max_n)  padding key は a_mean≈0 で寄与消失
        dx = (w.unsqueeze(-1) * diff).sum(dim=2)   # (B, max_n, 3)  Σ_j w_ij (x_i - x_j)

        # ノルムベース一様スケールで clamp（成分ごと clamp は同変性を破壊するため不可）
        if self.coord_clamp > 0:
            norm = dx.norm(dim=-1, keepdim=True)
            scale = (self.coord_clamp / norm.clamp(min=1e-8)).clamp(max=1.0)
            dx = dx * scale

        dx_flat = torch.zeros(N, 3, device=device, dtype=dx.dtype)
        dx_flat[gather_idx[pad_mask]] = dx[pad_mask]
        x_new = x + dx_flat

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
