"""
デコード後の 3D 座標に対する幾何緩和（geometry relaxation）後処理。

生成配座の多くは微小な立体衝突（clash）や結合長の逸脱によって妥当性ゲート
（eval_ensemble.validity_of_conformer）を落ちる。本モジュールはデコード座標 x0 を
初期値として、clash + bond のヒンジエネルギーに「元座標へのアンカー項」を加えた
目的関数を autograd で数ステップ最小化する。

意図:
  - アンカー項 w_anchor*||x - x0||^2 が大域構造（torsion など主鎖のねじれ）を
    保持し、配座を元の折り畳みから大きく動かさない。
  - clash_loss / bond_range_loss のヒンジ項は、閾値を破った局所違反（食い込み・
    結合長逸脱）だけに勾配を与えて押し出す（範囲内のペアには 0 勾配）。
  - 結果として torsion 等の大域構造を保ったまま、局所的な妥当性違反だけを解消する。

これは純生成（モデルによるサンプリング）ではなく、あくまでデコード後の後処理で
ある点に注意。学習は不要で、既存チェックポイントの出力にそのまま適用できる。
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch_scatter import scatter_mean

from poly3d.model.geo_losses import enumerate_clash_pairs, guardrail_energy


def relax_coords(pos0: Tensor, edge_index: Tensor, dist_mat: Tensor,
                 atomic_nums: Tensor, batch: Optional[Tensor] = None,
                 ptr: Optional[Tensor] = None,
                 steps: int = 20, lr: float = 0.02,
                 w_clash: float = 1.0, w_bond: float = 1.0, w_anchor: float = 0.5,
                 clash_factor: float = 0.6, min_graph_dist: int = 3,
                 bond_lo: float = 0.7, bond_hi: float = 2.6) -> Tensor:
    """
    clash + bond のヒンジエネルギー（+ 元座標へのアンカー）を autograd 最小化して
    デコード座標の局所的な妥当性違反を解消する後処理。

    アンカー項が大域構造（torsion）を保持しつつ、clash / bond のヒンジ項が閾値を
    破った局所違反だけを押し出す。純生成ではなくデコード後の後処理である。

    Parameters
    ----------
    pos0 : (N, 3)  デコード座標。バッチ全体を縦連結した block-diagonal 形式で可
                   （分子境界は edge_index / dist_mat / batch / ptr が扱う）。
    edge_index : (2, E) int  結合エッジ（block-diagonal）。bond_range_loss 用。
    dist_mat   : (N, N) int  ブロック対角グラフ距離行列。clash_loss 用。
    atomic_nums : (N,) int  原子番号。RDKit で van der Waals 半径 rvdw を構築する。
    batch : (N,) int  各原子の所属分子インデックス。None なら単一分子として
                      torch.zeros(N, long) を内部生成する。
    ptr   : (B+1,) int  分子境界。clash_loss にそのまま渡す（None 可）。
    steps : int  Adam の最適化ステップ数。
    lr    : float  Adam 学習率。
    w_clash / w_bond / w_anchor : float  各項の重み。
    clash_factor / min_graph_dist : clash_loss と同じ閾値（eval と一致させる）。
    bond_lo / bond_hi : bond_range_loss と同じ範囲（eval と一致させる）。

    Returns
    -------
    x : (N, 3)  緩和後の座標（勾配なし、pos0 と同じ device/dtype/shape）。
    """
    device = pos0.device
    N = pos0.size(0)

    if batch is None:
        batch = torch.zeros(N, dtype=torch.long, device=device)

    # rvdw: 原子番号 → RDKit の van der Waals 半径（eval の _PT.GetRvdw と一致させる）。
    # 原子ごとに GetRvdw を逐次呼ぶ代わりに、バッチ中に実在する元素番号のみで
    # ルックアップ表を 1 度構築し、index gather で全原子分を得る（高々 ~119 元素）。
    # GetRvdw 呼び出しは「ユニーク元素数」回のみで、逐次版と数値は完全一致する。
    from rdkit import Chem
    pt = Chem.GetPeriodicTable()
    uniq_z = torch.unique(atomic_nums).tolist()               # 同期 1 回
    max_z = int(max(uniq_z)) if uniq_z else 0
    lut = torch.zeros(max_z + 1, dtype=torch.float32)         # CPU 上で構築
    for z in uniq_z:
        lut[int(z)] = pt.GetRvdw(int(z))
    rvdw = lut.to(device)[atomic_nums.long()]                 # (N,) index gather

    # clash ペアの厳密列挙はトポロジ（結合・グラフ距離）不変なので Adam ループの
    # 外で 1 度だけ行い、毎ステップは座標更新のみに絞る（triu 再列挙と分子数 B 回の
    # Python ループを除去）。数値は毎ステップ列挙する場合と完全一致する。
    clash_pairs = enumerate_clash_pairs(
        dist_mat, batch, ptr=ptr, min_graph_dist=min_graph_dist
    )

    # 最適化変数（pos0 は勾配なしの参照点として保持し、x を動かす）
    x = pos0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)

    with torch.enable_grad():
        for _ in range(steps):
            # clash + bond ガードレール（列挙済みペアを使い回す）。per-term 重みは
            # guardrail_energy が受け取る。
            guard = guardrail_energy(
                x, dist_mat, rvdw, batch, ptr, edge_index,
                clash_factor=clash_factor, clash_min_graph_dist=min_graph_dist,
                bond_lo=bond_lo, bond_hi=bond_hi, exact=True,
                w_clash=w_clash, w_bond=w_bond, clash_pairs=clash_pairs,
            )
            # アンカー項も clash/bond と同じ「分子ごと scatter_mean → 分子間平均」に
            # 統一する。サイズ不均一バッチでアンカー vs 押し出しの相対強度が分子サイズに
            # 依存してブレる問題を解消する（等分子重み）。
            anchor_per_node = ((x - pos0) ** 2).sum(-1)             # (N,)
            anchor = scatter_mean(anchor_per_node, batch, dim=0).mean()

            loss = guard + w_anchor * anchor
            opt.zero_grad()
            loss.backward()
            opt.step()

    return x.detach()
