# 大域折れ問題の根本解決 設計書

作成: 2026-07-07 / 対象: Structural VAE（Encoder/Decoder）

## 1. 根本原因（コードで確定）

`egnn.py` の座標更新は結合エッジ限定の局所メッセージパッシング:

```
Δx_i = Σ_{j∈N(i)} (x_i - x_j) · φ_x(m_ij) / |N(i)|      # N(i) = 結合近傍のみ
```

- `edge_index` は化学結合のみ。K 層 EGNN の受容野は **K 結合ホップ**。
- 240 原子のポリマー繰返し単位はグラフ直径が数十〜100 結合 → 分子両端は情報交換不能。
- `vae.py` の `VAEDecoder.init_pos` は **原子ごと独立の MLP**。大域足場が最初から存在せず、折り畳みを純ローカル更新だけで作る → **torsion-flip（1本のねじれのズレで分子が折れる）で大域構造が崩壊**。

### 実験的裏付け（2026-07-07 過学習テスト, `runs/overfit/`）
- 小分子は rmsd 0.06Å で完璧再構築 → アーキ健全。
- 大分子（240+）は過学習でも **大域 Kabsch rmsd ~0.46Å で頭打ち、p90 が 1.2〜3.2Å（torsion-flip 外れ値）**。
- **深さ 4→8 も幅 128→256 も大域を解けない**（局所 bond は改善するが大域 rmsd は横ばい）。
- → スケーリングでは解けない。**大域通信機構が必要**という結論。

## 2. 設計目標と原理

**目標**: 分子サイズによらず、1 ブロックで全原子が大域情報を交換でき、座標配置を大域的に協調できるようにする。SE(3)（正確には E(3)）同変性は維持する。

**原理**: 局所 EGNN（局所幾何は既に良好）に、**同変な大域アテンション**を交互に挟む。アテンション重みは不変量から計算し、座標更新は相対ベクトルの重み付き和で行うため同変性を保つ。既存の `GraphDistanceBias` と DiT のバッチ化 attention を再利用する。

## 3. アーキテクチャ: Equivariant Graph Transformer (EGT) ブロック

各ブロックは「局所チャンネル」＋「大域チャンネル」の 2 経路。

### 3.1 局所チャンネル（既存 EGNNLayer をそのまま）
結合エッジ沿いの `h`, `x` 更新。局所幾何（bond/angle）担当。変更なし。

### 3.2 大域チャンネル（新規）
全原子ペア（分子内、パディング式バッチ）に対するアテンション。

**アテンション重み（すべて不変量から）:**
```
logit_ij = (W_q h_i)·(W_k h_j)/√d
           + b_graph(graph_dist_ij)      # GraphDistanceBias を再利用（静的・前計算）
           + g(d_ij²)                     # 3D 距離バイアス（動的・層ごと再計算）
a_ij = softmax_j(logit_ij)                # (H, N, N)、分子間は block-diagonal で遮断
```

**スカラー更新（不変・大域情報ルーティング）:**
```
h_i ← h_i + Σ_j a_ij · (W_v h_j)
```

**座標更新（同変・大域協調）:**
```
Δx_i = Σ_j a_ij · φ_x(m_ij) · (x_i - x_j)     # m_ij はエッジ MLP 出力（不変）
x_i ← x_i + clamp_scale(Δx_i)                  # ノルムベースの一様スケール（egnn.py と同方式）
```

- `a_ij`（不変）× `φ_x`（不変スカラー）× `(x_i−x_j)`（同変）→ **Δx_i は同変**。
- 全ペア和なので受容野 = 分子全体（1 ブロックで大域）。

### 3.3 配置
- **Encoder と Decoder の両方**に導入（Encoder が大域構造を per-atom 潜在に集約し、Decoder が復元）。
- 全層を EGT にする必要はない。例: `[EGNN, EGNN, EGT, EGNN, EGT, EGNN]` のように **数層おきに 1 EGT** を挟む（コスト削減）。
- 分子間遮断・パディングは DiT の `precompute_attn_inputs()`（`gather_idx`/`pad_mask`/`attn_mask`）を流用。`dist_mat` は既に dataset が前計算済み。

## 4. デコーダ初期座標の大域整合化

現状の per-atom 独立 init を、**トポロジー由来の大域足場**に置換（任意だが推奨）:

- **グラフ距離 MDS**: `dist_mat`（結合ホップ距離）を古典的 MDS で 3D 埋め込み → 決定論的な大域スキャフォールドを初期値に。分子端どうしが最初から離れて配置され、EGT が微修正するだけで済む。
- MDS は分子ごと 1 回、CPU で安価（N≤284）。dataset の前計算に含めてもよい。
- 代替: 現行 MLP init のまま EGT の大域座標更新に任せる（実装は軽いが最適化はやや不利）。

## 5. 損失: マルチスケール距離幾何損失

大域折れは torsion-flip で **Kabsch rmsd が不連続**＝勾配が病的。アーキで容量を与えても、滑らかな大域教師信号が要る。

- **local_distmat（既存, 2026-07-01 実装）**: 近接ペア（<cutoff）距離。局所担当。
- **long-range distmat（新規）**: グラフ距離 ≥ 閾値のペアからサンプリングした部分集合の距離を Huber 損失で教師化。回転不変・連続で **torsion-flip の不連続を回避**しつつ大域構造を直接監督。全 (N,N) は使わず sampling で O(N·k)。
- 合成: `L_pos = w_local · L_local_distmat + w_global · L_longrange_distmat`（+ 任意で Kabsch を小重みで補助）。
- `--pos_loss_type multiscale_distmat` として追加（既存の kabsch/distmat/local_distmat と併存）。

**要点: アーキ（EGT）が大域協調の"容量"、損失（マルチスケール距離）が大域の"滑らかな勾配"。両輪で初めて解ける。**

## 6. 同変性の担保

- アテンション重み `a_ij`、`φ_x`、`g(d²)`、`b_graph` はすべて不変量（`h`, `|x_i−x_j|`, グラフ距離）から計算 → 回転で不変。
- 座標更新は相対ベクトル `(x_i−x_j)` の不変重み付き和 → 回転同変・並進不変（差分のため）。
- 座標クランプはノルムベースの一様スケール（`egnn.py:131-134` と同じ。成分ごと clamp は禁止）。
- 出力は重心ゼロ化（既存 decoder と同じ）。
- → EGNN と同じ E(3) 同変性を維持。

## 7. コストと実現性（RTX 4060 Ti 16GB, N≤284）

- 大域 attention は分子ごと O(N²)。N=284 で ~80k ペア。DiT が既に同規模の密 attention を回せている（`F.scaled_dot_product_attention`, Flash 対応）。
- EGT を数層おきに 1 回に限れば追加コストは局所 EGNN の数割増程度。
- パラメータ増: 大域チャンネルの Q/K/V/φ_x で 1 EGT あたり概ね局所 1 層と同オーダー。

## 8. 実装計画（ファイル別）

| ファイル | 変更 |
|---------|------|
| `egnn.py` | `EGTLayer`（局所 EGNNLayer + 大域 attention head）を新規追加。既存 `EGNNLayer` は温存 |
| `pos_bias.py` | `GraphDistanceBias` 流用。3D 距離バイアス `g(d²)` を EGT 内に小 MLP で追加 |
| `vae.py` | `VAEEncoder`/`VAEDecoder` の EGNN スタックを「EGNN と EGT の交互配置」に変更。`egt_every`（例:2）引数追加。init を MDS スキャフォールド化（任意） |
| `geo_losses.py` | `longrange_distance_loss()`（サンプリング版）追加 |
| `vae_loss.py` | `multiscale_distmat` 経路を追加 |
| `builder.py` / `train.py` | `--egt_every`, `--pos_loss_type multiscale_distmat`, `--w_global` 等の引数配線 |
| `dataset.py` | （MDS 採用時）init スキャフォールドを前計算して pickle に含める（後方互換フォールバックあり） |

`dist_mat` はデータ側で前計算済み・collate 済み（`dataset.py`）なので、attention バイアスへの供給は追加コスト小。

## 9. 検証プロトコル（過学習ハーネス流用）

`runs/overfit/` の枠組みで安価に効果測定:

1. **大分子64セット（`tiny_large.lmdb`, 130–271原子）を EGT で過学習**（beta0/lr3e-4/3000ep）。
2. `eval_by_size.py` + `evaluate_vae.py` でサイズ別・局所/大域を比較。
3. **成功基準**:
   - **240+ 帯の大域 Kabsch rmsd が現行 ~0.46Å から明確に低下**（目標 <0.25Å）。
   - **p90 の壊滅的外れ値（1.2〜3.2Å）が解消**（torsion-flip が消える）。
   - 局所指標は現行同等以上を維持。
4. 効けば subset 実データで再ベースライン（幅256系 + EGT + multiscale 損失）へ。

比較対象は既存の `large_n64_long`(深4)/`large_n64_deep8`/`large_n64_wide256` で、EGT が「深さ・幅では動かなかった 240+ 大域」を動かせるかが判定点。

## 10. フェーズ分けとフォールバック

- **Phase A（最小・高効果狙い）**: EGT を Decoder のみに、`egt_every=2` で導入 ＋ `multiscale_distmat` 損失。まず大分子過学習で検証。
- **Phase B**: 効果確認後、Encoder にも導入し MDS init を追加。
- **フォールバック（コスト過大時）**: 全ペア attention の代わりに **少数の潜在グローバルノード（inducing points, M~16）** による O(N·M) 通信、または **階層プーリング**（骨格フラグメント単位で大域推論→ブロードキャスト）。表現力は落ちるが安価。

## 付記
- これは VAE（Stage 1）の設計。DiT（Stage 2）は既に大域 attention を持つため、VAE の再構築天井が全体の天井。まず VAE を直す。
- `flow_matching.py` の Euler 符号など既知の別バグ（レビュー済み・修正コミット済み）とは独立の課題。
