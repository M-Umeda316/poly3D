# デコーダ初期座標の大域整合化（グラフ距離 MDS スキャフォールド）実装計画

作成: 2026-07-07 / 対象: Structural VAE の VAEDecoder / 前提: `docs/design_global_folding.md` の §4・§10 Phase B の未着手項目

## 0. 位置づけ（一行）

デコーダの初期座標を「原子ごと独立の MLP」から「トポロジー由来の**大域足場（MDS 埋め込み）＋ 小さな学習 MLP 補正**」に置換し、大分子の折り畳みを最初から大域整合した状態から EGT が微修正できるようにする。**現状未実装**（`vae.py:143-154,200` は per-atom MLP のみ）。

## 1. 動機

- 過学習テストで確定: 大分子（240+）の大域 Kabsch rmsd は深さ・幅では頭打ち。EGT（enc+dec）が giant tail を叩けたが目標には未達で、主因は最適化予算。init が原子ごと独立＝**最初に大域足場が無い**ため、EGT が全折り畳みを更新だけで作る必要があり、torsion-flip の局所解に落ちやすい。
- MDS init は**決定論的な大域足場**を初期値に与える別アプローチ（EGT=更新機構で大域整合、MDS=初期値で大域整合）。両者は相補的で、EGT が残す 240+ tail に安価に効く可能性。

## 2. 重要な制約（コードで確定した落とし穴）

**既存 `dist_mat` は MDS にそのまま使えない。**

- `dataset.py:147` / `pos_bias.py:30` の `compute_graph_distance(edge_index, n, max_dist=4)` は**ホップ距離を 4 でクランプ**する（値域 0..4、`4 = far`）。用途は EGT/DiT の attention onehot バイアス（`dist_to_onehot`, 5 チャンネル）。
- collate（`dataset.py:207`）は分子間 off-block も `4` で埋める。
- → この行列で MDS を回すと、5 ホップ以上離れた原子ペアがすべて距離 4 に潰れ、**退化した塊状の埋め込み**になり大域足場として無意味。

**結論: MDS には非クランプの完全グラフ距離（分子内、値域 0..直径 ~50-100）が別途必要。** `shortest_path` は内部で完全 float 距離を計算済み（`pos_bias.py:67` の `dist_f`）で、その後 clamp しているだけなので、非クランプ版ヘルパを 1 本足せばよい。

## 3. アルゴリズム（古典 MDS / Torgerson）

分子ごと、完全グラフ距離 `D (n×n, ホップ)` から:

```
D2   = D ** 2                                   # 二乗距離
J    = I - (1/n) · 1 1ᵀ                          # 中心化行列
B    = -0.5 · J · D2 · J                          # 二重中心化グラム行列
λ, V = eigh(B)                                    # 昇順固有値・固有ベクトル
上位3個の正固有値 λ1≥λ2≥λ3 とベクトル v1,v2,v3 を取り
X    = [ v1·√λ1 , v2·√λ2 , v3·√λ3 ]  ∈ R^{n×3}   # 3D 埋め込み
```

### 3.1 物理スケールへの整合（ホップ→Å）
X はホップ単位（隣接≈1）。EGNN/EGT が Å で動くため、**結合原子間の埋め込み距離が平均 ~1.5Å になるよう一様スケール**する:

```
s = 1.5 / mean_{(i,j)∈bond} ||X_i - X_j||        # bond = edge_index の実結合
X ← s · X
```

分母が 0（結合ゼロ＝単原子）の場合は s=1。

### 3.2 エッジケース
- **正固有値が 3 未満**（線状/微小分子）: 不足次元は 0 で埋め、その後 `std=1e-2` の微小乱数を全体に加算（EGNN の d²=0 縮退回避、既存 init と同思想）。
- **非連結成分**（想定外だが防御的に）: `shortest_path` の `inf` を `(有限最大 + 1)` に置換してから MDS。build_dataset は基本的に連結分子のみ通す。
- **n ≤ 1**: そのまま零ベクトル（+微小乱数）。
- MDS は best rank-3 近似なので、グラフ距離が 3D に厳密に埋まらなくても「大域的な広がり」を与えられれば十分（EGT が精修正）。

### 3.3 同変性・決定性
- MDS 埋め込みは回転・鏡映・並進・固有ベクトル符号について任意。**が、デコーダの損失（Kabsch RMSD / distmat / local / longrange）はすべて回転・並進不変**なので、足場の絶対姿勢の任意性は損失に影響しない。→ **同変性は破れない**。
- 足場は入力座標に一切依存しない純トポロジー量なので、GT の回転に対して不変（＝EGNN の同変性関係を壊さない）。
- 固有ベクトル符号のフリップで足場が鏡映し得るが、これも Kabsch（reflection は含まないが）/ distmat では距離が保存されるため問題なし。決定論性のため `eigh`（対称・決定的）を使う。

## 4. 計算場所の決定 — DataLoader ワーカー（collate 前）

3 案比較の結論: **ワーカーの `__getitem__` で分子ごと計算し LRU キャッシュ、collate で `(total_N,3)` に連結**。

- 理由: 重い部分（`shortest_path` 非クランプ + `eigh`）を GPU/学習ホットパスから外し、8 ワーカーで並列化。既存 `_get_topology`（`dataset.py:120-160`）の LRU キャッシュ機構をそのまま流用でき、過学習（固定分子）では初回のみ計算→以降キャッシュ。
- 後方互換: フラグ OFF なら計算しない。既存 lmdb をそのまま使える（再前処理不要）。
- 将来: `build_dataset.py --precompute_topology` と同様に lmdb へ焼き込むオプションも追加可能（`init_scaffold` を pickle に格納、ワーカー fallback あり）。まずはワーカー計算で十分。

コスト: n≤284 で `eigh` は数 ms、`shortest_path` は C 実装で高速。学習全体に対して無視できる。

## 5. ファイル別変更（正確な差分）

| ファイル | 変更 | 具体 |
|---------|------|------|
| `pos_bias.py`（or 新規 `mds.py`） | 非クランプ距離 + MDS ヘルパ追加 | `def mds_init_coords(edge_index, num_nodes, bond_scale=1.5) -> Tensor(n,3)`。内部で `shortest_path(unweighted, directed=False)` を非クランプで取得 → 二重中心化 → `torch.linalg.eigh` → 上位3 → §3.1 スケール → §3.2 微小乱数。CPU で完結（ワーカー） |
| `dataset.py` | 足場を計算・キャッシュ・collate | (1) `__init__` に `mds_init: bool=False`。(2) `_get_topology` 隣に `_get_scaffold(idx, edge_index, n)`（LRU キャッシュ、`mds_init_coords` 呼び出し）。(3) `__getitem__`（`dataset.py:113` 付近）で `mds_init` 時 `kwargs['init_scaffold']=scaffold`。(4) `collate_fn`（`dataset.py:168`）で `init_scaffold` を `pos` と同様に縦連結（オフセット不要、単純 cat）→ `pyg_batch.init_scaffold=(total_N,3)`。`make_dataloader` に `mds_init` 引数追加 |
| `vae.py` | デコーダ init のブレンド | `VAEDecoder.forward`（`vae.py:179`）に `init_scaffold: Optional[Tensor]=None` 追加。`x0` 生成（`vae.py:200`）を「`base = init_scaffold if given else 0`、`x0 = base + self.init_pos(feat)`」に変更。以降の重心ゼロ化はそのまま。`StructuralVAE.decode`（`vae.py:305-312`）/`forward`（`vae.py:315-337`）にも `init_scaffold` を通す |
| `train.py` | バッチ→モデル配線 + フラグ | (1) `--mds_init`（action store_true）追加。(2) VAE 呼び出し（`train.py:443-446`）に `init_scaffold=getattr(batch, 'init_scaffold', None)` 追加。(3) `make_dataloader(..., mds_init=args.mds_init)` を train/val 両方に。(4) DiT stage は VAE 凍結・デコード時に足場が要るので `sample`/評価経路も同様に（後述） |
| `builder.py` | 変更ほぼ不要 | モデルは `init_scaffold` を「渡されれば使う」だけ。`build_vae` はアーキ引数に MDS 依存なし。DiT の凍結 VAE でデコードする箇所に足場を渡す配線のみ確認 |
| `evaluate_vae.py` / `eval_by_size.py` | 評価時も足場を渡す | `make_dataloader(mds_init=...)` とデコード呼び出しに `init_scaffold` を追加。チェックポイントの `args['mds_init']` を見て自動 ON（`_load_models` が margs を持つのと同様に）。**MDS で学習したモデルは評価も MDS 足場必須**（init 分布が変わるため）|

### 5.1 ブレンド方式の根拠
`x0 = init_scaffold + init_pos(feat)`（足場＋小 MLP 補正）を採用:
- 足場が大域構造、MLP（`std=0.01` 初期化のまま）が per-atom 微補正 + 学習余地。
- 足場が `None`（フラグ OFF / 旧 lmdb / 単原子）なら `x0 = init_pos(feat)` に自然縮退＝**完全後方互換**。
- 全ゼロ回避（d²=0 縮退）は足場の非ゼロ性 or MLP 微小乱数で担保。

## 6. 検証プロトコル（過学習ハーネス流用、EGT と同じ土俵）

`docs/design_global_folding.md §9` と同じく `runs/overfit/` で安価に効果測定。**MDS の効果は EGT とは独立に切り分ける**。

1. **対照群（既存資産）**: `runs/overfit/large_n64_base_bs4`（非EGT/非MDS）, `large_n64_egt_encdec`（EGT/非MDS）。
2. **新規 2 条件**（`tiny_large.lmdb` 64 分子, beta0/lr3e-4/3000ep/bs4, max_atoms300）:
   - **MDS 単独**: `--mds_init`（EGT なし） → 「足場だけ」で 240+ が動くか。
   - **MDS + EGT**: `--mds_init --egt_every 2 --enc_egt_every 2` → 相補効果。
3. `eval_by_size.py`（val でなく暗記対象で可、まず暗記能力）でサイズ別比較。
4. **成功基準**（240+ 帯、n=5 なので p90 と成功率を重視）:
   - 大域 Kabsch rmsd 中央値が対照から明確低下（EGT の 0.162 / bs4 非EGT の 0.296 と比較）。
   - **収束が速い**（同 3000ep で val_pos がより低く／早く）。足場効果は特に序盤の収束加速に出るはず。
   - p90 外れ値（torsion-flip）が悪化しないこと。
5. 効けば実データ subset 汎化テスト（現行 gen_v1 と同枠）に MDS 条件を追加。

**判定点**: 「初期値の大域足場」が「更新機構（EGT）」や「最適化予算（ステップ数）」に対してどれだけ独立に効くか。特に MDS 単独が非EGT対照を上回れば、安価なレバーとして価値大。

## 7. 段階的実装チェックリスト

1. `mds_init_coords()` を実装（`pos_bias.py` か新規 `mds.py`）。**単体テスト**: (a) 既知の鎖状分子で bonded 距離 ~1.5Å、(b) 環状分子が平面的に開く、(c) 回転不変性は不要だが `eigh` 決定性を確認、(d) n=1/結合ゼロで例外を出さない。
2. `dataset.py`: `_get_scaffold` + LRU + `__getitem__`/`collate_fn`/`make_dataloader` 配線。**確認**: バッチの `init_scaffold.shape == (total_N,3)`、分子順が `pos` と一致。
3. `vae.py`: `VAEDecoder.forward` の `x0` ブレンド + `decode`/`forward` の引数通し。**確認**: 足場 None で旧数値一致（後方互換）、足場ありで `x0` が足場中心。
4. `train.py`: `--mds_init` + 呼び出し配線（train/val dataloader 両方）。
5. `evaluate_vae.py`/`eval_by_size.py`: `margs.mds_init` 自動 ON + デコードに足場配線。
6. `py_compile` 全ファイル + 過学習スモーク（数十 step）で shape/NaN/OOM チェック。
7. §6 の検証 2 条件を Start-Process デタッチで実行、`eval_by_size` で判定。

## 8. リスクとフォールバック

- **リスク: 足場が悪さをする**（MDS の平面性が 3D 折り畳みとズレて EGT を誤誘導）。→ ブレンド式なので MLP 補正で吸収可能。それでも劣化するなら `x0 = α·scaffold + init_pos`（α<1 の学習可能 or 固定ゲート）を導入。
- **リスク: DiT stage（Stage2）でのデコード**。DiT は VAE を凍結してデコードするため、推論 `sample.py` でも足場が必要。トポロジーのみから計算可能（座標不要）なので SMILES→edge_index→`mds_init_coords` で推論時も生成できる。**配線漏れに注意**（学習は batch 経由、推論は明示計算）。
- **フォールバック**: 効果が薄ければ MDS init は不採用とし、EGT + 最適化予算に集中。実装は後方互換なので撤去も容易（フラグ OFF）。

## 9. 現行タスクとの関係

- いま走行中の**汎化テスト（gen_v1: 非EGT vs enc+dec EGT, kabsch, 非MDS）には無関係**。MDS はその結果を見た上での次の一手。
- 実装自体は gen_v1 の走行中に並行して進めても学習プロセスに影響しない（別ファイル・別実行）。ただし MDS 検証ジョブは GPU を使うので、gen_v1 完了後に実行する。
