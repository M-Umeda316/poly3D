# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

polyGen（arXiv:2504.17656）インスパイアのポリマー3D構造生成モデル。
PolyOmics（RadonPy 古典 MD の非晶質構造 DB、HF `yhayashi1986/PolyOmics`）を学習データとして使用。
非晶質セルの平衡構造から繰返し単位を切り出し、同一 SMILES の多数配座をアンサンブルとして学習する。

- **入力**: 繰り返し単位の SMILES
- **出力**: 繰り返し単位の 3D コンフォーマー（水素付き孤立分子）
- **モデル**: Conditional Encoder + Structural VAE + Latent DiT (Flow Matching)

## 環境

```bash
# polygen conda 環境の python を直接指定（Windows）
"C:/Users/shanu/anaconda3/envs/polygen/python.exe" <script>
```

- OS: Windows 11（bash は Git Bash）
- GPU: RTX 4060 Ti（CUDA 12.8）
- 環境: `polygen`（torch 2.8 + torch-geometric + rdkit + lmdb + scipy）
- 元素記号の取得は `Chem.GetPeriodicTable().GetElementSymbol(z)` を使用（`'X'` などのプレースホルダー不可）
- src レイアウト: `pip install -e .` で `poly3d` パッケージとして editable install 必須

## セットアップ

```bash
cd C:/Users/shanu/Documents/Python/poly3D
"C:/Users/shanu/anaconda3/envs/polygen/python.exe" -m pip install -e .
```

## コマンド

### 動作確認

```bash
"C:/Users/shanu/anaconda3/envs/polygen/python.exe" scripts/test_pipeline.py
```

### 前処理（PolyOmics tar.gz → 処理済み lmdb → train/val 分割）

`data/` に置いた `<CLASS>.tar.gz`（HF `yhayashi1986/PolyOmics` の `MD_snapshot_JSON`）を読む。
本学習は `scripts/run_main_build.ps1` が下記2段をまとめて実行する（手順は `docs/MAIN_TRAINING.md`）。

```bash
# 単一クラスで試す（--classes 省略で data/ 直下の全 tar.gz）
"C:/Users/shanu/anaconda3/envs/polygen/python.exe" scripts/build_polyomics_dataset.py \
    --data_dir data/ --classes PG \
    --out_path data/polyomics_PG.lmdb \
    --precompute_topology --per_cell_stride 3 --max_atoms 288 --map_size_gb 40

# uuid（＝ポリマー）単位で train/val 分割（同一 SMILES の train/val 漏洩を防ぐ）
"C:/Users/shanu/anaconda3/envs/polygen/python.exe" scripts/split_lmdb.py \
    --src data/polyomics_PG.lmdb \
    --train_out data/polyomics_PG_train.lmdb \
    --val_out   data/polyomics_PG_val.lmdb
```

- `--precompute_topology` で dist_mat/triplets/quartets を lmdb に埋め込む＝学習時の BFS が消える
- `--per_cell_stride N` で 1 セル内の単位を N 個ごとに間引く（配座相関を減らしつつ容量抑制）
- `split_lmdb.py` の `--train_map_gb/--val_map_gb` は既定 0＝src の実使用量から自動見積り

### 学習（2段階）

```bash
# Stage 1: Structural VAE（シングル GPU）
# RTX4060Ti 16GB / Ryzen9 7900X 推奨設定
"C:/Users/shanu/anaconda3/envs/polygen/python.exe" scripts/train.py \
    --stage vae \
    --train_lmdb data/polyomics_all_train.lmdb \
    --val_lmdb   data/polyomics_all_val.lmdb \
    --out_dir    ./runs/polygen_v1 --epochs 300 \
    --batch_size 64 --num_workers 8 --grad_accum 2

# RTX5000 Ada 32GB / Xeon 8558 推奨設定
# --batch_size 128 --num_workers 16 --grad_accum 2

# Stage 1: Structural VAE（マルチ GPU: torchrun）
torchrun --nproc_per_node=4 scripts/train.py \
    --stage vae \
    --train_lmdb data/polyomics_all_train.lmdb \
    --val_lmdb   data/polyomics_all_val.lmdb \
    --out_dir    ./runs/polygen_v1 --epochs 300

# Stage 2: Latent DiT（シングル GPU）
"C:/Users/shanu/anaconda3/envs/polygen/python.exe" scripts/train.py \
    --stage dit \
    --train_lmdb data/polyomics_all_train.lmdb \
    --val_lmdb   data/polyomics_all_val.lmdb \
    --out_dir    ./runs/polygen_v1 --epochs 600 \
    --vae_checkpoint ./runs/polygen_v1/vae_best.pt

# Stage 2: Latent DiT（マルチノード: 2ノード × 4GPU の例）
# node0 (MASTER_ADDR をここの IP に設定)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=<node0_ip> --master_port=29500 \
    scripts/train.py --stage dit \
    --train_lmdb ... --val_lmdb ... --out_dir ... \
    --vae_checkpoint ./runs/polygen_v1/vae_best.pt
```

### 推論

```bash
"C:/Users/shanu/anaconda3/envs/polygen/python.exe" scripts/sample.py \
    --vae_checkpoint ./runs/polygen_v1/vae_best.pt \
    --dit_checkpoint ./runs/polygen_v1/dit_best.pt \
    --smiles "CC(C)c1ccc(CC)cc1" \
    --out out.sdf --n_steps 100
```

### TensorBoard

```bash
tensorboard --logdir ./runs/polygen_v1
# VAE ログ: runs/polygen_v1/tb_vae/
# DiT ログ: runs/polygen_v1/tb_dit/
```

## ディレクトリ構成

```
poly3D/
├── src/
│   └── poly3d/                 # インストール可能パッケージ
│       ├── data/
│       │   └── dataset.py      # ConformerDataset（処理済み lmdb）
│       └── model/
│           ├── features.py     # 特徴量定義・RWPE/LapPE 計算
│           ├── egnn.py         # SE(3)-同変 GNN（汎用プリミティブ）
│           ├── cond_encoder.py # Graph Conditioning（トポロジー→Ci）
│           ├── vae.py          # Structural VAE（Encoder E + Decoder D）
│           ├── geo_losses.py   # 幾何損失（bond/angle/dihedral/Kabsch RMSD/distmat）
│           ├── vae_loss.py     # VAE 損失の組み立て
│           ├── pos_bias.py     # グラフ距離 → attention bias
│           ├── dit.py          # Latent DiT（block-diagonal attention）
│           └── flow_matching.py # Flow matching（損失 + Euler ODE）
├── scripts/
│   ├── build_polyomics_dataset.py # PolyOmics tar.gz → 処理済み lmdb
│   ├── split_lmdb.py           # uuid 単位で train/val 分割
│   ├── train.py                # 2段階学習（VAETrainer / DiTTrainer）
│   ├── sample.py               # 推論（SMILES → SDF）
│   ├── eval_ensemble.py        # 妥当性評価（全体・サイズ帯別）
│   ├── run_main_*.ps1          # 本学習ランチャ4段（build/vae/dit/ditcons）
│   └── test_pipeline.py        # 動作確認（4 テスト）
├── docs/MAIN_TRAINING.md       # 本学習の手順書
└── pyproject.toml              # パッケージ定義（src レイアウト）
```

## アーキテクチャ

### 3 コンポーネント + Flow Matching

```
SMILES → ConditionalEncoder → h_cond, e_cond, Ci
                                        ↓
                              [VAE Stage 1 (300ep)]
                              Ci + pos → VAE Encoder → Z0 (μ, σ)
                              Z0 + Ci  → VAE Decoder → pos_pred

                                        ↓
                              [DiT Stage 2 (600ep)]
                              Z1 ~ N(0,I)
                              Zt = (1-t)Z0 + t*Z1
                              DiT(Zt, Ci, t) → Ẑ1  (flow matching)

                              [推論]
                              Z1 → Euler ODE → Z0 → VAE Decoder → 3D座標
```

### VAE 損失

| 損失 | 内容 | 重み |
|------|------|------|
| Lpos | Kabsch RMSD（回転・並進不変） or 距離行列 MSE | 1.0 |
| Lbond | 結合長 MSE | 1.0 |
| Langle | 結合角 MSE | 0.5 |
| Ldihedral | 二面角損失 (1-cos) | 0.1 |
| Lkl | KL(q\|\|N(0,I))、β warm-up | 0.0→1.0 |

### 特徴量

| 種類 | 内容 | 表現 |
|------|------|------|
| 原子タイプ | H/C/N/O/F/S/Cl/Br/I/Si/P/Ge/Se/Sn/Te/B + other | `nn.Embedding(17, 32)` |
| 混成軌道 | SP/SP2/SP3/SP3D/SP3D2 + other | `nn.Embedding(6, 16)` |
| 結合タイプ | 単結合/二重/三重/芳香族/other | `nn.Embedding(5, 16)` |
| 原子連続値 | 芳香族・環内・形式電荷・H数・質量 | 5 次元 float |
| 結合連続値 | 共役・環内 | 2 次元 float |
| RWPE | 16 ステップのランダムウォーク | 16 次元 float |

## 注意事項

- PolyOmics の commonchem JSON は既定値と異なる原子にだけ `chg` を書く。ニトロ基は `[N+](=O)[O-]` と電荷分離して格納されているので、`chg` を落とすと中性 N の明示価数が 4 になり `SanitizeMol` が必ず失敗する（＝そのポリマーが全 RU 丸ごと欠測する）
- Windows の LMDB は `map_size` を非 sparse で実確保する。`map_size_gb` を上げた分だけディスクが即座に消える（split はソースと出力が同時に載るのでピークは約 2.2 倍）
- DataLoader の `num_workers > 0` 時は worker ごとに lmdb env を開く（`worker_init_fn`）
- 元素記号には必ず `Chem.GetPeriodicTable().GetElementSymbol(z)` を使う（`'X'` 不可）
- RWPE は scipy.sparse で計算（scipy は polygen env に含まれる）
- DiT は attn_bias（(H,N,N) float）で分子間 cross-attention を遮断（異分子間 = -1e9）
- VAE Decoder の初期座標は `init_pos MLP` で生成（全ゼロ禁止: EGNN の d²=0 問題）
- DiT チェックポイントの `'flow'` キーは `LatentDiT.state_dict()` のみ（`model.*` プレフィックスなし）

### 設計上の注記（文献照合レビュー 2026-07-31）

- **同変性の方針は polyGen と逆**: polyGen（arXiv:2504.17656）は意図的に**非同変アーキテクチャ＋回転/並進 augmentation**を採用するが、本実装は **EGNN/EGT による明示的な SE(3)/置換同変性**に置き換えている（augmentation 不使用）。μ は不変量から計算され SE(3) 不変、座標出力は絶対姿勢不定・内部無矛盾で、Kabsch/distmat（回転不変損失）と数学的に整合。この相違は意図的な設計判断であり、polyGen の踏襲ではない。
- **β の呼称**: beta を 0→0.1 に warm-up し 0.1 固定で運用するのは Bowman et al. 2016 型の **KL weight annealing / down-weighting**（posterior collapse 対策）であって、Higgins et al. 2017 の β-VAE（β>1・disentanglement 目的）とは方向が逆。ドキュメント上「β-VAE」と呼ばないこと。

## 計算効率

- **BFS（グラフ距離行列）**: 純 Python のため高コスト。`precompute_attn_inputs()` を 1 バッチ 1 回だけ呼ぶこと（FlowMatching が自動で行う）
- **DiT Attention**: `F.scaled_dot_product_attention` を使用（Flash Attention 対応）
- **attn_bias**: pos_bias + block-diagonal mask を (H,N,N) float に統合して DiTBlock に渡す
- **DDP**: LatentDiT のみを DDP ラップ（FlowMatching はラップ不要）。`self.flow.model = DDP(dit, ...)` の形式

## 分散学習（DDP）

- `torchrun` が `LOCAL_RANK` / `RANK` / `WORLD_SIZE` を自動設定 → `init_dist()` で検出
- 学習モデルのみ DDP ラップ: VAE stage は cond_encoder + vae、DiT stage は flow のみ
- 凍結モデル（DiT stage の cond_encoder / vae）は DDP 不要
- チェックポイント保存・ログ出力は rank 0 のみ（`is_main_process()` ガード）
- state_dict 保存時は `_unwrap(model).state_dict()` で DDP ラップを外す
- val loss は `_all_reduce_dict()` でランク間集計してからグローバル平均を返す
- 各 epoch 開始時に `train_sampler.set_epoch(epoch)` を呼ぶ（shuffle 再シード）
