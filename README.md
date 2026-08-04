# poly3D

polyGen（[arXiv:2504.17656](https://arxiv.org/abs/2504.17656)）インスパイアのポリマー繰り返し単位 3D 構造生成モデル。

SMILES を入力として、水素付き孤立分子の 3D コンフォーマーを生成する。
学習データには OPoly26（[arXiv:2512.23117](https://arxiv.org/abs/2512.23117)）を使用。

## アーキテクチャ概要

2 段階学習。

```
Stage 1 — Structural VAE
  SMILES → ConditionalEncoder (MPNN × 4) → Ci, e_cond
  Ci + 3D座標 → VAE Encoder (EGNN × 4) → μ, σ
  Z = μ + σ·ε → VAE Decoder (EGNN × 4) → 座標復元

Stage 2 — Latent DiT (Flow Matching)
  Z1 ~ N(0,I)
  Zt = (1-t)·Z0 + t·Z1
  DiT(Zt, Ci, t) → Ẑ1    loss: 1/(1-t)² ||Z1 - Ẑ1||²

推論
  SMILES → Ci
  Z1 ~ N(0,I) —[Euler ODE, 100 steps]→ Z0
  Z0 + Ci → VAE Decoder → 3D座標 → .sdf
```

**ConditionalEncoder**: atom/bond の `nn.Embedding` + RWPE + MPNN × 4 + mean global pooling → Ci
**Structural VAE**: EGNN ベース（SE(3)-equivariant）、各原子に per-atom 潜在変数
**Latent DiT**: Transformer + block-diagonal attention mask（分子間 cross-attention 遮断）+ グラフ距離 positional bias + self-conditioning

## 依存関係

| パッケージ | 用途 |
|---|---|
| PyTorch 2.8 + CUDA 12.8 | 学習・推論 |
| torch-geometric | グラフバッチ処理 |
| torch-scatter | scatter 演算 |
| RDKit | 分子処理・結合推定 |
| lmdb | データセット I/O |
| scipy | RWPE・LapPE・グラフ距離計算 |

## セットアップ

```bash
conda create -n polygen python=3.11
conda activate polygen
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install torch-geometric torch-scatter
pip install rdkit lmdb scipy tqdm tensorboard

# src レイアウト: editable install 必須
cd poly3D
pip install -e .
```

## データ準備

OPoly26 の `.aselmdb` から処理済み lmdb を生成する。

```bash
# val（動作確認用）
python -m poly3d.preprocess.build_dataset \
    --src_dir  D:/Dataset/OMol_base/OPoly26/val \
    --out_path D:/Dataset/OMol_base/OPoly26/processed/val.lmdb \
    --n_workers 8

# train（大規模、数時間）
python -m poly3d.preprocess.build_dataset \
    --src_dir  D:/Dataset/OMol_base/OPoly26/train \
    --out_path D:/Dataset/OMol_base/OPoly26/processed/train.lmdb \
    --n_workers 8 --map_size_gb 80
```

## 学習

### Stage 1: Structural VAE

```bash
# シングル GPU（RTX4060Ti 16GB 推奨設定）
python scripts/train.py \
    --stage vae \
    --train_lmdb D:/Dataset/OMol_base/OPoly26/processed/train.lmdb \
    --val_lmdb   D:/Dataset/OMol_base/OPoly26/processed/val.lmdb \
    --out_dir    ./runs/polygen_v1 \
    --epochs 300 \
    --batch_size 64 --num_workers 8 --grad_accum 2

# マルチ GPU（torchrun）
torchrun --nproc_per_node=4 scripts/train.py --stage vae \
    --train_lmdb D:/Dataset/OMol_base/OPoly26/processed/train.lmdb \
    --val_lmdb   D:/Dataset/OMol_base/OPoly26/processed/val.lmdb \
    --out_dir    ./runs/polygen_v1 --epochs 300
```

### Stage 2: Latent DiT

#### オプション A: 標準モード（VAE Encoder を毎バッチ実行）

```bash
python scripts/train.py \
    --stage dit \
    --train_lmdb D:/Dataset/OMol_base/OPoly26/processed/train.lmdb \
    --val_lmdb   D:/Dataset/OMol_base/OPoly26/processed/val.lmdb \
    --out_dir    ./runs/polygen_v1 \
    --vae_checkpoint ./runs/polygen_v1/vae_best.pt \
    --epochs 600 \
    --batch_size 64 --num_workers 8 --grad_accum 2
```

#### オプション B: 事前エンコードモード（推奨・高速）

ConditionalEncoder + VAE Encoder の推論コストを事前にキャッシュして DiT 学習を高速化する。

```bash
# 1. 潜在変数を事前エンコード
python scripts/precompute_latents.py \
    --vae_checkpoint ./runs/polygen_v1/vae_best.pt \
    --src_lmdb  D:/Dataset/OMol_base/OPoly26/processed/train.lmdb \
    --out_lmdb  D:/Dataset/OMol_base/OPoly26/latents/train.lmdb \
    --batch_size 256 --num_workers 8

python scripts/precompute_latents.py \
    --vae_checkpoint ./runs/polygen_v1/vae_best.pt \
    --src_lmdb  D:/Dataset/OMol_base/OPoly26/processed/val.lmdb \
    --out_lmdb  D:/Dataset/OMol_base/OPoly26/latents/val.lmdb \
    --batch_size 256 --num_workers 8

# 2. キャッシュ済み LMDB で DiT 学習
python scripts/train.py \
    --stage dit \
    --train_lmdb D:/Dataset/OMol_base/OPoly26/processed/train.lmdb \
    --val_lmdb   D:/Dataset/OMol_base/OPoly26/processed/val.lmdb \
    --latent_lmdb     D:/Dataset/OMol_base/OPoly26/latents/train.lmdb \
    --latent_val_lmdb D:/Dataset/OMol_base/OPoly26/latents/val.lmdb \
    --out_dir    ./runs/polygen_v1 \
    --vae_checkpoint ./runs/polygen_v1/vae_best.pt \
    --epochs 600 \
    --batch_size 64 --num_workers 8 --grad_accum 2
```

### マルチノード（例: 2ノード × 4GPU）

```bash
# node0（マスター: IP=192.168.1.10）
torchrun \
    --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=192.168.1.10 --master_port=29500 \
    scripts/train.py --stage vae \
    --train_lmdb ... --val_lmdb ... --out_dir ./runs/polygen_v1 --epochs 300

# node1
torchrun \
    --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr=192.168.1.10 --master_port=29500 \
    scripts/train.py --stage vae \
    --train_lmdb ... --val_lmdb ... --out_dir ./runs/polygen_v1 --epochs 300
```

### 主な学習オプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--batch_size` | 32 | ランクごとのバッチサイズ |
| `--grad_accum` | 1 | 勾配累積ステップ数（実効 batch = batch_size × grad_accum） |
| `--num_workers` | 8 | DataLoader ワーカー数 |
| `--lr` | 1e-4 | 学習率 |
| `--grad_clip` | 1.0 | 勾配クリッピング |
| `--val_every` | 1 | validation を実行するエポック間隔 |
| `--save_every` | 10 | チェックポイント保存間隔 |
| `--no_amp` | False | bf16 混合精度を無効化（fp32 で実行） |
| `--compile` | False | `torch.compile` でモデルをコンパイル |
| `--tb_log_every` | 100 | TensorBoard にステップ単位で書き込む間隔（0 = エポック末のみ） |
| `--benchmark N` | 0 | N バッチのタイミング計測を実行して終了 |
| `--hidden_dim` | 128 | ConditionalEncoder 隠れ次元 |
| `--latent_dim` | 16 | 潜在変数次元 |
| `--beta_warmup_epochs` | 50 | KL warm-up エポック数 |
| `--pos_loss_type` | kabsch | 座標損失の種類（`kabsch` / `distmat`）→ 後述 |
| `--dit_hidden_dim` | 256 | DiT 隠れ次元 |
| `--dit_n_layers` | 6 | DiT 層数 |

### 推奨設定

| 環境 | `--batch_size` | `--num_workers` | `--grad_accum` |
|---|---|---|---|
| RTX4060Ti 16GB / Ryzen9 7900X | 64 | 8 | 2 |
| RTX5000 Ada 32GB / Xeon 8558 | 128 | 16 | 2 |

### 座標損失の選択（`--pos_loss_type`）

VAE Stage 1 の座標損失は回転・並進不変な指標を使用する。

| 値 | 内容 | 特徴 |
|---|---|---|
| `kabsch`（デフォルト）| Kabsch RMSD: SVD で最適回転を求めてアライメント後に MSE | O(n) / molecule、精度高 |
| `distmat` | 全原子ペア距離行列の MSE | SVD 不要でシンプル、O(n²) / molecule |

```bash
# 距離行列損失に切り替える場合
python scripts/train.py --stage vae ... --pos_loss_type distmat
```

旧来の座標 MSE（回転非不変）は廃止済み。

### TensorBoard

```bash
tensorboard --logdir ./runs/polygen_v1
# VAE ログ: runs/polygen_v1/tb_vae/  （エポック + ステップ両軸）
# DiT ログ: runs/polygen_v1/tb_dit/
```

### ベンチマーク

ボトルネック特定に使うセクション別タイミング計測。

```bash
python scripts/train.py \
    --stage vae \
    --train_lmdb D:/Dataset/OMol_base/OPoly26/processed/train.lmdb \
    --val_lmdb   D:/Dataset/OMol_base/OPoly26/processed/val.lmdb \
    --out_dir ./runs/bench \
    --batch_size 64 --benchmark 20
```

## 推論

```bash
python scripts/sample.py \
    --vae_checkpoint ./runs/polygen_v1/vae_best.pt \
    --dit_checkpoint ./runs/polygen_v1/dit_best.pt \
    --smiles "CC(C)c1ccc(CC)cc1" \
    --out conformers.sdf \
    --n_conf 5 \
    --n_steps 100

# 複数 SMILES をまとめて処理
python scripts/sample.py \
    --vae_checkpoint ./runs/polygen_v1/vae_best.pt \
    --dit_checkpoint ./runs/polygen_v1/dit_best.pt \
    --smiles_file smiles_list.txt \
    --out conformers.sdf
```

## 動作確認

```bash
python scripts/test_pipeline.py
```

4 テスト（features / cond_encoder / vae / dit+flow）が全て `[ OK ]` になれば環境構築完了。

## ディレクトリ構成

```
poly3D/
├── src/
│   └── poly3d/
│       ├── data/
│       │   ├── dataset.py          # ConformerDataset（処理済み lmdb → PyG Batch）
│       │   └── latent_dataset.py   # LatentDataset（事前エンコード済み lmdb）
│       ├── model/
│       │   ├── features.py         # 特徴量定義・RWPE/LapPE 計算・mol→dict
│       │   ├── egnn.py             # SE(3)-equivariant GNN
│       │   ├── cond_encoder.py     # Graph Conditioning（MPNN + mean global pooling）
│       │   ├── vae.py              # Structural VAE（Encoder + Decoder）
│       │   ├── geo_losses.py       # 幾何損失（bond / angle / dihedral / Kabsch RMSD / distmat）
│       │   ├── vae_loss.py         # VAE 損失の組み立て
│       │   ├── pos_bias.py         # グラフ距離 → attention bias
│       │   ├── dit.py              # Latent DiT
│       │   ├── flow_matching.py    # Flow matching（損失 + Euler ODE）
│       │   └── builder.py          # モデル構築ファクトリ（train/sample 共用）
│       └── preprocess/
│           ├── lmdb_reader.py      # .aselmdb 読み込み（zlib+JSON）
│           └── build_dataset.py    # 前処理パイプライン
├── scripts/
│   ├── train.py                    # 学習エントリポイント（VAETrainer / DiTTrainer）
│   ├── sample.py                   # 推論エントリポイント
│   ├── precompute_latents.py       # 事前エンコード（DiT Stage 2 高速化用）
│   └── test_pipeline.py            # 動作確認テスト（4 項目）
└── pyproject.toml
```

## 参考文献

- **polyGen**: Ruan et al., "polyGen: A Conditional Generative Model for Polymer 3D Structures", arXiv:2504.17656 (2025)
- **OPoly26**: arXiv:2512.23117 (2024)
- **EGNN**: Satorras et al., "E(n) Equivariant Graph Neural Networks", ICML 2021
- **RWPE**: Dwivedi et al., "Graph Neural Networks with Learnable Structural and Positional Representations", ICLR 2022
- **Flow Matching**: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
