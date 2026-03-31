# poly3D

polyGen（[arXiv:2504.17656](https://arxiv.org/abs/2504.17656)）インスパイアのポリマー繰り返し単位 3D 構造生成モデル。

SMILES を入力として、水素付き孤立分子の 3D コンフォーマーを生成する。
学習データには OPoly26（[arXiv:2512.23117](https://arxiv.org/abs/2512.23117)）を使用。

## アーキテクチャ概要

2 段階学習。

```
Stage 1 — Structural VAE
  SMILES → Graph Conditioning (MPNN) → Ci
  Ci + 3D座標 → VAE Encoder → Z (N, d_z)
  Z  + Ci      → VAE Decoder → 座標復元

Stage 2 — Latent DiT (Flow Matching)
  Z1 ~ N(0,I)
  Zt = (1-t)·Z0 + t·Z1
  DiT(Zt, Ci, t) → Ẑ1    loss: 1/(1-t)² ||Z1 - Ẑ1||²

推論
  SMILES → Ci
  Z1 ~ N(0,I) —[Euler ODE]→ Z0
  Z0 + Ci → VAE Decoder → 3D座標 → .sdf
```

**Graph Conditioning**: atom/bond の `nn.Embedding` + RWPE + MPNN × 4 層 + global pooling
**VAE**: EGNN ベース（SE(3)-equivariant）
**DiT**: Transformer + block-diagonal attention mask（分子間 cross-attention 遮断）+ グラフ距離ベース positional bias + self-conditioning

## 依存関係

| パッケージ | 用途 |
|---|---|
| PyTorch 2.8 + CUDA 12.8 | 学習・推論 |
| torch-geometric | グラフバッチ処理 |
| torch-scatter | scatter 演算 |
| RDKit | 分子処理・結合推定 |
| lmdb | データセット I/O |
| scipy | RWPE・LapPE 計算 |

## セットアップ

```bash
# conda 環境作成（例）
conda create -n polygen python=3.11
conda activate polygen
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install torch-geometric torch-scatter
pip install rdkit lmdb scipy tqdm

# パッケージを editable install（src レイアウト）
cd poly3D
pip install -e .
```

## データ準備

OPoly26 の `.aselmdb` ファイルから処理済み lmdb を生成する。

```bash
# val（約 20 万件、動作確認用）
python -m poly3d.preprocess.build_dataset \
    --src_dir  D:/Dataset/OMol_base/OPoly26/val \
    --out_path D:/Dataset/OMol_base/OPoly26/processed/val.lmdb \
    --n_workers 8

# train（約 600 万件、数時間）
python -m poly3d.preprocess.build_dataset \
    --src_dir  D:/Dataset/OMol_base/OPoly26/train \
    --out_path D:/Dataset/OMol_base/OPoly26/processed/train.lmdb \
    --n_workers 8 --map_size_gb 80
```

## 学習

### シングル GPU

```bash
# Stage 1: Structural VAE (300 epochs)
python scripts/train.py \
    --stage vae \
    --train_lmdb D:/Dataset/OMol_base/OPoly26/processed/train.lmdb \
    --val_lmdb   D:/Dataset/OMol_base/OPoly26/processed/val.lmdb \
    --out_dir    ./runs/polygen_v1 \
    --epochs 300

# Stage 2: Latent DiT (600 epochs)
python scripts/train.py \
    --stage dit \
    --train_lmdb D:/Dataset/OMol_base/OPoly26/processed/train.lmdb \
    --val_lmdb   D:/Dataset/OMol_base/OPoly26/processed/val.lmdb \
    --out_dir    ./runs/polygen_v1 \
    --vae_checkpoint ./runs/polygen_v1/vae_best.pt \
    --epochs 600
```

### マルチ GPU（単一ノード）

```bash
torchrun --nproc_per_node=4 scripts/train.py --stage vae \
    --train_lmdb D:/Dataset/OMol_base/OPoly26/processed/train.lmdb \
    --val_lmdb   D:/Dataset/OMol_base/OPoly26/processed/val.lmdb \
    --out_dir    ./runs/polygen_v1 --epochs 300
```

### マルチノード（例: 2ノード × 4GPU = 8GPU）

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

`--batch_size` はランクごとのバッチサイズ。実効 batch = `batch_size × world_size`。

主なオプション（`python scripts/train.py --help` で全一覧）:

| オプション | デフォルト | 説明 |
|---|---|---|
| `--hidden_dim` | 128 | ConditionalEncoder 隠れ次元 |
| `--latent_dim` | 16 | 潜在変数次元 |
| `--dit_hidden_dim` | 256 | DiT 隠れ次元 |
| `--dit_n_layers` | 6 | DiT 層数 |
| `--batch_size` | 32 | バッチサイズ |
| `--lr` | 1e-4 | 学習率 |
| `--beta_warmup_epochs` | 50 | KL warm-up エポック数 |

## 推論

```bash
python scripts/sample.py \
    --vae_checkpoint ./runs/polygen_v1/vae_best.pt \
    --dit_checkpoint ./runs/polygen_v1/dit_best.pt \
    --smiles "CC(C)c1ccc(CC)cc1" \
    --out conformers.sdf \
    --n_conf 5 \
    --n_steps 100
```

複数 SMILES をまとめて処理する場合:

```bash
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

6 テスト（lmdb_reader / features / cond_encoder / vae / dit+flow / preprocess）が全て `[ OK ]` になれば環境構築完了。

## ディレクトリ構成

```
poly3D/
├── src/
│   └── poly3d/
│       ├── data/
│       │   └── dataset.py          # ConformerDataset（lmdb → PyG Batch）
│       ├── model/
│       │   ├── features.py         # 特徴量定義・RWPE/LapPE 計算・mol→dict
│       │   ├── egnn.py             # SE(3)-equivariant GNN
│       │   ├── cond_encoder.py     # Graph Conditioning（MPNN + global pooling）
│       │   ├── vae.py              # Structural VAE
│       │   ├── geo_losses.py       # 幾何損失（bond / angle / dihedral）
│       │   ├── vae_loss.py         # VAE 損失の組み立て
│       │   ├── pos_bias.py         # グラフ距離 → attention bias
│       │   ├── dit.py              # Latent DiT
│       │   └── flow_matching.py    # Flow matching（損失 + Euler ODE）
│       └── preprocess/
│           ├── lmdb_reader.py      # .aselmdb 読み込み（zlib+JSON）
│           └── build_dataset.py    # 前処理パイプライン
├── scripts/
│   ├── train.py                    # 学習エントリポイント
│   ├── sample.py                   # 推論エントリポイント
│   └── test_pipeline.py            # 動作確認テスト
└── pyproject.toml
```

## 参考文献

- **polyGen**: Ruan et al., "polyGen: A Conditional Generative Model for Polymer 3D Structures", arXiv:2504.17656 (2025)
- **OPoly26**: arXiv:2512.23117 (2024)
- **EGNN**: Satorras et al., "E(n) Equivariant Graph Neural Networks", ICML 2021
- **RWPE**: Dwivedi et al., "Graph Neural Networks with Learnable Structural and Positional Representations", ICLR 2022
