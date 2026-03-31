# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

polyGen（arXiv:2504.17656）インスパイアのポリマー3D構造生成モデル。
OPoly26 データセット（ASE lmdb 形式）を学習データとして使用。

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
- **fairchem 不使用**（ASE lmdb を直接 zlib+JSON 読み込み）
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

### 前処理（ASE lmdb → 処理済み lmdb）

```bash
# val（小規模、まずここから試す）
"C:/Users/shanu/anaconda3/envs/polygen/python.exe" -m poly3d.preprocess.build_dataset \
    --src_dir D:/Dataset/OMol_base/OPoly26/val \
    --out_path D:/Dataset/OMol_base/OPoly26/processed/val.lmdb \
    --n_workers 8

# train（大規模、数時間）
"C:/Users/shanu/anaconda3/envs/polygen/python.exe" -m poly3d.preprocess.build_dataset \
    --src_dir D:/Dataset/OMol_base/OPoly26/train \
    --out_path D:/Dataset/OMol_base/OPoly26/processed/train.lmdb \
    --n_workers 8 --map_size_gb 80
```

### 学習（2段階）

```bash
# Stage 1: Structural VAE
"C:/Users/shanu/anaconda3/envs/polygen/python.exe" scripts/train.py \
    --stage vae \
    --train_lmdb D:/Dataset/OMol_base/OPoly26/processed/train.lmdb \
    --val_lmdb   D:/Dataset/OMol_base/OPoly26/processed/val.lmdb \
    --out_dir    ./runs/polygen_v1 --epochs 300

# Stage 2: Latent DiT
"C:/Users/shanu/anaconda3/envs/polygen/python.exe" scripts/train.py \
    --stage dit \
    --train_lmdb D:/Dataset/OMol_base/OPoly26/processed/train.lmdb \
    --val_lmdb   D:/Dataset/OMol_base/OPoly26/processed/val.lmdb \
    --out_dir    ./runs/polygen_v1 --epochs 600 \
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

## ディレクトリ構成

```
poly3D/
├── src/
│   └── poly3d/                 # インストール可能パッケージ
│       ├── data/
│       │   └── dataset.py      # ConformerDataset（処理済み lmdb）
│       ├── model/
│       │   ├── features.py     # 特徴量定義・RWPE/LapPE 計算
│       │   ├── egnn.py         # SE(3)-同変 GNN（汎用プリミティブ）
│       │   ├── cond_encoder.py # Graph Conditioning（トポロジー→Ci）
│       │   ├── vae.py          # Structural VAE（Encoder E + Decoder D）
│       │   ├── geo_losses.py   # 幾何損失（bond/angle/dihedral）
│       │   ├── vae_loss.py     # VAE 損失の組み立て
│       │   ├── pos_bias.py     # グラフ距離 → attention bias
│       │   ├── dit.py          # Latent DiT（block-diagonal attention）
│       │   └── flow_matching.py # Flow matching（損失 + Euler ODE）
│       └── preprocess/
│           ├── lmdb_reader.py  # .aselmdb 直接読み込み（zlib+JSON）
│           └── build_dataset.py # 前処理パイプライン（spawn multiprocessing）
├── scripts/
│   ├── train.py                # 2段階学習（VAETrainer / DiTTrainer）
│   ├── sample.py               # 推論（SMILES → SDF）
│   └── test_pipeline.py        # 動作確認（6 テスト）
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
| Lpos | 座標 MSE | 1.0 |
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

- Windows multiprocessing は `spawn` 必須（`build_dataset.py`）
- DataLoader の `num_workers > 0` 時は worker ごとに lmdb env を開く（`worker_init_fn`）
- 元素記号には必ず `Chem.GetPeriodicTable().GetElementSymbol(z)` を使う（`'X'` 不可）
- RWPE は scipy.sparse で計算（scipy は polygen env に含まれる）
- DiT は block-diagonal attention mask で分子間 cross-attention を遮断
- VAE Decoder の初期座標は `init_pos MLP` で生成（全ゼロ禁止: EGNN の d²=0 問題）
