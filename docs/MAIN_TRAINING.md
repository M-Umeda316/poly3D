# 本学習 手順書（全22クラス PolyOmics, from-scratch）

PG パイロットで確定したレシピで、全22クラスの本学習を回すための手順。
32GB 機（RTX5000 Ada 32GB / Xeon）想定。16GB 機でも回るが遅い。

## 確定レシピ（なぜこの構成か）
- **アーキ固定**: width256 / EGT enc+dec(egt_every 2) / enc・dec 4層 / latent 16。
  overfit 診断で「多様な大単位も暗記可＝容量は十分」と確定済み。容量は増やさない。
- **VAE 損失 = multiscale_distmat**（局所＋long-range 距離）。Kabsch RMSD の torsion-flip
  感度が大単位の汎化を阻害していた問題を解消（PG で大単位 recon 0.47→0.85 を実証）。
- **beta = 0→0.1 warmup で 0.1 固定**（1.0 にしない＝posterior collapse 回避）。
- **ditcons**（デコーダを実 DiT 潜在にロバスト化）は DiT 学習後の仕上げ段で適用。
  PG で全体生成 0.70→0.96、大単位 0.08→0.44 を実証済み。

---

## 前提（実行前に必ず）

1. **conda env**: `polygen`。python は `C:/Users/shanu/anaconda3/envs/polygen/python.exe`。
   ```powershell
   cd C:\Users\shanu\Documents\Python\poly3D
   "C:/Users/shanu/anaconda3/envs/polygen/python.exe" -m pip install -e .   # src レイアウト
   ```
2. **学習データ tar.gz を `data/` に配置**（22クラス、HuggingFace、~90GB）:
   ```bash
   # Git Bash
   cd /c/Users/shanu/Documents/Python/poly3D/data
   BASE="https://huggingface.co/datasets/yhayashi1986/PolyOmics/resolve/main/MD_snapshot_JSON"
   for C in PAMD PANH PARC PCBN PDIE PEST PG PHAL PHYC PI PIMD PIMN PKTN POXI PPHS PPNL PSFO PSTR PSUL PURA PURT PVNL; do
     curl -L -o "$C.tar.gz" "$BASE/$C.tar.gz"
   done
   ```
   `ls data/*.tar.gz` が 22 件になれば OK。（`huggingface-cli download` でも可）
3. **ランチャ実行時の注意（Windows）**:
   - `.ps1` はダブルクリックすると**メモ帳で開くだけ**（実行されない）。必ず下記コマンドで起動。
   - `-ExecutionPolicy Bypass` を付けないと「スクリプト実行が無効」で止まる。
   - ランチャは python パスを `$env:POLY3D_PY` から取る（未設定だと `python` にフォールバック）。

---

## 実行順（4段）

各ランチャは Start-Process デタッチ・`status.txt` 状態機械・冪等 resume（クラッシュ後の再投入で
続きから）。**段ごとに結果を確認してから次へ**進むこと。

起動の共通形（デタッチ）:
```powershell
cd C:\Users\shanu\Documents\Python\poly3D
$env:POLY3D_PY = "C:/Users/shanu/anaconda3/envs/polygen/python.exe"
Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','scripts/run_main_<STAGE>.ps1' -WindowStyle Hidden
```
（`<STAGE>` = build / vae / dit / ditcons）

| # | ランチャ | 内容 | 出力 | 完了後に確認 |
|---|---|---|---|---|
| 0 | `run_main_build.ps1` | 全クラスビルド(`--precompute_topology --per_cell_stride 1`)→uuid単位split。CPU | `data/polyomics_all_{train,val}.lmdb` | `runs/polyomics_main_build/build_out.log` の skip サマリ（multi_molecules/triclinic_box 等の警告件数） |
| 1 | `run_main_vae.ps1` | Stage1 VAE from-scratch(multiscale, beta 0→0.1)。**起動前に下記「学習予算の決め方」必読** | `runs/polyomics_main_vae/vae_best.pt`, `eval_recon.json` | recon 妥当性（サイズ帯別）、val_pos が上振れしないか |
| 2 | `run_main_dit.ps1` | 潜在 precompute→DiT 学習(bs256/200ep)→eval dit | `runs/polyomics_main_dit/dit_best.pt`, `eval_dit_final.json` | dit 妥当性（サイズ帯別） |
| 3 | `run_main_ditcons.ps1` | dit潜在 precompute→デコーダ仕上げ(freeze_encoder, w_ditcons/w_robust, 15ep) | `runs/polyomics_main_ditcons/eval_{recon,dit}_final.json` | 最終の生成妥当性（大単位が伸びたか） |

### ★ 1回目の Stage1 は posterior collapse で全損（2026-08-17 記録）

既定の `-Lr 3e-4` のまま 80 epoch 回して **完全に失敗**した。記録しておく:

| | val_pos | val_kl | recon 妥当性 |
|---|---|---|---|
| 本学習 1回目（22クラス, width256, lr 3e-4, ep80）| 3.64 | **0.0009** | **0.00** |
| PG v3b（width256, from-scratch, lr **1.5e-4**）| 0.187 | 0.137 | 健全 |
| PG v3（width256, from-scratch, lr **3e-4**）| 0.705 | 0.0059 | 途中で破棄 |

**lr 3e-4 × width256 の from-scratch は bad basin ＋ posterior collapse を起こす**、というのは
PG パイロットの v3 で既に分かっていたのに、ランチャの既定値に反映されていなかった。
`-Lr` の既定は **1.5e-4**（v3b で唯一 collapse せずに収束した値）に修正済み。

**collapse のトリップワイヤ**: beta が 0.1 に到達したあと **val_kl が 0.1 を下回ったら赤信号、
0.05 を切ったら即停止**。健全な run（v3b）は 0.137 で床を打ち、それ以下には行かない。
80 epoch 完走を待ってから気付くと数日を捨てることになる。1回目は ep40 の時点で
val_kl 0.013 まで落ちていて、そこで止められた。

### 再走の手順

1. **データを先に検査する**（GPU 数日を投じる前に、CPU 数分で潰す）:
   ```powershell
   & $env:POLY3D_PY scripts/check_lmdb_geometry.py `
       --lmdb data/polyomics_all_train.lmdb --sample 5000 `
       --out runs/data_check_train.json
   ```
   学習ターゲットである **GT 配座**が `eval_ensemble` と同じ妥当性ゲートを通るかを見る。
   **PG（クリーンと実測済み）の基準値は GT 妥当率 0.995 / clash-free 1.000 / 結合長 p99 2.02Å**。
   全22クラスがこれに近ければデータは白、大きく下回る帯・uuid があれば
   そこはビルド（PBC アンラップ・キャップ）の失敗を疑う。PG 以外の21クラスの
   GT 幾何は一度も検証していないので、ここは必ず通すこと。
2. **出力先を変えて起動する**。ランチャは `vae_epoch*.pt` があれば自動 resume するので、
   同じ `-OutName` で叩き直すと**壊れた ckpt から再開してしまう**:
   ```powershell
   Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
     '-File','scripts/run_main_vae.ps1','-OutName','polyomics_main_vae2',
     '-BatchSize','64','-GradAccum','2', ... -WindowStyle Hidden
   ```
   `-OutName` は Stage2/3 にも `-VaeRun` で渡す（`run_main_dit.ps1 -VaeRun polyomics_main_vae2`）。

### 学習予算の決め方（Stage1 VAE 起動前に必ず）

`run_main_vae.ps1` の既定 `-Epochs 300 -BetaWarmupEpochs 20 -SaveEvery 5` は
**PG 1クラス（train 34万件）のパイロット規模**の数字。全22クラスの本ビルドは
train が数百万件になり、1 epoch がそのまま数万 step になるので、**epoch 基準の
3つのパラメータを一緒にスケールし直さないと壊れる**:

- `-Epochs`: `CosineAnnealingLR(T_max=epochs)` なので**これが LR を下げ切る長さそのもの**。
  到達できない 300 を入れると LR が高いまま打ち切ることになる。
- `-BetaWarmupEpochs`: beta は `(epoch-1)/warmup` の **epoch 基準**。総 epoch より
  warmup が長いと **beta が 0.1 に到達しないまま学習が終わる**。
- `-SaveEvery`: resume 粒度。本データは 1 epoch が数時間なので 1 にする。

**さらに本ビルドでは「全件1パス＝1エポック」が成立しない**。train が約530万件なので
bs64 で 1 パス＝約 83,000 step。step 予算 25万を全件パスで消化すると **epochs が 3 しか
取れず、LR も beta も val も ckpt も 3 点の階段**になる（実測: epochs 3 / warmup 1 で
beta が 0 → 0.1 の単なる段差になり warmup が消える、LR は 3 step 目で eta_min 到達）。

→ `--steps_per_epoch N` で **1 エポックを N バッチに切る（仮想エポック）**。shuffle は
毎エポック引き直されるので**毎回違う N バッチ**を見る＝ `--subset_ratio`（seed 固定の
サブセットを一度選んで残りを永久に捨てる）とは別物で、データは捨てない。これで
epochs を 40 程度に戻すと epoch 基準スケジュールの分解能が復活する。

実件数と原子数分布から推奨値と起動コマンドを出すスクリプトを使う:

```powershell
& $env:POLY3D_PY scripts/suggest_vae_budget.py `
    --train_lmdb data/polyomics_all_train.lmdb `
    --val_lmdb   data/polyomics_all_val.lmdb --batch_size 128
```

出力の最後にそのまま貼れる `Start-Process ...` が出る。`--target_steps` が予算
（既定 25万 step ＝ PG パイロットの best 相当 12万 step の約2倍）。

**batch_size について**: EGT の大域アテンションは `(B, N_max, N_max)` の密テンソルで、
VRAM は `batch_size × バッチ内最大原子数²`（`N_max` はバッチ内の**最大**。1 件でも
大きい分子が混ざると全件そこまでパディングされる）。PG は最大 79 原子だったが本データは
`max_atoms=288` まで来るので、同じ bs でもピークが桁で変わる。

判定は「平均的なバッチ」ではなく**エポック中のピーク**で行うこと。1 エポックは数千
バッチあり、最大級の分子を含むバッチはほぼ確実に来る。

**VRAM 実測（2026-08-06, RTX4060Ti, width256/edge128/egt_every2, PG, `--benchmark 10`）**:

| batch_size | alloc | reserved |
|---|---|---|
| 64 | 0.84 GB | 1.05 GB |
| 128 | 1.82 GB | 2.64 GB |
| 256 | 3.47 GB | 5.06 GB |

定数項はほぼ 0 で `alloc ≈ 2.2e-6 × B × N_max²` [GB]、`reserved ≈ 1.45 × alloc`。
**旧ドキュメントの「bs64 で reserved 11.5GB」は EGT dist_bias 修正前の v3c 実測値**で、
現行コードでは約 1/10。古い値を基準に外挿すると桁で過大評価になる。

`--benchmark N` がピーク VRAM を出すので、bs はそれで確認してから決めること。
稀な重量バッチで run ごと落とさないよう `-OomMaxSkips 20` を付けるのが安全（既定 0 ＝
fail-fast）。ただし**捨てられるのは大単位を含むバッチ＝まさに再構築が課題の当の対象**
なので、常用すると学習が難しい側から静かに逸れる。恒常的に溢れるなら bs を半分にして
`--grad_accum` を倍にする（実効バッチが保たれるので損失曲線の比較性は維持される）。

### 進捗の見方
- `status.txt`（ascii, bash-grep 可）: `..._START` → `..._DONE` → `ALL_DONE`。
- ログは PowerShell リダイレクトで **UTF-16**。bash grep でなく **`Get-Content <log> -Wait -Tail 20`** で追う。
- `vae_log.csv` / `dit_log.csv`（train.py が UTF-8 で直接書く）で epoch 毎の loss。

### 最初の1段だけ画面で見たい場合（build）
ランチャは出力をログに落とすので、初回は build を**直接**叩くと進捗が画面に出る:
```powershell
& "C:/Users/shanu/anaconda3/envs/polygen/python.exe" scripts/build_polyomics_dataset.py `
  --data_dir data/ --out_path data/polyomics_all.lmdb --precompute_topology `
  --per_cell_stride 1 --max_atoms 288 --map_size_gb 250
```
終われば `data/polyomics_all.lmdb` ができ、`run_main_build.ps1` は build をスキップして split だけ実行する。

---

## トラブルシュート
- **`.ps1` がメモ帳で開く** → ダブルクリックしただけ。上記 `Start-Process ... -File ...` で起動する。
- **「スクリプトの実行が無効」** → `-ExecutionPolicy Bypass` を付ける。
- **「ソースが空です」** → `data/` に tar.gz が無い（前提2）。
- **別 python が使われる** → `$env:POLY3D_PY` を設定してから起動。
- **from-scratch で gnorm が発散** → 主因は angle/dihedral 損失(atan2)。`run_main_vae.ps1` の
  `-Lr` を下げる、または w_angle/w_dihedral を下げる/warmup を伸ばす。全クラス大データなら
  平均化されて安定する見込み（v3c 実績 gnorm~2.8）。
- **VRAM** → dist_bias 修正で width256 は大幅減。32GB なら bs 更に上げ可（`-BatchSize`）。

## 注意
- **旧 ckpt(v3c/v3f/v3h) は非互換**（EGT の dist_bias/coord_g 幅を変更したため）。本学習は
  from-scratch 前提なので問題ないが、`--init_weights` で旧 ckpt を読むことはできない。
- データ(lmdb/ckpt)は git 管理外。コードは `git pull` で最新化してから実行すること。
