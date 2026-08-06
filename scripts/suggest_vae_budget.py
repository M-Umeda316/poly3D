"""Stage 1 VAE の学習予算（epochs / beta warmup / val 比率）を実データから決める。

なぜ必要か
----------
run_main_vae.ps1 の既定値（300 epoch, beta_warmup 20 epoch, save_every 5）は
PG 1 クラス（train 34 万件）のパイロットを前提に決めたもので、全 22 クラスの
本データ（数百万件規模）ではそのまま使えない:

  * CosineAnnealingLR(T_max=epochs) なので epochs は「LR を下げ切る長さ」そのもの。
    到達不能な 300 を入れると LR が高いまま打ち切ることになる。
  * beta warmup は **epoch 単位**（train.py: progress = (epoch-1)/beta_warmup_epochs）。
    総 epoch が数回しかないのに warmup 20 だと beta が 0.1 に到達しないまま終わる。
  * EGT の大域アテンションは (B, N_max, N_max) の密テンソルなので VRAM は
    batch_size x (バッチ内最大原子数)^2。PG は最大 60 原子程度だったが本データは
    max_atoms=288 まで来るため、同じ bs でもピークが桁で変わる。

この 3 点は全部「実際の件数と原子数分布」を見ないと決まらないので、lmdb を
直接サンプリングして推奨値と起動コマンドをそのまま出力する。

使い方
------
    python scripts/suggest_vae_budget.py \
        --train_lmdb data/polyomics_all_train.lmdb \
        --val_lmdb   data/polyomics_all_val.lmdb

出力は ASCII のみ（PowerShell の既定コンソールで文字化けさせないため）。
"""

from __future__ import annotations

import argparse
import math
import pickle
import random

import lmdb

# VRAM モデル: alloc_GB = K * batch_size * N_pad^2
#   N_pad = そのバッチ内の最大原子数（EGT の大域 attention が (B,N,N) の密テンソル
#   なので、1 件でも大きい分子が混ざると全件そこまでパディングされる）
#
# K は現行コードの実測から回帰（2026-08-06, RTX4060Ti, width256/edge128/egt_every2,
# PG lmdb, --benchmark 10）:
#   bs64 -> alloc 0.84GB / bs128 -> 1.82GB / bs256 -> 3.47GB, N_pad ~79
#   → 定数項ほぼ 0、K = 2.2e-6 [GB / (sample * atom^2)]
# reserved は alloc の約 1.45 倍（キャッシュ・フラグメント込み）。
#
# 注意: 旧版はこの基準に v3c 実行時の reserved 11.5GB を使っていたが、あれは
# EGT の dist_bias 修正**前**の値で、現行コードでは約 1/10 に下がっている。
# 古い実測値を基準に据えると見積りが桁で過大になるので、係数を更新するときは
# 必ず現行コードの --benchmark を取り直すこと。
K_GB_PER_SAMPLE_ATOM2 = 2.2e-6
RESERVED_OVER_ALLOC = 1.45


def lmdb_len(path: str) -> int:
    env = lmdb.open(path, subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False)
    try:
        with env.begin() as txn:
            meta = txn.get(b'__len__')
            if meta is None:
                raise SystemExit(
                    f'ERROR: {path} has no __len__ key. '
                    'The build did not finish (partial lmdb).')
            return int(meta.decode('ascii'))
    finally:
        env.close()


def sample_sizes(path: str, n_total: int, n_sample: int, seed: int = 0) -> list[int]:
    """ランダムな idx を引いて原子数だけ集める（全走査は数百万件で重すぎる）。"""
    rng = random.Random(seed)
    idxs = rng.sample(range(n_total), min(n_sample, n_total))
    sizes: list[int] = []
    env = lmdb.open(path, subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False)
    try:
        with env.begin() as txn:
            for i in idxs:
                val = txn.get(f'{i:09d}'.encode('ascii'))
                if val is None:
                    continue
                d = pickle.loads(val)
                sizes.append(int(d['atom_type_idx'].shape[0]))
    finally:
        env.close()
    return sizes


def pct(xs: list[int], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return float(s[int(k)])
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--train_lmdb', default='data/polyomics_all_train.lmdb')
    p.add_argument('--val_lmdb', default='data/polyomics_all_val.lmdb')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--max_atoms', type=int, default=288)
    p.add_argument('--target_steps', type=int, default=250000,
                   help='VAE に割く optimizer step の総数。PG パイロットの '
                        'best(ep45, bs128, 34万件) = 約 12 万 step が下限の目安で、'
                        '22 クラスは多様性が上がるのでその 2 倍を既定にしている。')
    p.add_argument('--val_target', type=int, default=20000,
                   help='1 回の validation で回したいサンプル数の目安。')
    p.add_argument('--target_epochs', type=int, default=40,
                   help='epoch 基準スケジュール（cosine LR / beta warmup / val / ckpt）の'
                        '分解能。本データは 1 パスが数万 step あるので、全件走査を'
                        '1 epoch にすると epochs が数回しか取れず全部が階段になる。'
                        'ここで決めた epoch 数になるよう steps_per_epoch を逆算する。')
    p.add_argument('--sample', type=int, default=3000,
                   help='原子数分布の推定に使うサンプル件数。')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    n_train = lmdb_len(a.train_lmdb)
    n_val = lmdb_len(a.val_lmdb)
    sizes = sample_sizes(a.train_lmdb, n_train, a.sample, a.seed)
    if not sizes:
        raise SystemExit('ERROR: could not sample any record from the train lmdb.')

    # max_atoms を超えるレコードは Dataset が None を返して collate で落ちる。
    kept = [s for s in sizes if s <= a.max_atoms]
    keep_ratio = len(kept) / len(sizes)
    n_eff = int(n_train * keep_ratio)

    p50, p95, p99, smax = pct(kept, .50), pct(kept, .95), pct(kept, .99), max(kept)

    full_pass = max(1, n_eff // a.batch_size)      # 全件 1 パスの step 数

    # epoch 基準のスケジュールが階段にならないよう、まず epoch 数を決めてから
    # 1 epoch の step 数を逆算する（仮想エポック = train.py --steps_per_epoch）。
    # 全件 1 パスが目標 epoch あたりの step 数より短いなら、仮想エポックは不要。
    epochs = a.target_epochs
    steps_per_epoch = max(1, round(a.target_steps / epochs))
    if steps_per_epoch >= full_pass:
        steps_per_epoch = full_pass                # 打ち切り不要（0 = 無効を渡す）
        epochs = max(3, round(a.target_steps / full_pass))
    virtual = steps_per_epoch < full_pass

    beta_warmup = max(1, round(epochs * 0.10))
    val_ratio = min(1.0, max(0.02, round(a.val_target / max(1, n_val), 3)))
    passes = a.target_steps * a.batch_size / max(1, n_eff)

    def vram(bs: float, npad: float) -> float:
        """reserved GB の推定。npad = そのバッチ内の最大原子数。"""
        return K_GB_PER_SAMPLE_ATOM2 * bs * npad * npad * RESERVED_OVER_ALLOC

    # 学習を殺すのは「平均的なバッチ」ではなく**エポック中のピーク**なので、
    # 判定はサンプル最大原子数を含むバッチで行う。1 エポックは数千バッチあるので
    # 最大級の分子を含むバッチはほぼ確実に来る。
    # 検証(2026-08-06): PG/bs128 でこの式が 2.58GB、実測 reserved 2.64GB ＝一致。
    common_gb = vram(a.batch_size, p99)
    peak_gb = vram(a.batch_size, float(smax))
    cap_gb = vram(a.batch_size, float(a.max_atoms))

    print('=' * 68)
    print(' DATASET')
    print('=' * 68)
    print(f'  train entries        : {n_train:,}')
    print(f'  val entries          : {n_val:,}')
    print(f'  sampled              : {len(sizes):,}')
    print(f'  kept (<= {a.max_atoms} atoms) : {keep_ratio * 100:.1f}%  '
          f'-> effective train {n_eff:,}')
    print(f'  atoms p50/p95/p99/max: {p50:.0f} / {p95:.0f} / {p99:.0f} / {smax}')
    print()
    print('=' * 68)
    print(f' BUDGET  (batch_size={a.batch_size}, target {a.target_steps:,} steps)')
    print('=' * 68)
    print(f'  full pass over train : {full_pass:,} steps')
    print(f'  total data seen      : {passes:.1f} passes '
          f'({a.target_steps * a.batch_size:,} samples)')
    print(f'  -> steps_per_epoch   : {steps_per_epoch:,}'
          + ('   (VIRTUAL epoch: truncated, reshuffled each epoch, no data discarded)'
             if virtual else '   (= full pass; no truncation needed -> pass 0)'))
    print(f'  -> epochs            : {epochs}')
    print(f'  -> beta_warmup_epochs: {beta_warmup}   (10% of the run; '
          f'epoch-based in train.py)')
    print(f'  -> val_subset_ratio  : {val_ratio}   '
          f'(~{int(n_val * val_ratio):,} samples per validation)')
    print(f'  -> save_every        : 1   (1 epoch is already {steps_per_epoch:,} steps)')
    if virtual:
        print()
        print('  WHY virtual epochs: CosineAnnealingLR steps ONCE per epoch, beta ramps')
        print(f'  as (epoch-1)/warmup, and val/ckpt are per-epoch. A full pass here is')
        print(f'  {full_pass:,} steps, so spending the budget in whole passes would leave only')
        print(f'  {max(3, round(a.target_steps / full_pass))} epochs = a {max(3, round(a.target_steps / full_pass))}-point staircase for LR, beta and ckpt choice.')
    print()
    print('=' * 68)
    print(' VRAM (padded EGT attention scales as batch_size * max_atoms^2)')
    print('=' * 68)
    print(f'  common batch (N={p99:.0f})  : {common_gb:5.1f} GB reserved')
    print(f'  PEAK   batch (N={smax:3d}) : {peak_gb:5.1f} GB reserved   '
          f'<- this decides OOM (thousands of batches per epoch)')
    print(f'  if a N={a.max_atoms} unit exists : {cap_gb:5.1f} GB reserved')
    if peak_gb > 28:
        print('  VERDICT: the peak batch will OOM on 32GB. Halve --batch_size and double')
        print('           --grad_accum (effective batch and thus the loss curve are kept).')
        print('           Do NOT just paper over it with --oom_max_skips: the dropped')
        print('           batches are exactly the large units whose reconstruction is')
        print('           the open problem, so it biases training away from them.')
    elif peak_gb > 20:
        print('  VERDICT: fits, but with little headroom. Keep --oom_max_skips 20 as a')
        print('           safety net and watch the [VRAM] lines in the log.')
    else:
        print('  VERDICT: fits with headroom, peak included.')
    print()
    print('  This is a regression from measured points, not a measurement. Confirm with')
    print('  --benchmark before committing to a batch size (command printed below).')
    print()
    print('=' * 68)
    print(' LAUNCH (32GB machine, detached)')
    print('=' * 68)
    print('  $env:POLY3D_PY = "C:/Users/shanu/anaconda3/envs/polygen/python.exe"')
    print('  Start-Process powershell -ArgumentList \'-NoProfile\','
          '\'-ExecutionPolicy\',\'Bypass\',\'-File\',\'scripts/run_main_vae.ps1\',')
    print(f'    \'-Epochs\',\'{epochs}\',\'-BatchSize\',\'{a.batch_size}\','
          f'\'-BetaWarmupEpochs\',\'{beta_warmup}\',')
    print(f'    \'-StepsPerEpoch\',\'{steps_per_epoch if virtual else 0}\','
          f'\'-ValSubsetRatio\',\'{val_ratio}\',\'-SaveEvery\',\'1\','
          f'\'-OomMaxSkips\',\'20\' -WindowStyle Hidden')
    print()
    print('  Measure the real VRAM first (the estimate above is a crude extrapolation):')
    print(f'    & $env:POLY3D_PY scripts/train.py --stage vae --benchmark 30 \\')
    print(f'      --train_lmdb {a.train_lmdb} --val_lmdb {a.val_lmdb} \\')
    print(f'      --batch_size {a.batch_size} --hidden_dim 256 --edge_dim 128 \\')
    print(f'      --vae_hidden_dim 256 --egt_every 2 --enc_egt_every 2 --max_atoms '
          f'{a.max_atoms}')
    print('=' * 68)


if __name__ == '__main__':
    main()
