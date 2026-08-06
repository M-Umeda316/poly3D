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

# PG パイロット（v3c）の実測基準点: bs64 / バッチ内最大 60 原子程度で
# reserved 11.5GB。padded attention は B*N^2 で効くので、この積を 1.0 とした
# 相対指数で本データの bs を評価する。
PG_PAD_INDEX = 64 * 60 * 60
PG_RESERVED_GB = 11.5


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

    steps_per_epoch = max(1, n_eff // a.batch_size)
    epochs = max(3, round(a.target_steps / steps_per_epoch))
    beta_warmup = max(1, round(epochs * 0.10))
    val_ratio = min(1.0, max(0.02, round(a.val_target / max(1, n_val), 3)))

    # padded attention の相対コスト（PG パイロット = 1.0）。バッチ内最大原子数は
    # 実質 p99 付近が支配する（1 件でも大きいと全件そこまでパディングされる）。
    pad_index = a.batch_size * p99 * p99 / PG_PAD_INDEX
    est_gb = PG_RESERVED_GB * pad_index

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
    print(f'  steps / epoch        : {steps_per_epoch:,}')
    print(f'  -> epochs            : {epochs}')
    print(f'  -> beta_warmup_epochs: {beta_warmup}   (10% of the run; '
          f'epoch-based in train.py)')
    print(f'  -> val_subset_ratio  : {val_ratio}   '
          f'(~{int(n_val * val_ratio):,} samples per validation)')
    print(f'  -> save_every        : 1   (1 epoch is already {steps_per_epoch:,} steps)')
    print()
    print('=' * 68)
    print(' VRAM (padded EGT attention scales as batch_size * max_atoms^2)')
    print('=' * 68)
    print(f'  pad index vs PG pilot: {pad_index:.2f}x  '
          f'(PG: bs64 x 60 atoms -> {PG_RESERVED_GB}GB reserved)')
    print(f'  rough reserved est.  : {est_gb:.1f} GB')
    if est_gb > 26:
        print('  VERDICT: too tight for 32GB. Lower --batch_size '
              '(re-run this script) or keep bs and rely on --oom_max_skips.')
    elif est_gb > 18:
        print('  VERDICT: fits, but spiky batches may OOM. '
              'Use --oom_max_skips 20 so a rare heavy batch is dropped, '
              'not the whole run.')
    else:
        print('  VERDICT: comfortable on 32GB.')
    print()
    print('=' * 68)
    print(' LAUNCH (32GB machine, detached)')
    print('=' * 68)
    print('  $env:POLY3D_PY = "C:/Users/shanu/anaconda3/envs/polygen/python.exe"')
    print('  Start-Process powershell -ArgumentList \'-NoProfile\','
          '\'-ExecutionPolicy\',\'Bypass\',\'-File\',\'scripts/run_main_vae.ps1\',')
    print(f'    \'-Epochs\',\'{epochs}\',\'-BatchSize\',\'{a.batch_size}\','
          f'\'-BetaWarmupEpochs\',\'{beta_warmup}\',')
    print(f'    \'-ValSubsetRatio\',\'{val_ratio}\',\'-SaveEvery\',\'1\','
          f'\'-OomMaxSkips\',\'20\' -WindowStyle Hidden')
    print('=' * 68)


if __name__ == '__main__':
    main()
