"""
VAE ハイパーパラメータ探索スクリプト。

定義済みの設定を順次実行し、best val_loss を比較してサマリーを出力する。
各実行は <base_out_dir>/<name>/ に独立して保存される。

【アーキテクチャ探索】
    python scripts/search_params.py --group arch \
        --train_lmdb D:/Dataset/OMol_base/OPoly26/processed/train.lmdb \
        --val_lmdb   D:/Dataset/OMol_base/OPoly26/processed/val.lmdb \
        --base_out_dir ./runs/search_v1

【Loss weight 探索】
    python scripts/search_params.py --group weights \
        --train_lmdb ... --val_lmdb ... --base_out_dir ./runs/search_v1

【特定設定のみ】
    python scripts/search_params.py --names A1_baseline,A4_large ...

【結果確認のみ（再実行なし）】
    python scripts/search_params.py --group arch --dry_run ...
    python scripts/search_params.py --summarize --base_out_dir ./runs/search_v1
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

TRAIN_SCRIPT = str(Path(__file__).parent / 'train.py')

# ── デフォルト値 ──────────────────────────────────────────────────────────────
_D_ARCH = dict(
    hidden_dim=128, edge_dim=64, cond_layers=4,
    vae_hidden_dim=128, enc_layers=4, dec_layers=4, latent_dim=16,
)
_L_ARCH = dict(
    hidden_dim=256, edge_dim=128, cond_layers=4,
    vae_hidden_dim=256, enc_layers=6, dec_layers=6, latent_dim=16,
)
_D_WEIGHT = dict(
    w_pos=1.0, w_bond=1.0, w_angle=0.5, w_dihedral=0.1,
    beta_end=1.0, beta_warmup_epochs=50,
)

# ── アーキテクチャ探索設定 ────────────────────────────────────────────────────
# 問い: モデルの容量をどこに割り当てるか（幅 vs 深さ、encoder vs decoder）
ARCH_CONFIGS: list[dict] = [
    # ベースライン（現在のデフォルト）
    dict(name='A1_baseline',
         **_D_ARCH, **_D_WEIGHT),

    # 幅だけ拡大（enc/dec 層数はそのまま）
    dict(name='A2_wide',
         hidden_dim=256, edge_dim=128, cond_layers=4,
         vae_hidden_dim=256, enc_layers=4, dec_layers=4, latent_dim=16,
         **_D_WEIGHT),

    # decoder だけ深く（encoder は軽量のまま）
    dict(name='A3_deep_dec',
         hidden_dim=128, edge_dim=64, cond_layers=4,
         vae_hidden_dim=128, enc_layers=4, dec_layers=6, latent_dim=16,
         **_D_WEIGHT),

    # 幅・深さともに拡大（推奨ベースライン候補）
    dict(name='A4_large',
         **_L_ARCH, **_D_WEIGHT),

    # 潜在次元を拡大
    dict(name='A5_large_z32',
         hidden_dim=256, edge_dim=128, cond_layers=4,
         vae_hidden_dim=256, enc_layers=6, dec_layers=6, latent_dim=32,
         **_D_WEIGHT),

    # さらに大きい（VRAM 許容範囲で試す）
    dict(name='A6_xlarge',
         hidden_dim=384, edge_dim=128, cond_layers=4,
         vae_hidden_dim=256, enc_layers=6, dec_layers=8, latent_dim=32,
         **_D_WEIGHT),
]

# ── Loss weight 探索設定（A4_large アーキテクチャ固定）────────────────────────
# 問い: bond/angle/dihedral の重みバランスをどう設定するか
WEIGHT_CONFIGS: list[dict] = [
    # W1: デフォルト（アーキテクチャ探索と同じ設定 → ベースライン）
    dict(name='W1_default',
         **_L_ARCH,
         w_pos=1.0, w_bond=1.0, w_angle=0.5, w_dihedral=0.10,
         beta_end=1.0, beta_warmup_epochs=50),

    # W2: 幾何損失を強化（bond + angle + dihedral を一律 2x）
    dict(name='W2_geo_heavy',
         **_L_ARCH,
         w_pos=1.0, w_bond=2.0, w_angle=1.0, w_dihedral=0.30,
         beta_end=1.0, beta_warmup_epochs=50),

    # W3: KL を弱める（posterior collapse 対策）
    dict(name='W3_weak_kl',
         **_L_ARCH,
         w_pos=1.0, w_bond=1.0, w_angle=0.5, w_dihedral=0.10,
         beta_end=0.5, beta_warmup_epochs=50),

    # W4: 幾何強化 + 弱 KL（組み合わせ）
    dict(name='W4_geo_weak_kl',
         **_L_ARCH,
         w_pos=1.0, w_bond=2.0, w_angle=1.0, w_dihedral=0.30,
         beta_end=0.5, beta_warmup_epochs=50),

    # W5: bond に集中（angle/dihedral は抑える）
    dict(name='W5_bond_focus',
         **_L_ARCH,
         w_pos=1.0, w_bond=3.0, w_angle=0.3, w_dihedral=0.05,
         beta_end=1.0, beta_warmup_epochs=50),

    # W6: distmat pos loss（Kabsch の代替）
    dict(name='W6_distmat',
         **_L_ARCH,
         w_pos=1.0, w_bond=1.0, w_angle=0.5, w_dihedral=0.10,
         beta_end=1.0, beta_warmup_epochs=50,
         pos_loss_type='distmat'),

    # W7: KL warm-up を長くする
    dict(name='W7_long_warmup',
         **_L_ARCH,
         w_pos=1.0, w_bond=1.0, w_angle=0.5, w_dihedral=0.10,
         beta_end=1.0, beta_warmup_epochs=80),

    # W8: geo_heavy + long_warmup（W2 + W7 の組み合わせ）
    dict(name='W8_geo_long_warmup',
         **_L_ARCH,
         w_pos=1.0, w_bond=2.0, w_angle=1.0, w_dihedral=0.30,
         beta_end=0.5, beta_warmup_epochs=80),
]

ALL_CONFIGS = ARCH_CONFIGS + WEIGHT_CONFIGS

# _build_cmd が base_args から追加するキー → config dict に同名があっても無視
_BASE_KEYS = frozenset({
    'name', 'train_lmdb', 'val_lmdb', 'base_out_dir',
    'epochs', 'subset_ratio', 'batch_size', 'num_workers',
    'grad_accum', 'seed', 'val_every', 'group', 'names',
    'dry_run', 'skip_existing', 'summarize',
})


def _read_best_val(log_csv: Path) -> dict | None:
    """vae_log.csv から best val_total 行を読む。val=0 は未実行エポックとして除外。"""
    if not log_csv.exists():
        return None
    best = None
    try:
        with open(log_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    val = float(row['val_total'])
                except (KeyError, ValueError):
                    continue
                if val <= 0:
                    continue
                if best is None or val < best['val_total']:
                    best = {
                        'epoch':       int(row['epoch']),
                        'val_total':   val,
                        'val_pos':     float(row.get('val_pos', 0)),
                        'val_kl':      float(row.get('val_kl', 0)),
                        'train_total': float(row.get('train_total', 0)),
                    }
    except Exception as e:
        print(f'  [WARN] CSV 読み込みエラー ({log_csv}): {e}')
        return None
    return best


def _build_cmd(cfg: dict, base_args: argparse.Namespace) -> list[str]:
    """設定辞書 + base_args から train.py コマンドラインを組み立てる。"""
    name    = cfg['name']
    out_dir = str(Path(base_args.base_out_dir) / name)

    cmd = [
        sys.executable, TRAIN_SCRIPT,
        '--stage',        'vae',
        '--train_lmdb',   base_args.train_lmdb,
        '--val_lmdb',     base_args.val_lmdb,
        '--out_dir',      out_dir,
        '--epochs',       str(base_args.epochs),
        '--subset_ratio', str(base_args.subset_ratio),
        '--batch_size',   str(base_args.batch_size),
        '--num_workers',  str(base_args.num_workers),
        '--grad_accum',   str(base_args.grad_accum),
        '--seed',         str(base_args.seed),
        '--val_every',    str(base_args.val_every),
        '--save_every',   str(base_args.epochs + 1),  # 定期保存なし、best のみ
    ]

    # 設定固有パラメータ（base_args が既に担うキーは除外）
    for k, v in cfg.items():
        if k in _BASE_KEYS:
            continue
        cmd += [f'--{k}', str(v)]

    return cmd


def _summarize(configs: list[dict], base_out_dir: Path):
    """実行済み結果のみ読み取ってサマリー表示（再実行なし）。"""
    results = []
    for cfg in configs:
        name = cfg['name']
        log_csv = base_out_dir / name / 'vae_log.csv'
        best = _read_best_val(log_csv)
        if best:
            results.append({'name': name, 'status': 'OK', **best})
        else:
            results.append({'name': name, 'status': 'NOT_RUN'})
    _print_summary(results)


def _print_summary(results: list[dict]):
    ok  = sorted([r for r in results if r['status'] == 'OK'],
                 key=lambda r: r['val_total'])
    ng  = [r for r in results if r['status'] != 'OK']

    print(f'\n{"="*72}')
    print('  Search Summary')
    print(f'{"="*72}')
    print(f'  {"name":<24} {"val_total":>10} {"val_pos":>9} {"val_kl":>8} '
          f'{"train":>9} {"epoch":>6}')
    print(f'  {"-"*70}')
    for r in ok:
        mark = ' ◀ best' if r is ok[0] else ''
        print(f'  {r["name"]:<24} {r["val_total"]:>10.4f} {r["val_pos"]:>9.4f} '
              f'{r["val_kl"]:>8.4f} {r["train_total"]:>9.4f} {r["epoch"]:>6}{mark}')
    for r in ng:
        print(f'  {r["name"]:<24} {"—":>10} {"—":>9} {"—":>8} {"—":>9} {"—":>6}  '
              f'{r["status"]}')
    print()


def run_search(configs: list[dict], base_args: argparse.Namespace):
    base_out = Path(base_args.base_out_dir)
    results: list[dict] = []
    total = len(configs)

    for i, cfg in enumerate(configs, 1):
        name    = cfg['name']
        log_csv = base_out / name / 'vae_log.csv'

        print(f'\n{"="*72}')
        print(f'  [{i}/{total}] {name}')
        print(f'{"="*72}')

        # 既存スキップ
        if base_args.skip_existing and log_csv.exists():
            best = _read_best_val(log_csv)
            if best:
                print(f'  [SKIP] ログ既存 → val_total={best["val_total"]:.4f}')
                results.append({'name': name, 'status': 'OK', **best})
                continue

        cmd = _build_cmd(cfg, base_args)
        print('  $ ' + ' '.join(cmd[2:]))  # python + script は省略

        if base_args.dry_run:
            print('  [DRY RUN] 実行をスキップ')
            results.append({'name': name, 'status': 'DRY_RUN'})
            continue

        ret = subprocess.run(cmd)

        if ret.returncode != 0:
            print(f'  [FAILED] returncode={ret.returncode}')
            results.append({'name': name, 'status': 'FAILED'})
            continue

        best = _read_best_val(log_csv)
        if best is None:
            print(f'  [NO LOG] {log_csv} が見つからない')
            results.append({'name': name, 'status': 'NO_LOG'})
        else:
            print(f'  best val_total={best["val_total"]:.4f} '
                  f'(pos={best["val_pos"]:.4f} kl={best["val_kl"]:.4f}) '
                  f'@ epoch {best["epoch"]}')
            results.append({'name': name, 'status': 'OK', **best})

    _print_summary(results)

    # CSV 保存
    if not base_args.dry_run:
        summary_csv = base_out / 'search_results.csv'
        base_out.mkdir(parents=True, exist_ok=True)
        fields = ['name', 'status', 'val_total', 'val_pos', 'val_kl', 'train_total', 'epoch']
        with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k, '') for k in fields})
        print(f'  結果保存: {summary_csv}')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='VAE ハイパーパラメータ探索',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--train_lmdb',    type=str, default='')
    p.add_argument('--val_lmdb',      type=str, default='')
    p.add_argument('--base_out_dir',  type=str, default='./runs/search')
    p.add_argument('--epochs',        type=int, default=100,
                   help='1設定あたりの学習エポック数')
    p.add_argument('--subset_ratio',  type=float, default=0.1,
                   help='訓練データのサブセット比率')
    p.add_argument('--batch_size',    type=int, default=64)
    p.add_argument('--num_workers',   type=int, default=8)
    p.add_argument('--grad_accum',    type=int, default=2)
    p.add_argument('--seed',          type=int, default=42)
    p.add_argument('--val_every',     type=int, default=5,
                   help='val 実行間隔（エポック）')
    p.add_argument('--group', choices=['arch', 'weights', 'all'], default='arch',
                   help='実行グループ: arch（アーキテクチャ）/ weights（Loss重み）/ all')
    p.add_argument('--names', type=str, default=None,
                   help='カンマ区切りで特定設定のみ実行 (例: --names A1_baseline,A4_large)')
    p.add_argument('--skip_existing', action='store_true',
                   help='ログが既に存在する設定をスキップ（中断再開用）')
    p.add_argument('--dry_run', action='store_true',
                   help='コマンドを表示するだけで実際には実行しない')
    p.add_argument('--summarize', action='store_true',
                   help='実行済みの結果のみ集計して表示（再実行なし）')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # 設定リストの決定
    if args.names:
        names_set = set(args.names.split(','))
        configs = [c for c in ALL_CONFIGS if c['name'] in names_set]
        missing = names_set - {c['name'] for c in configs}
        if missing:
            print(f'[WARNING] 不明な設定名: {missing}', file=sys.stderr)
    elif args.group == 'arch':
        configs = ARCH_CONFIGS
    elif args.group == 'weights':
        configs = WEIGHT_CONFIGS
    else:
        configs = ALL_CONFIGS

    if not configs:
        print('[ERROR] 実行する設定がありません', file=sys.stderr)
        sys.exit(1)

    if args.summarize:
        _summarize(configs, Path(args.base_out_dir))
        sys.exit(0)

    if not args.train_lmdb or not args.val_lmdb:
        print('[ERROR] --train_lmdb と --val_lmdb は必須です', file=sys.stderr)
        sys.exit(1)

    print(f'探索設定数: {len(configs)}')
    print(f'各設定: {args.epochs} epoch, subset_ratio={args.subset_ratio}')
    for c in configs:
        print(f'  - {c["name"]}')

    run_search(configs, args)
