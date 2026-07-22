"""eval_by_size.py の出力 JSON を 2 本以上並べて差分テーブルを吐く。

存在理由: 本番学習は別マシン（32GB機）で回しており、結果ファイルは持ち帰れず
**画面を見て手打ち報告**している。eval_by_size のフルテーブルは帯 × 指標 × 末端で
数十行あり手打ちには重すぎる。このスクリプトは判定に必要な行だけを 3-6 行に
圧縮して出すので、それだけ打てば（or 撮れば）比較が成立する。

使い方（比較したい JSON を左から古い順に並べる。ラベルはファイル名から自動）:
    python scripts/cmp_runs.py runs/gen_v1/cmp_C_large.json runs/gen_v1/cmp_G_large.json
    python scripts/cmp_runs.py runs/gen_v1/cmp_{C,D,E}_large.json --full
    python scripts/cmp_runs.py runs/gen_v1/cmp_C_valgen.json runs/gen_v1/cmp_G_valgen.json

  --full  : p90 / bond / 末端 e2e_ratio も出す（既定は rmsd 中央値と成功率のみ）
  --label : ラベルを明示（例 --label C G）

判定の向き: rmsd と bond は**小さいほど良い**、成功率<1A と e2e_ratio は
**大きいほど良い**（e2e_ratio は 1.0 が正解＝<1 で末端が内に巻いている）。
"""
import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass


def parse_args():
    p = argparse.ArgumentParser(description='eval_by_size の JSON を並べて比較')
    p.add_argument('json', nargs='+', help='cmp_*.json を 2 本以上（左が基準）')
    p.add_argument('--label', nargs='*', default=None, help='各 JSON のラベル')
    p.add_argument('--full', action='store_true',
                   help='p90 / bond / e2e_ratio も出す')
    return p.parse_args()


def auto_label(path: str) -> str:
    """runs/gen_v1/cmp_C_large.json -> 'C'"""
    b = os.path.basename(path)
    for ext in ('.json',):
        if b.endswith(ext):
            b = b[: -len(ext)]
    if b.startswith('cmp_'):
        b = b[4:]
    for suf in ('_large', '_valgen'):
        if b.endswith(suf):
            b = b[: -len(suf)]
    return b or '?'


def fmt_delta(base, cur, lower_is_better: bool) -> str:
    """基準比の変化率。改善=▽ / 悪化=▲ を付ける（記号は ASCII 安全のため後述）。"""
    if base is None or cur is None or base == 0:
        return ''
    pct = 100.0 * (cur - base) / abs(base)
    better = (cur < base) if lower_is_better else (cur > base)
    mark = 'good' if better else 'BAD '
    return f'{pct:+6.1f}% {mark}'


def get_band(d: dict, band: str) -> dict:
    return d.get('by_size', {}).get(band, {})


def get_ep_band(d: dict, band: str) -> dict:
    return d.get('endpoint', {}).get('by_size', {}).get(band, {})


def main():
    a = parse_args()
    if len(a.json) < 2:
        sys.exit('JSON は 2 本以上（左が基準）')

    data, labels = [], []
    for i, path in enumerate(a.json):
        if not os.path.exists(path):
            sys.exit(f'見つかりません: {path}')
        with open(path, encoding='utf-8') as f:
            data.append(json.load(f))
        if a.label and i < len(a.label):
            labels.append(a.label[i])
        else:
            labels.append(auto_label(path))

    base, base_label = data[0], labels[0]

    # 帯は全 JSON の和集合（n が 0 の帯は eval 側で落ちているため）
    bands = []
    for d in data:
        for b in d.get('by_size', {}):
            if b not in bands:
                bands.append(b)

    def band_key(b):
        try:
            return int(b.split(',')[0].lstrip('['))
        except ValueError:
            return 0
    bands.sort(key=band_key)

    print(f'=== {" -> ".join(labels)} | {os.path.basename(a.json[0])} 他 ===')
    for d, lab in zip(data, labels):
        print(f'  {lab}: n_mol={d.get("n_molecules","?")} '
              f'corr(natoms,rmsd)={d.get("corr_natoms_rmsd", float("nan")):+.3f} '
              f'ckpt={os.path.basename(str(d.get("checkpoint","?")))}')
    print()

    # ---- 判定用の最小テーブル: rmsd 中央値 と 成功率 -------------------------
    for band in bands:
        n = get_band(base, band).get('n', get_band(data[-1], band).get('n', '?'))
        vals, succ = [], []
        for d in data:
            b = get_band(d, band)
            vals.append(b.get('rmsd_median'))
            succ.append(b.get('success_rmsd<1A'))
        vs = ' -> '.join('  n/a' if v is None else f'{v:6.3f}' for v in vals)
        ss = ' -> '.join('n/a' if s is None else f'{100*s:.0f}%' for s in succ)
        delta = fmt_delta(vals[0], vals[-1], lower_is_better=True)
        print(f'{band:<11} n={str(n):<4} rmsd {vs}  {delta:<14} succ<1A {ss}')

    if not a.full:
        print('\n(--full で p90 / bond / e2e_ratio も表示)')
        return

    print()
    for band in bands:
        p90 = [get_band(d, band).get('rmsd_p90') for d in data]
        bond = [get_band(d, band).get('bond_median') for d in data]
        ratio = [get_ep_band(d, band).get('e2e_ratio_median') for d in data]
        f = lambda xs: ' -> '.join('  n/a' if x is None else f'{x:6.3f}' for x in xs)
        print(f'{band:<11} p90  {f(p90)}  {fmt_delta(p90[0], p90[-1], True)}')
        print(f'{band:<11} bond {f(bond)}  {fmt_delta(bond[0], bond[-1], True)}')
        # e2e_ratio は 1.0 が正解なので「1 からの隔たり」で良否を見る
        d0 = None if ratio[0] is None else abs(ratio[0] - 1.0)
        dN = None if ratio[-1] is None else abs(ratio[-1] - 1.0)
        mark = ''
        if d0 is not None and dN is not None:
            mark = 'good' if dN < d0 else 'BAD '
        print(f'{band:<11} e2e_ratio {f(ratio)}  (1.0 が正解) {mark}')


if __name__ == '__main__':
    main()
