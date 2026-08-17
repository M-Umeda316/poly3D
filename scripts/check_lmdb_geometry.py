"""ビルド済み lmdb の参照(GT)幾何が妥当性ゲートを通るかを検査する。

なぜ必要か
----------
`eval_ensemble.py` の妥当性 pass 率は「生成配座がゲートを通る割合」だが、
**そもそも学習ターゲットである GT 配座がゲートを通らない**なら、モデルは
学習不可能なものを学習させられていることになる。PG 1クラスでは GT が
クリーンであることを実測で確認済みだが、残り21クラスは未検証のまま
本ビルドに入っている。再学習に数日の GPU を投じる前に、ここを潰す。

ゲートは eval_ensemble.validity_of_conformer と同一定義:
  - clash        : グラフ距離 >= 3 のペアで dist < (rvdw_i + rvdw_j) * 0.6
  - 結合長サニティ: 結合ペア距離が [0.7, 2.6] Å の外
sanitize は幾何非依存なのでここでは見ない(データ幾何の検査が目的)。

使い方:
    python scripts/check_lmdb_geometry.py --lmdb data/polyomics_all_train.lmdb \
        --sample 5000 --out runs/data_check_train.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict

import lmdb
import numpy as np
from rdkit import Chem

_PT = Chem.GetPeriodicTable()

# eval_ensemble と同じサイズ帯
SIZE_BANDS = [(0, 11), (11, 16), (16, 30), (30, 60), (60, 10 ** 9)]


def band_of(n_heavy: int) -> str:
    for lo, hi in SIZE_BANDS:
        if lo <= n_heavy < hi:
            return f'[{lo},{hi})' if hi < 10 ** 9 else f'[{lo},+)'
    return 'unknown'


def graph_distance(edge_index: np.ndarray, n: int, max_dist: int = 4) -> np.ndarray:
    """結合グラフ上のホップ距離を BFS で計算（max_dist でクランプ）。"""
    adj = [[] for _ in range(n)]
    for a, b in zip(edge_index[0], edge_index[1]):
        adj[int(a)].append(int(b))
    out = np.full((n, n), max_dist, dtype=np.int16)
    for src in range(n):
        out[src, src] = 0
        frontier = [src]
        for d in range(1, max_dist):
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if out[src, v] > d:
                        out[src, v] = d
                        nxt.append(v)
            if not nxt:
                break
            frontier = nxt
    return out


def check_record(d: dict, clash_factor: float, bond_lo: float, bond_hi: float) -> dict:
    pos = np.asarray(d['pos'], dtype=np.float64)
    edge_index = np.asarray(d['edge_index'])
    atomic_nums = np.asarray(d['atomic_nums']).astype(int)
    n = pos.shape[0]

    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))

    # 非結合マスク: 保存済み dist_mat があれば使う（ビルド時 max_dist=4 クランプ）
    if 'dist_mat' in d and d['dist_mat'] is not None:
        hop = np.asarray(d['dist_mat'])
    else:
        hop = graph_distance(edge_index, n)
    nonbond = hop >= 3

    rvdw = np.array([_PT.GetRvdw(int(z)) for z in atomic_nums], dtype=np.float64)
    thr = (rvdw[:, None] + rvdw[None, :]) * clash_factor
    n_clash = int(np.count_nonzero(nonbond & (dist < thr)) // 2)

    bi, bj = edge_index[0], edge_index[1]
    if bi.size > 0:
        bd = dist[bi, bj]
        bond_max = float(bd.max())
        bond_min = float(bd.min())
        n_bad_bond = int(np.count_nonzero((bd < bond_lo) | (bd > bond_hi)) // 2)
    else:
        bond_max = bond_min = float('nan')
        n_bad_bond = 0

    n_heavy = int(np.count_nonzero(atomic_nums > 1))
    return {
        'n_atoms': n,
        'n_heavy': n_heavy,
        'n_clash': n_clash,
        'n_bad_bond': n_bad_bond,
        'bond_max': bond_max,
        'bond_min': bond_min,
        'valid': (n_clash == 0 and n_bad_bond == 0),
        'sid': d.get('sid', ''),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--lmdb', required=True)
    p.add_argument('--sample', type=int, default=5000, help='検査するレコード数（一様サンプル）')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--clash_factor', type=float, default=0.6)
    p.add_argument('--bond_lo', type=float, default=0.7)
    p.add_argument('--bond_hi', type=float, default=2.6)
    p.add_argument('--worst', type=int, default=20, help='fail 率が高い uuid の表示数')
    p.add_argument('--out', default=None, help='結果 JSON の出力先')
    a = p.parse_args()

    env = lmdb.open(a.lmdb, subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False, max_readers=256)
    with env.begin() as txn:
        meta = txn.get(b'__len__')
        total = int(meta.decode('ascii')) if meta else txn.stat()['entries']

    n_sample = min(a.sample, total)
    rng = np.random.default_rng(a.seed)
    idxs = rng.choice(total, size=n_sample, replace=False)
    idxs.sort()

    recs = []
    with env.begin() as txn:
        for idx in idxs:
            val = txn.get(f'{int(idx):09d}'.encode('ascii'))
            if val is None:
                continue
            recs.append(check_record(pickle.loads(val), a.clash_factor, a.bond_lo, a.bond_hi))
    env.close()

    if not recs:
        print('レコードを1件も読めませんでした。')
        return

    def summarize(rs: list) -> dict:
        bmax = np.array([r['bond_max'] for r in rs if np.isfinite(r['bond_max'])])
        return {
            'n': len(rs),
            'gt_valid_rate': float(np.mean([r['valid'] for r in rs])),
            'clash_free_rate': float(np.mean([r['n_clash'] == 0 for r in rs])),
            'bond_ok_rate': float(np.mean([r['n_bad_bond'] == 0 for r in rs])),
            'bond_max_p50': float(np.median(bmax)) if bmax.size else float('nan'),
            'bond_max_p99': float(np.percentile(bmax, 99)) if bmax.size else float('nan'),
            'bond_max_max': float(bmax.max()) if bmax.size else float('nan'),
            'mean_clash_per_conf': float(np.mean([r['n_clash'] for r in rs])),
        }

    overall = summarize(recs)
    by_band = {}
    bands = defaultdict(list)
    for r in recs:
        bands[band_of(r['n_heavy'])].append(r)
    for k in sorted(bands, key=lambda s: float(s.strip('[)+,').split(',')[0])):
        by_band[k] = summarize(bands[k])

    by_uuid = defaultdict(list)
    for r in recs:
        by_uuid[str(r['sid']).split(':')[0]].append(r)
    uuid_rates = [(u, float(np.mean([x['valid'] for x in rs])), len(rs))
                  for u, rs in by_uuid.items()]
    uuid_rates.sort(key=lambda t: t[1])

    print(f'lmdb            : {a.lmdb}')
    print(f'全レコード       : {total:,}   検査: {len(recs):,}')
    print()
    print(f'GT 妥当率(全体)  : {overall["gt_valid_rate"]:.4f}'
          f'   (clash-free {overall["clash_free_rate"]:.4f} / bond-ok {overall["bond_ok_rate"]:.4f})')
    print(f'結合長 max       : p50 {overall["bond_max_p50"]:.3f} / p99 {overall["bond_max_p99"]:.3f}'
          f' / max {overall["bond_max_max"]:.3f} A')
    print(f'clash/配座 平均  : {overall["mean_clash_per_conf"]:.3f}')
    print()
    print('サイズ帯別 (重原子数):')
    print(f'  {"band":<10} {"n":>7} {"GT妥当":>8} {"clash-free":>11} {"bond-ok":>9} {"bond_max_max":>13}')
    for k, s in by_band.items():
        print(f'  {k:<10} {s["n"]:>7} {s["gt_valid_rate"]:>8.4f} {s["clash_free_rate"]:>11.4f}'
              f' {s["bond_ok_rate"]:>9.4f} {s["bond_max_max"]:>13.3f}')
    print()
    print(f'GT 妥当率が低い uuid (下位 {a.worst}):')
    for u, rate, n in uuid_rates[:a.worst]:
        print(f'  {u}  rate={rate:.3f}  n={n}')

    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump({
                'lmdb': a.lmdb, 'total': total, 'checked': len(recs),
                'overall': overall, 'by_band': by_band,
                'worst_uuids': [{'uuid': u, 'gt_valid_rate': r, 'n': n}
                                for u, r, n in uuid_rates[:a.worst]],
            }, f, indent=2)
        print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
