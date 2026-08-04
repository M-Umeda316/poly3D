"""PolyOmics(RadonPy) 平衡化構造 → poly3d 処理済み lmdb ビルダ。

PolyOmics の各 `<uuid>_eq.json`（1 非晶質セル = 同一繰返し単位の多数コピー）から
繰返し単位(RU)を 1 つずつ切り出し、鎖文脈で緩和された 3D 配座を持つ H 込みの単位
分子として `mol_to_data` に通し、既存 ConformerDataset と同形式の lmdb に格納する。

キモは「同一 SMILES に対する多数の非晶質配座」がそのまま学習アンサンブルになること
（実測: 1 セルで同一単位 ~180 配座、重原子 GetBestRMS 中央値 ~3 A の広がり）。

--------------------------------------------------------------------------------
必要なデータ（「tar.gz と csv があればいい？」への答え）:
  * tar.gz だけで学習用 lmdb は作れる。構造 JSON(commonchem) に原子/結合/3D座標/
    残基タグが全部入っており、繰返し単位の SMILES は切り出した単位から RDKit で
    導出する（--csv 不要）。
  * CSV は任意。UUID 列で <uuid>_eq.json と結合でき、正式なポリマー SMILES
    (smiles_list) や物性(temp/tacticity/QM記述子…)を各レコードに付与したいときだけ
    --csv で渡す（物性条件付け・フィルタ用）。無くても学習は可能。
--------------------------------------------------------------------------------

ソースの与え方（オンライン/オフライン両対応。ローカル指定なら完全オフライン動作）:
  --data_dir <DIR>   … DIR 直下の *.tar.gz を全処理（--classes で名前フィルタ）。← オフライン
  --sources <...>    … URL / ローカル tar.gz / 単一 _eq.json / グロブ（複数可）。

例:
  # 完全オフライン: ダウンロード済み MD_snapshot_JSON フォルダを丸ごと
  python scripts/build_polyomics_dataset.py \
      --data_dir D:/PolyOmics/MD_snapshot_JSON \
      --out_path data/polyomics_all.lmdb --per_cell_stride 3 --max_atoms 288 --map_size_gb 40

  # オフライン + 一部クラス + 物性CSV結合
  python scripts/build_polyomics_dataset.py \
      --data_dir D:/PolyOmics/MD_snapshot_JSON --classes PG PI PEST \
      --csv D:/PolyOmics/general_polymers_with_sp_abbe_dynamic-dielectric.csv \
      --csv_cols smiles_list temp tacticity \
      --out_path data/polyomics_sub.lmdb

  # オンライン: HF から直接ストリーム（全DL不要）
  python scripts/build_polyomics_dataset.py \
      --sources https://huggingface.co/datasets/yhayashi1986/PolyOmics/resolve/main/MD_snapshot_JSON/PG.tar.gz \
      --out_path data/polyomics_PG.lmdb
"""
from __future__ import annotations

import argparse
import csv as csvmod
import glob
import json
import pickle
import tarfile
import urllib.request
from collections import defaultdict, Counter
from pathlib import Path
from typing import Iterator, Optional, Tuple

import lmdb
import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from poly3d.model.features import mol_to_data

_BT = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}

# トリクリニック box 警告をクラス単位で1回だけ出すための既出集合（プロセス内で共有）
_warned_triclinic_classes: set = set()

# SanitizeMol 失敗を (クラス, 例外種別) ごとに1回だけ出すための既出集合
_warned_sanitize: set = set()


# ── セル JSON → 単位 RDKit mol の切り出し ───────────────────────────────────────

def cut_units_from_cell(cell: dict, uuid: str = '', class_name: str = '',
                         skip: Optional[Counter] = None) -> Iterator[Tuple[str, "Chem.Mol"]]:
    """1 セルの commonchem JSON から RU 単位を (residue_tag, H込みmol) で yield。

    ``uuid``/``class_name``/``skip`` は観測性向上のためのオプション引数（後方互換：
    省略時も従来どおり動作する）。PG クラスのような単純な cell（molecules/conformers
    が各1件・box が直交）では警告は一切出ない。
    """
    if skip is None:
        skip = Counter()

    molecules = cell['molecules']
    if len(molecules) > 1:
        print(f'  [warn] {class_name}:{uuid} molecules={len(molecules)} 件 → 先頭のみ処理')
        skip['multi_molecules'] += 1
    m = molecules[0]
    z_def = cell.get('defaults', {}).get('atom', {}).get('z', 6)
    chg_def = cell.get('defaults', {}).get('atom', {}).get('chg', 0)
    bo_def = cell.get('defaults', {}).get('bond', {}).get('bo', 1)

    atoms = m['atoms']
    N = len(atoms)
    Z = np.array([a.get('z', z_def) for a in atoms], dtype=int)
    # commonchem は既定値と異なる原子にだけ chg を書く。ニトロ基は [N+](=O)[O-] と
    # 電荷分離した形で格納されており、chg を落とすと中性 N の明示価数が 4 になって
    # SanitizeMol が必ず失敗する（= そのポリマーが全 RU 丸ごと欠測する）。
    CHG = np.array([a.get('chg', chg_def) for a in atoms], dtype=int)

    conformers = m['conformers']
    if len(conformers) > 1:
        print(f'  [warn] {class_name}:{uuid} conformers={len(conformers)} 件 → 先頭のみ処理')
        skip['multi_conformers'] += 1
    uw = np.array(conformers[0]['coords'], dtype=float)  # 後で in-place アンラップ

    bonds = [(b['atoms'][0], b['atoms'][1], int(round(b.get('bo', bo_def)))) for b in m['bonds']]

    rp = next(e for e in m['extensions'] if e.get('name') == 'radonpy_extention')
    rp_atoms = rp['atoms']
    mol_id = [a['mol_id'] for a in rp_atoms]
    resname = [a['ResidueName'] for a in rp_atoms]
    resnum = [a['ResidueNumber'] for a in rp_atoms]

    box = _find_box(m)
    L = np.array([box['xhi'] - box['xlo'], box['yhi'] - box['ylo'], box['zhi'] - box['zlo']])
    tilt = [box.get(k, 0.0) for k in ('xy', 'xz', 'yz')]
    if any(abs(t) > 1e-9 for t in tilt) and class_name not in _warned_triclinic_classes:
        _warned_triclinic_classes.add(class_name)
        print(f'  [warn] {class_name}:{uuid} トリクリニック箱を検出（xy/xz/yz={tilt}）'
              f'→ 直交前提の PBC アンラップは不正確な可能性（このクラスは以降サイレント）')
        skip['triclinic_box'] += 1

    adj = defaultdict(list)
    for i, j, _ in bonds:
        adj[i].append(j)
        adj[j].append(i)

    # PBC アンラップ（結合BFS・最小イメージ）
    seen = np.zeros(N, dtype=bool)
    for start in range(N):
        if seen[start]:
            continue
        seen[start] = True
        stack = [start]
        while stack:
            a = stack.pop()
            for b in adj[a]:
                if not seen[b]:
                    delta = uw[b] - uw[a]
                    delta -= L * np.round(delta / L)
                    uw[b] = uw[a] + delta
                    seen[b] = True
                    stack.append(b)

    units = defaultdict(list)
    for idx in range(N):
        units[(mol_id[idx], resnum[idx])].append(idx)

    for (mid, rn), idxs in sorted(units.items()):
        if not any(resname[k].startswith('RU') for k in idxs):
            continue  # TU=末端キャップは学習対象外
        idx_set = set(idxs)
        old2new = {o: n for n, o in enumerate(idxs)}
        rw = Chem.RWMol()
        for o in idxs:
            at = Chem.Atom(int(Z[o]))
            if CHG[o]:
                at.SetFormalCharge(int(CHG[o]))
            rw.AddAtom(at)
        severed = []
        for i, j, bo in bonds:
            if i in idx_set and j in idx_set:
                rw.AddBond(old2new[i], old2new[j], _BT.get(bo, Chem.BondType.SINGLE))
            elif i in idx_set:
                severed.append((i, j, bo))
            elif j in idx_set:
                severed.append((j, i, bo))
        cap_pos = []
        for inside, outside, bo in severed:
            if bo != 1:
                # 残基境界を跨ぐ結合が非単結合（二重/三重/芳香族）＝ H 単結合キャップは
                # 価数を変える近似。共役主鎖クラスで SanitizeMol 失敗の主因になりうるため
                # 可視化のみ行い、キャップ自体は従来どおり単結合のまま継続する。
                skip['non_single_severed'] += 1
            hidx = rw.AddAtom(Chem.Atom(1))
            rw.AddBond(old2new[inside], hidx, Chem.BondType.SINGLE)
            v = uw[outside] - uw[inside]
            nv = float(np.linalg.norm(v))
            v = v / nv * 1.09 if nv > 1e-6 else np.array([1.09, 0.0, 0.0])
            cap_pos.append(uw[inside] + v)
        mol = rw.GetMol()
        conf = Chem.Conformer(mol.GetNumAtoms())
        for n, o in enumerate(idxs):
            conf.SetAtomPosition(n, Point3D(*uw[o]))
        for k, cp in enumerate(cap_pos):
            conf.SetAtomPosition(len(idxs) + k, Point3D(*cp))
        mol.AddConformer(conf, assignId=True)
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            # 握り潰すと RDKit の stderr 行だけが残ってクラス/uuid の文脈が失われ、
            # skip サマリ上は全件成功に見えてしまう。必ず種別ごとに計上する。
            kind = type(e).__name__
            skip[f'sanitize:{class_name}:{kind}'] += 1
            if (class_name, kind) not in _warned_sanitize:
                _warned_sanitize.add((class_name, kind))
                print(f'  [warn] {class_name}:{uuid} SanitizeMol 失敗 ({kind}): {e}'
                      f' → この単位は除外（同種は以降サイレント、件数は skip に計上）')
            continue
        yield f'{mid}:{rn}', mol


def _find_box(o):
    if isinstance(o, dict):
        if 'xhi' in o and 'xlo' in o:
            return o
        for v in o.values():
            r = _find_box(v)
            if r:
                return r
    elif isinstance(o, list):
        for v in o:
            r = _find_box(v)
            if r:
                return r
    return None


# ── ソース展開（--data_dir / --sources → 具体的なソース列） ─────────────────────

def expand_sources(data_dir: Optional[str], sources, classes) -> list:
    out = []
    if data_dir:
        out += sorted(glob.glob(str(Path(data_dir) / '*.tar.gz')))
    for s in (sources or []):
        # ローカルグロブは展開、URL やそのままのパスはスルー
        if any(ch in s for ch in '*?[') and not s.startswith('http'):
            out += sorted(glob.glob(s))
        else:
            out.append(s)
    if classes:
        cset = set(classes)
        # tar.gz のみクラス名（拡張子除いた stem）でフィルタ。URL/json はそのまま残す
        def keep(p):
            name = Path(p).name
            if name.endswith('.tar.gz'):
                return name[:-len('.tar.gz')] in cset
            return True
        out = [p for p in out if keep(p)]
    return out


# ── ソース（URL / ローカル tar.gz / 単一 json）を (uuid, cell_dict) で yield ──────

def iter_cells(source: str, max_cells: int = 0) -> Iterator[Tuple[str, dict]]:
    n = 0
    if source.endswith('.json') and not source.endswith('.tar.gz'):
        uuid = Path(source).stem.replace('_eq', '')
        with open(source, encoding='utf-8') as f:
            yield uuid, json.load(f)
        return
    # tar.gz（URL or ローカル）をストリーム展開
    if source.startswith('http://') or source.startswith('https://'):
        fobj = urllib.request.urlopen(source)  # 302→CDN 自動追従（オンライン時のみ）
    else:
        fobj = open(source, 'rb')              # ローカル = 完全オフライン
    try:
        with tarfile.open(fileobj=fobj, mode='r|gz') as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith('.json'):
                    continue
                data = tar.extractfile(member).read()  # stream モードは前進前に読み切る
                uuid = Path(member.name).stem.replace('_eq', '')
                yield uuid, json.loads(data)
                n += 1
                if max_cells and n >= max_cells:
                    break
    finally:
        fobj.close()


# ── 任意: 物性CSV（UUID 結合）─────────────────────────────────────────────────

def load_csv_map(paths, cols) -> dict:
    """UUID → {col: value} を複数 CSV から構築。UUID 列は必須（PolyOmics CSV の 1列目）。"""
    m: dict = {}
    want = set(cols) if cols else {'smiles_list'}
    for p in paths:
        with open(p, newline='', encoding='utf-8') as f:
            r = csvmod.DictReader(f)
            if 'UUID' not in (r.fieldnames or []):
                print(f'  [csv] {p}: UUID 列なし → スキップ')
                continue
            avail = [c for c in want if c in r.fieldnames]
            for row in r:
                u = row['UUID']
                if u:
                    m[u] = {c: row.get(c) for c in avail}
    print(f'  [csv] {len(m):,} UUID を読み込み（列: {sorted(want)}）')
    return m


# ── ビルド ────────────────────────────────────────────────────────────────────

def build(args) -> None:
    srcs = expand_sources(args.data_dir, args.sources, args.classes)
    if not srcs:
        raise SystemExit('ソースが空です（--data_dir か --sources を指定）')
    print(f'sources ({len(srcs)}):')
    for s in srcs:
        print('  -', s)
    csv_map = load_csv_map(args.csv, args.csv_cols) if args.csv else {}

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(out), map_size=args.map_size_gb * 1024 ** 3,
                    subdir=False, meminit=False, map_async=True)
    txn = env.begin(write=True)

    n_ok = 0
    n_cells = 0
    # 新規カウンタは 0 件でもサマリに表示されるよう事前に seed しておく（観測性のため）
    skip = Counter({'multi_molecules': 0, 'multi_conformers': 0, 'triclinic_box': 0,
                    'non_single_severed': 0, 'topology_fallback': 0})
    for source in srcs:
        class_name = Path(source).name
        if class_name.endswith('.tar.gz'):
            class_name = class_name[:-len('.tar.gz')]
        for uuid, cell in iter_cells(source, max_cells=args.max_cells_per_class):
            n_cells += 1
            props = csv_map.get(uuid)
            for u, (tag, mol) in enumerate(
                    cut_units_from_cell(cell, uuid=uuid, class_name=class_name, skip=skip)):
                if args.per_cell_stride > 1 and (u % args.per_cell_stride) != 0:
                    continue
                if args.max_atoms and mol.GetNumAtoms() > args.max_atoms:
                    skip['max_atoms'] += 1
                    continue
                try:
                    dd = mol_to_data(mol, sid=f'{uuid}:{tag}', use_rwpe=True, use_lappe=False)
                except Exception as e:
                    skip[f'mol_to_data:{type(e).__name__}'] += 1
                    continue
                if props:  # 任意: CSV 由来の物性/正式SMILESを付与（ConformerDataset は無視）
                    dd['csv'] = props
                if args.precompute_topology:
                    # dist_mat/triplets/quartets（edge_index のみに依存する不変値）を
                    # 事前計算して埋め込む。ConformerDataset は「保存済みなら使う・
                    # 無ければ計算」のフォールバックなので、指定なしの既存 lmdb は
                    # 従来どおり dataset.py 側で計算される（後方互換）。
                    try:
                        import torch
                        from poly3d.model.pos_bias import compute_graph_distance
                        from poly3d.model.geo_losses import build_angle_triplets, build_dihedral_quartets

                        n = dd['atom_type_idx'].shape[0]
                        edge_index_t = torch.from_numpy(dd['edge_index'])
                        dist_mat = compute_graph_distance(edge_index_t, n, max_dist=4).to(torch.int8)
                        triplets = build_angle_triplets(edge_index_t, n)
                        quartets = build_dihedral_quartets(edge_index_t, n)
                        dd['dist_mat'] = dist_mat.numpy()
                        dd['triplets'] = triplets.numpy()
                        dd['quartets'] = quartets.numpy()
                    except Exception:
                        # トポロジー事前計算に失敗しても単位分子自体は保存する
                        # （dataset.py 側が起動時にフォールバック計算する）
                        skip['topology_fallback'] += 1
                        dd.pop('dist_mat', None)
                        dd.pop('triplets', None)
                        dd.pop('quartets', None)
                txn.put(f'{n_ok:09d}'.encode('ascii'), pickle.dumps(dd))
                n_ok += 1
                if n_ok % 10_000 == 0:
                    txn.commit()
                    txn = env.begin(write=True)
            if n_cells % 20 == 0:
                print(f'  cells={n_cells} records={n_ok} skip={dict(skip)}', flush=True)
    txn.commit()
    with env.begin(write=True) as t:
        t.put(b'__len__', str(n_ok).encode('ascii'))
    env.sync()
    env.close()
    print(f'完了: {n_ok:,} records from {n_cells:,} cells → {out}  skip={dict(skip)}')


def parse_args():
    p = argparse.ArgumentParser(description='PolyOmics 平衡化構造 → poly3d 処理済み lmdb（オフライン対応）')
    p.add_argument('--data_dir', default=None,
                   help='ローカルの MD_snapshot_JSON フォルダ（直下の *.tar.gz を全処理）＝オフライン')
    p.add_argument('--sources', nargs='+', default=None,
                   help='URL / ローカル tar.gz / 単一 _eq.json / グロブ（複数可）')
    p.add_argument('--classes', nargs='+', default=None,
                   help='tar.gz のクラス名でフィルタ（例: PG PI PEST）')
    p.add_argument('--csv', nargs='+', default=None,
                   help='任意: 物性CSV（UUID 結合）。無くても学習 lmdb は作れる')
    p.add_argument('--csv_cols', nargs='+', default=None,
                   help='CSV から付与する列（既定 smiles_list）。例: smiles_list temp tacticity')
    p.add_argument('--out_path', required=True)
    p.add_argument('--per_cell_stride', type=int, default=1,
                   help='1 セル内で単位を N 個ごとに間引く（配座相関を減らしつつ容量抑制）')
    p.add_argument('--max_cells_per_class', type=int, default=0, help='0=全セル')
    p.add_argument('--max_atoms', type=int, default=0, help='0=無制限（単位あたり原子数上限）')
    p.add_argument('--map_size_gb', type=int, default=50)
    p.add_argument('--precompute_topology', action='store_true',
                   help='dist_mat/triplets/quartets を前処理時に計算して lmdb 保存＝学習高速化'
                        '（デフォルト無効。既存 lmdb との後方互換のため、指定しない限り付与しない）')
    return p.parse_args()


if __name__ == '__main__':
    build(parse_args())
