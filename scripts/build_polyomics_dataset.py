"""PolyOmics(RadonPy) 平衡化構造 → poly3d 処理済み lmdb ビルダ。

PolyOmics の各 `<uuid>_eq.json`（1 非晶質セル = 同一繰返し単位の多数コピー）から
繰返し単位(RU)を 1 つずつ切り出し、鎖文脈で緩和された 3D 配座を持つ H 込みの単位
分子として `mol_to_data` に通し、既存 ConformerDataset と同形式の lmdb に格納する。

キモは「同一 SMILES に対する多数の非晶質配座」がそのまま学習アンサンブルになること
（実測: 1 セルで同一単位 ~180 配座、重原子 GetBestRMS 中央値 ~3 A の広がり）。

ソースは全て**ストリーム**で読むので tar.gz をローカルに溜め込まない:
  - URL   :  https://huggingface.co/datasets/yhayashi1986/PolyOmics/resolve/main/MD_snapshot_JSON/<CLASS>.tar.gz
  - ローカル tar.gz / 単一 _eq.json も可（smoke 用）。

例:
  # PG クラスを HF から直接ストリームして lmdb 化（各セル 3 単位ごとに間引き）
  python scripts/build_polyomics_dataset.py \
      --sources https://huggingface.co/datasets/yhayashi1986/PolyOmics/resolve/main/MD_snapshot_JSON/PG.tar.gz \
      --out_path data/polyomics_PG.lmdb --per_cell_stride 3 --max_atoms 288

  # ローカル 1 ファイルで smoke
  python scripts/build_polyomics_dataset.py --sources <path>/xxxx_eq.json --out_path /tmp/smoke.lmdb
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
import tarfile
import urllib.request
from collections import defaultdict, Counter
from pathlib import Path
from typing import Iterator, Tuple

import lmdb
import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from poly3d.model.features import mol_to_data

_BT = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}


# ── セル JSON → 単位 RDKit mol の切り出し ───────────────────────────────────────

def cut_units_from_cell(cell: dict) -> Iterator[Tuple[str, "Chem.Mol"]]:
    """1 セルの commonchem JSON から RU 単位を (residue_tag, H込みmol) で yield。"""
    m = cell['molecules'][0]
    z_def = cell.get('defaults', {}).get('atom', {}).get('z', 6)
    bo_def = cell.get('defaults', {}).get('bond', {}).get('bo', 1)

    atoms = m['atoms']
    N = len(atoms)
    Z = np.array([a.get('z', z_def) for a in atoms], dtype=int)
    uw = np.array(m['conformers'][0]['coords'], dtype=float)  # 後で in-place アンラップ

    bonds = [(b['atoms'][0], b['atoms'][1], int(round(b.get('bo', bo_def)))) for b in m['bonds']]

    rp = next(e for e in m['extensions'] if e.get('name') == 'radonpy_extention')
    rp_atoms = rp['atoms']
    mol_id = [a['mol_id'] for a in rp_atoms]
    resname = [a['ResidueName'] for a in rp_atoms]
    resnum = [a['ResidueNumber'] for a in rp_atoms]

    box = _find_box(m)
    L = np.array([box['xhi'] - box['xlo'], box['yhi'] - box['ylo'], box['zhi'] - box['zlo']])

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
            rw.AddAtom(Chem.Atom(int(Z[o])))
        severed = []
        for i, j, bo in bonds:
            if i in idx_set and j in idx_set:
                rw.AddBond(old2new[i], old2new[j], _BT.get(bo, Chem.BondType.SINGLE))
            elif i in idx_set:
                severed.append((i, j))
            elif j in idx_set:
                severed.append((j, i))
        cap_pos = []
        for inside, outside in severed:
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
        except Exception:
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


# ── ソース（URL / ローカル tar.gz / 単一 json）を (uuid, cell_dict) で yield ──────

def iter_cells(source: str, max_cells: int = 0) -> Iterator[Tuple[str, dict]]:
    n = 0
    if source.endswith('_eq.json') or (source.endswith('.json') and 'tar' not in source):
        uuid = Path(source).stem.replace('_eq', '')
        with open(source, encoding='utf-8') as f:
            yield uuid, json.load(f)
        return
    # tar.gz（URL or ローカル）をストリーム展開
    if source.startswith('http://') or source.startswith('https://'):
        fobj = urllib.request.urlopen(source)  # 302→CDN 自動追従
    else:
        fobj = open(source, 'rb')
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
    fobj.close()


# ── ビルド ────────────────────────────────────────────────────────────────────

def build(args) -> None:
    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(out), map_size=args.map_size_gb * 1024 ** 3,
                    subdir=False, meminit=False, map_async=True)
    txn = env.begin(write=True)

    n_ok = 0
    n_cells = 0
    skip = Counter()
    for source in args.sources:
        for uuid, cell in iter_cells(source, max_cells=args.max_cells_per_class):
            n_cells += 1
            for u, (tag, mol) in enumerate(cut_units_from_cell(cell)):
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
    p = argparse.ArgumentParser(description='PolyOmics 平衡化構造 → poly3d 処理済み lmdb')
    p.add_argument('--sources', nargs='+', required=True,
                   help='tar.gz の URL / ローカルパス / 単一 _eq.json（複数可）')
    p.add_argument('--out_path', required=True)
    p.add_argument('--per_cell_stride', type=int, default=1,
                   help='1 セル内で単位を N 個ごとに間引く（配座相関を減らしつつ容量抑制）')
    p.add_argument('--max_cells_per_class', type=int, default=0, help='0=全セル')
    p.add_argument('--max_atoms', type=int, default=0, help='0=無制限（単位あたり原子数上限）')
    p.add_argument('--map_size_gb', type=int, default=50)
    return p.parse_args()


if __name__ == '__main__':
    build(parse_args())
