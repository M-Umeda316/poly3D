"""
DiT 生成配座の品質評価スクリプト。

VAE 再構築（evaluate_vae.py / eval_by_size.py）ではなく、**DiT からサンプルした
生成配座**の品質を測る。sample.py の生成経路（cond_encoder → flow.sample →
vae.decode、n_conf ブロック対角複製・EGT/MDS ゲート）をそのまま lmdb 分子に適用し、
以下の 2 系統で評価する。

  ● 精度（分子ごと, 生成 vs 単一 GT）
      - best-of-N: n_conf 本のうち GT への Kabsch RMSD 最小値（_kabsch_rmsd_single）
      - その best 配座の bond_rmse（_bond_rmse）
      - median / p90 ＋ サイズ帯別 ＋ 成功率（min-RMSD < 1Å）

  ● 妥当性（生成配座ごと, 割合で集計）
      (i)   clash       : 非結合原子ペア（edge_index に無いペア）の最小距離が
                          clash_thresh 未満を含むか → clash なしの割合
      (ii)  bond サニティ: 結合ペア（edge_index）距離が絶対レンジ [0.7, 2.6] Å に
                          全て収まるか → OK の割合
      (iii) RDKit サニティ: 生成配座の原子番号（lmdb の atomic_nums）＋既知グラフ
                          （edge_index + bond_type_idx）から RWMol を構成し
                          Chem.SanitizeMol 成功割合。原子番号が取れない場合は
                          None 記録（スキップ）。

使い方:
    "C:/Users/shanu/anaconda3/envs/polygen/python.exe" scripts/eval_generation.py \
        --vae_checkpoint ./runs/gen_v1/A_nonEGT/vae_best.pt \
        --dit_checkpoint ./runs/gen_v1/dit_best.pt \
        --val_lmdb data/val.lmdb --n_conf 8 --out gen_eval.json

    # DiT 未学習の配管確認（ランダム初期化 flow）
    ... （--dit_checkpoint 省略）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Windows の cp932 コンソールでも Unicode を出力できるようにする
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

from rdkit import Chem
from rdkit import RDLogger

from poly3d.data.dataset import make_dataloader
from poly3d.model.builder import build_dit
from poly3d.model.flow_matching import FlowMatching
from poly3d.model.pos_bias import compute_graph_distance, mds_init_coords

# evaluate_vae の単一分子指標・VAE/cond_encoder ロードを再利用
from evaluate_vae import _kabsch_rmsd_single, _bond_rmse, _load_models

RDLogger.DisableLog('rdApp.*')   # SanitizeMol のエラーログを抑制（成否のみ使う）

# サイズ帯別 bins（eval_by_size.py と同一）
BINS = [0, 40, 60, 80, 100, 130, 170, 240, 10000]

# bond_type_idx → RDKit BondType（features.py: 0=SINGLE,1=DOUBLE,2=TRIPLE,3=AROMATIC,4=OTHER）
_BOND_ORDER = {
    0: Chem.BondType.SINGLE,
    1: Chem.BondType.DOUBLE,
    2: Chem.BondType.TRIPLE,
    3: Chem.BondType.AROMATIC,
    4: Chem.BondType.SINGLE,   # OTHER/UNKNOWN は単結合で代替
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='DiT 生成配座の品質評価')
    p.add_argument('--vae_checkpoint', type=str, required=True)
    p.add_argument('--dit_checkpoint', type=str, default=None,
                   help='省略時は smoke 用にランダム初期化 flow を構築（数値は無意味）')
    p.add_argument('--val_lmdb', type=str, required=True)
    p.add_argument('--n_conf', type=int, default=8)
    p.add_argument('--max_batches', type=int, default=0,
                   help='評価するバッチ数の上限（0 = 制限なし）')
    p.add_argument('--n_mols', type=int, default=0,
                   help='評価する分子数の上限（0 = 制限なし）')
    p.add_argument('--max_atoms', type=int, default=288)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--n_steps', type=int, default=100, help='ODE ステップ数')
    p.add_argument('--clash_thresh', type=float, default=1.0,
                   help='非結合ペアの clash 判定距離 [Å]')
    p.add_argument('--rmsd_thresh', type=float, default=1.0,
                   help='成功率算出用の best-of-N RMSD 閾値 [Å]')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', type=str, default=None, help='結果 JSON の保存先')
    return p.parse_args()


# ── flow（DiT）のロード / ランダム初期化構築 ─────────────────────────────────

def _load_flow(dit_ckpt_path, vae_args, device):
    """
    flow（FlowMatching）を構築して返す。

    - dit_ckpt_path 指定時: DiT チェックポイントの args から build_dit し、
      'flow' キー（LatentDiT の state_dict のみ）をロードする（sample.py と同一）。
    - 省略時: VAE args（latent_dim / hidden_dim を持つ）に DiT 既定値（train.py の
      argparse デフォルト）を補って build_dit し、**ランダム初期化 flow** を構築する。
      DiTTrainer.__init__ と同じ FlowMatching(dit, t_max=args.t_max) の形で作る。
    """
    if dit_ckpt_path is not None:
        dit_ckpt = torch.load(dit_ckpt_path, map_location=device, weights_only=False)
        dit_args = argparse.Namespace(**dit_ckpt['args'])
        dit = build_dit(dit_args).to(device)
        dit.load_state_dict(dit_ckpt['flow'])   # LatentDiT の weights のみ
        flow = FlowMatching(dit, t_max=getattr(dit_args, 't_max', 0.9))
        flow.eval()
        return flow, True

    # ── ランダム初期化 flow（smoke 用）──
    # build_dit が参照する属性: latent_dim, hidden_dim（VAE 由来）＋ DiT 既定値。
    rand_args = argparse.Namespace(**vars(vae_args))
    rand_args.time_dim = getattr(rand_args, 'time_dim', 64)
    rand_args.dit_hidden_dim = getattr(rand_args, 'dit_hidden_dim', 256)
    rand_args.dit_n_heads = getattr(rand_args, 'dit_n_heads', 8)
    rand_args.dit_n_layers = getattr(rand_args, 'dit_n_layers', 6)
    rand_args.use_pos_bias = getattr(rand_args, 'use_pos_bias', True)
    dit = build_dit(rand_args).to(device)
    flow = FlowMatching(dit, t_max=getattr(rand_args, 't_max', 0.9))
    flow.eval()
    return flow, False


# ── 単一分子の n_conf 本生成（sample.py の generate_conformers 経路を流用）───────

@torch.no_grad()
def _generate_for_molecule(mol, cond_encoder, vae, flow, margs, device,
                           n_conf, n_steps):
    """
    lmdb の単一分子 `mol`（バッチから切り出したノード・エッジテンソル群）について
    n_conf 本の配座を生成し、(n_conf, n_atoms, 3) の numpy 座標を返す。

    sample.py:generate_conformers と同じ:
      - cond / e_cond は分子 1 本分だけ計算しブロック対角に複製
      - グラフ距離 BFS は 1 本分だけ計算し block_diag で複製 → flow.sample に渡す
      - EGT/MDS ゲート（margs.egt_every / margs.mds_init）で decode 入力を切替
    """
    n_atoms = mol['atom_type_idx'].size(0)
    batch_idx = torch.zeros(n_atoms, dtype=torch.long, device=device)

    _, e_cond, cond = cond_encoder(
        mol['atom_type_idx'], mol['hyb_idx'], mol['atom_cont'],
        mol['bond_type_idx'], mol['bond_cont'], mol['edge_index'],
        rwpe=mol.get('rwpe'), lappe=mol.get('lappe'), batch=batch_idx,
    )

    edge_index = mol['edge_index']
    total_atoms = n_atoms * n_conf

    cond_rep = cond.repeat(n_conf, 1)
    e_cond_rep = e_cond.repeat(n_conf, 1) if e_cond.numel() > 0 else e_cond

    if edge_index.numel() > 0:
        offsets = (torch.arange(n_conf, device=device) * n_atoms).repeat_interleave(
            edge_index.size(1))
        edge_index_rep = edge_index.repeat(1, n_conf) + offsets.unsqueeze(0)
    else:
        edge_index_rep = edge_index.new_empty((2, 0))

    batch_rep = torch.arange(n_conf, device=device).repeat_interleave(n_atoms)

    dist_single = compute_graph_distance(edge_index, n_atoms)   # (n_atoms, n_atoms)
    dist_mat_rep = (torch.block_diag(*([dist_single] * n_conf))
                    if n_conf > 1 else dist_single)

    z0 = flow.sample(
        n_atoms=total_atoms, cond=cond_rep, batch=batch_rep, n_steps=n_steps,
        edge_index=edge_index_rep, device=device, dist_mat=dist_mat_rep,
    )

    # ── EGT/MDS ゲート（sample.py と同一・旧 ckpt 後方互換）──
    dec_uses_egt = getattr(margs, 'egt_every', 0) > 0
    use_mds = getattr(margs, 'mds_init', False)
    decode_dist_mat = dist_mat_rep if dec_uses_egt else None
    if use_mds:
        scaffold_single = mds_init_coords(edge_index, n_atoms)
        init_scaffold = (torch.cat([scaffold_single] * n_conf, dim=0)
                         if n_conf > 1 else scaffold_single)
    else:
        init_scaffold = None

    pos = vae.decode(z0, cond_rep, edge_index_rep, e_cond_rep, batch_rep,
                     dist_mat=decode_dist_mat, init_scaffold=init_scaffold)
    pos_np = pos.detach().cpu().numpy().astype(np.float64)
    return pos_np.reshape(n_conf, n_atoms, 3)


# ── 妥当性チェック ────────────────────────────────────────────────────────────

def _no_clash(pos: torch.Tensor, edge_index: torch.Tensor, thresh: float) -> bool:
    """非結合原子ペア（edge_index に無いペア）の最小距離が thresh 以上なら True。"""
    n = pos.size(0)
    if n < 2:
        return True
    dm = torch.cdist(pos, pos)
    bonded = torch.zeros(n, n, dtype=torch.bool, device=pos.device)
    if edge_index.numel() > 0:
        bonded[edge_index[0], edge_index[1]] = True
        bonded[edge_index[1], edge_index[0]] = True
    bonded |= torch.eye(n, dtype=torch.bool, device=pos.device)
    nonbonded = ~bonded
    if not nonbonded.any():
        return True
    return dm[nonbonded].min().item() >= thresh


def _bond_sane(pos: torch.Tensor, edge_index: torch.Tensor,
               lo: float = 0.7, hi: float = 2.6) -> bool:
    """結合ペア距離が全て [lo, hi] Å に収まれば True。"""
    if edge_index.numel() == 0:
        return True
    d = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)
    return bool(((d >= lo) & (d <= hi)).all().item())


def _rdkit_sane(atomic_nums, edge_index, bond_type_idx, pos_np) -> bool:
    """
    原子番号 + 既知グラフ（bond order 付き）+ 生成座標から RWMol を構成し
    Chem.SanitizeMol が成功すれば True。

    edge_index は両方向格納なので i<j でユニーク化。芳香族結合は両端原子に
    IsAromatic を立ててから sanitize する。
    """
    try:
        n = atomic_nums.shape[0]
        rw = Chem.RWMol()
        for z in atomic_nums.tolist():
            rw.AddAtom(Chem.Atom(int(z)))

        ei = edge_index.cpu().numpy()
        bt = bond_type_idx.cpu().numpy()
        seen = set()
        arom_atoms = set()
        for k in range(ei.shape[1]):
            i, j = int(ei[0, k]), int(ei[1, k])
            a, b = (i, j) if i < j else (j, i)
            if a == b or (a, b) in seen:
                continue
            seen.add((a, b))
            order = _BOND_ORDER.get(int(bt[k]), Chem.BondType.SINGLE)
            rw.AddBond(a, b, order)
            if order == Chem.BondType.AROMATIC:
                arom_atoms.update((a, b))
        for idx in arom_atoms:
            rw.GetAtomWithIdx(idx).SetIsAromatic(True)

        mol = rw.GetMol()
        conf = Chem.Conformer(n)
        for i in range(n):
            x, y, z = pos_np[i]
            conf.SetAtomPosition(i, (float(x), float(y), float(z)))
        mol.AddConformer(conf, assignId=True)

        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False


# ── 集計ユーティリティ ────────────────────────────────────────────────────────

def _bin_label(lo, hi) -> str:
    return f'[{lo},{hi if hi < 10000 else "+"})'


def _frac(mask_ok: torch.Tensor) -> float:
    return mask_ok.float().mean().item() if mask_ok.numel() > 0 else float('nan')


@torch.no_grad()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if args.device == 'auto' else torch.device(args.device))

    vae_ckpt = torch.load(args.vae_checkpoint, map_location=device, weights_only=False)
    cond_encoder, vae, margs = _load_models(vae_ckpt, device)
    flow, dit_loaded = _load_flow(args.dit_checkpoint, margs, device)

    loader = make_dataloader(
        args.val_lmdb, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, max_atoms=args.max_atoms,
        precompute_topology=False,   # 生成経路は per-molecule BFS を自前で行う
        mds_init=False,
    )

    print(f'vae_checkpoint : {args.vae_checkpoint}')
    print(f'dit_checkpoint : {args.dit_checkpoint if dit_loaded else "(ランダム初期化 flow)"}')
    print(f'val_lmdb       : {args.val_lmdb}')
    print(f'device         : {device}  |  n_conf={args.n_conf}  n_steps={args.n_steps}')
    print(f'アーキ         : hidden={margs.vae_hidden_dim} enc={margs.enc_layers} '
          f'dec={margs.dec_layers} latent={margs.latent_dim}')
    print('生成・評価中...')

    # 精度（分子ごと, best-of-N）
    acc_rows = []       # (natoms, best_rmsd, best_bond)
    # 妥当性（生成配座ごと）
    val_rows = []       # (natoms, no_clash(bool), bond_sane(bool), rdkit_ok(bool or None))

    n_mols_done = 0
    for bi, batch in enumerate(loader):
        if args.max_batches and bi >= args.max_batches:
            break
        if batch is None:
            continue
        batch = batch.to(device)
        ptr = batch.ptr
        ei = batch.edge_index

        for m in range(ptr.numel() - 1):
            if args.n_mols and n_mols_done >= args.n_mols:
                break
            s, e = int(ptr[m]), int(ptr[m + 1])
            n = e - s
            if n < 2:
                continue

            # ── 分子 m のノード・エッジを切り出しローカル化 ──
            emask = (ei[0] >= s) & (ei[0] < e)
            local_ei = ei[:, emask] - s
            mol = {
                'atom_type_idx': batch.atom_type_idx[s:e],
                'hyb_idx': batch.hyb_idx[s:e],
                'atom_cont': batch.atom_cont[s:e],
                'bond_type_idx': batch.bond_type_idx[emask],
                'bond_cont': batch.bond_cont[emask],
                'edge_index': local_ei,
            }
            if hasattr(batch, 'rwpe') and batch.rwpe is not None:
                mol['rwpe'] = batch.rwpe[s:e]
            if hasattr(batch, 'lappe') and batch.lappe is not None:
                mol['lappe'] = batch.lappe[s:e]

            pos_gt = batch.pos[s:e].float()
            atomic_nums = batch.atomic_nums[s:e] if hasattr(batch, 'atomic_nums') else None

            gen = _generate_for_molecule(mol, cond_encoder, vae, flow, margs,
                                         device, args.n_conf, args.n_steps)  # (n_conf,n,3)

            # ── 精度: best-of-N（GT への Kabsch RMSD 最小）──
            rmsds = []
            for c in range(gen.shape[0]):
                pc = torch.from_numpy(gen[c]).float().to(device)
                rmsds.append(_kabsch_rmsd_single(pc, pos_gt))
            best_c = int(np.argmin(rmsds))
            best_rmsd = float(rmsds[best_c])
            best_pos = torch.from_numpy(gen[best_c]).float().to(device)
            best_bond = _bond_rmse(best_pos, pos_gt, local_ei, 0, n)
            acc_rows.append((n, best_rmsd, best_bond))

            # ── 妥当性: 生成配座ごと ──
            for c in range(gen.shape[0]):
                pc = torch.from_numpy(gen[c]).float().to(device)
                no_clash = _no_clash(pc, local_ei, args.clash_thresh)
                bond_sane = _bond_sane(pc, local_ei)
                if atomic_nums is not None:
                    rdkit_ok = _rdkit_sane(atomic_nums.cpu().numpy(), local_ei,
                                           mol['bond_type_idx'], gen[c])
                else:
                    rdkit_ok = None
                val_rows.append((n, no_clash, bond_sane, rdkit_ok))

            n_mols_done += 1
        if args.n_mols and n_mols_done >= args.n_mols:
            break

    if not acc_rows:
        print('分子なし')
        return

    result = _summarize(args, margs, dit_loaded, acc_rows, val_rows)
    _print_report(result)

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                  encoding='utf-8')
        print(f'\n結果を保存: {args.out}')


def _summarize(args, margs, dit_loaded, acc_rows, val_rows) -> dict:
    a_nat = torch.tensor([r[0] for r in acc_rows], dtype=torch.float64)
    rmsd = torch.tensor([r[1] for r in acc_rows], dtype=torch.float64)
    bond = torch.tensor([r[2] for r in acc_rows], dtype=torch.float64)

    v_nat = torch.tensor([r[0] for r in val_rows], dtype=torch.float64)
    no_clash = torch.tensor([r[1] for r in val_rows], dtype=torch.bool)
    bond_sane = torch.tensor([r[2] for r in val_rows], dtype=torch.bool)
    rdkit_vals = [r[3] for r in val_rows]
    rdkit_available = any(v is not None for v in rdkit_vals)
    rdkit_ok = (torch.tensor([bool(v) for v in rdkit_vals], dtype=torch.bool)
                if rdkit_available else None)

    result = {
        'vae_checkpoint': str(args.vae_checkpoint),
        'dit_checkpoint': (str(args.dit_checkpoint) if dit_loaded
                           else '(random-init flow)'),
        'val_lmdb': str(args.val_lmdb),
        'n_conf': args.n_conf,
        'n_steps': args.n_steps,
        'clash_thresh': args.clash_thresh,
        'n_molecules': len(acc_rows),
        'n_conformers': len(val_rows),
        # 精度（best-of-N）全体
        'accuracy': {
            'best_rmsd_median': rmsd.median().item(),
            'best_rmsd_p90': torch.quantile(rmsd, 0.9).item(),
            'best_bond_median': bond[bond == bond].median().item()
            if (bond == bond).any() else float('nan'),
            f'success_min_rmsd<{args.rmsd_thresh}A':
                (rmsd < args.rmsd_thresh).float().mean().item(),
        },
        # 妥当性 全体
        'validity': {
            'no_clash_frac': _frac(no_clash),
            'bond_sane_frac': _frac(bond_sane),
            'rdkit_sane_frac': (_frac(rdkit_ok) if rdkit_available else None),
        },
        'by_size': {},
    }

    for i in range(len(BINS) - 1):
        lo, hi = BINS[i], BINS[i + 1]
        label = _bin_label(lo, hi)
        am = (a_nat >= lo) & (a_nat < hi)
        vm = (v_nat >= lo) & (v_nat < hi)
        ka = int(am.sum())
        kv = int(vm.sum())
        if ka == 0 and kv == 0:
            continue
        entry = {}
        if ka > 0:
            r = rmsd[am]
            bd = bond[am]
            bd = bd[bd == bd]
            entry.update({
                'n_molecules': ka,
                'best_rmsd_median': r.median().item(),
                'best_rmsd_p90': torch.quantile(r, 0.9).item(),
                'best_bond_median': bd.median().item() if bd.numel() > 0 else float('nan'),
                f'success_min_rmsd<{args.rmsd_thresh}A':
                    (r < args.rmsd_thresh).float().mean().item(),
            })
        if kv > 0:
            entry['n_conformers'] = kv
            entry['no_clash_frac'] = _frac(no_clash[vm])
            entry['bond_sane_frac'] = _frac(bond_sane[vm])
            entry['rdkit_sane_frac'] = (_frac(rdkit_ok[vm]) if rdkit_available else None)
        result['by_size'][label] = entry

    return result


def _print_report(r: dict) -> None:
    a = r['accuracy']
    v = r['validity']
    print(f'\n{"="*72}')
    print(f'  DiT 生成品質  |  {r["n_molecules"]} 分子 / {r["n_conformers"]} 配座 '
          f'(n_conf={r["n_conf"]})')
    print(f'{"="*72}')
    print('  [精度 best-of-N vs 単一GT]')
    print(f'    best_rmsd  : median={a["best_rmsd_median"]:.3f}  '
          f'p90={a["best_rmsd_p90"]:.3f} Å')
    print(f'    best_bond  : median={a["best_bond_median"]:.3f} Å')
    for k, val in a.items():
        if k.startswith('success'):
            print(f'    成功率 {k:<20}: {val*100:.1f}%')
    print('  [妥当性 生成配座ごとの割合]')
    print(f'    no_clash   : {v["no_clash_frac"]*100:.1f}%'
          if v['no_clash_frac'] == v['no_clash_frac'] else '    no_clash   : n/a')
    print(f'    bond_sane  : {v["bond_sane_frac"]*100:.1f}%'
          if v['bond_sane_frac'] == v['bond_sane_frac'] else '    bond_sane  : n/a')
    rf = v['rdkit_sane_frac']
    print(f'    rdkit_sane : {rf*100:.1f}%' if rf is not None else
          '    rdkit_sane : (skip / 原子番号なし)')

    print(f'\n{"原子数帯":>12} {"n_mol":>6} {"rmsd_med":>9} {"rmsd_p90":>9} '
          f'{"bond_med":>9} {"succ":>6} {"noClash":>8} {"bondOK":>7} {"rdkit":>7}')
    print('  ' + '-' * 82)
    for label, e in r['by_size'].items():
        nmol = e.get('n_molecules', 0)
        rmed = e.get('best_rmsd_median', float('nan'))
        rp90 = e.get('best_rmsd_p90', float('nan'))
        bmed = e.get('best_bond_median', float('nan'))
        succ = next((val for k, val in e.items() if k.startswith('success')), float('nan'))
        ncl = e.get('no_clash_frac', float('nan'))
        bok = e.get('bond_sane_frac', float('nan'))
        rdk = e.get('rdkit_sane_frac', None)
        rdk_s = f'{rdk*100:5.0f}%' if rdk is not None else '   n/a'
        print(f'{label:>12} {nmol:>6} {rmed:>9.3f} {rp90:>9.3f} {bmed:>9.3f} '
              f'{succ*100:>5.0f}% {ncl*100:>7.0f}% {bok*100:>6.0f}% {rdk_s:>7}')
    print()


if __name__ == '__main__':
    main()
