"""
生成ペナルティ切り分け診断
============================

DiT 生成の妥当性 pass 率 (dit_v2 = 0.44) が VAE recon 天井 (v3c = 0.70) に
届かない 0.26 差の「原因」を、同一デコーダに 3 経路の潜在を通して比較すること
で切り分ける。

  recon : encode(実配座)→μ を decode     … encoder posterior 多様体（天井）
  dit   : N(0,I)→ODE→z0 を decode        … DiT が作る潜在（生成の実体）
  prior : z~N(0,I) を直接 decode          … off-manifold の下限（最悪ケース）

pass/fail の bool ではなく、各配座について **量的な幾何健全性** を測る:
  - clash : グラフ距離>=3 の非結合ペアの貫入深さ (thr - dist) [Å] の最大/本数
  - bond  : 結合長が [0.7, 2.6]Å を外れた逸脱量 [Å] の最大/本数
  - 失敗帰属 : その配座が clash で落ちたか / bond で落ちたか

さらに潜在空間で off-manifold を直接確認:
  - 各配座の per-atom 潜在 L2 ノルムの平均（recon < dit < prior なら off-manifold）

出力: mode ごと・サイズ帯ごとの集計 + per-conformer 生データ JSON。

使い方:
  "C:/Users/shanu/anaconda3/envs/polygen/python.exe" scripts/diag_gen_penalty.py \
      --checkpoint     runs/polyomics_vae_v3c/vae_best.pt \
      --dit_checkpoint runs/polyomics_dit_v2/dit_best.pt \
      --val_lmdb       data/polyomics_PG_val.lmdb \
      --n_gen 50 --out runs/polyomics_dit_v2/diag_gen_penalty.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lmdb  # noqa: F401  (eval_ensemble 経由で使うが明示 import で環境確認)
import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

# eval_ensemble の機構を再利用（uuid グルーピング / 生成 / トポロジー / RDKit）
from eval_ensemble import (
    _PT,
    build_rdmol,
    build_uuid_index,
)
from evaluate_vae import _load_models
from poly3d.data.dataset import ConformerDataset, collate_fn


# ══════════════════════════════════════════════════════════════════════════════
# 量的な幾何健全性（bool でなく貫入深さ・逸脱量を返す）
# ══════════════════════════════════════════════════════════════════════════════

def geom_health(pos: np.ndarray, rvdw: np.ndarray, nonbond_mask: np.ndarray,
                bi: np.ndarray, bj: np.ndarray, sanitize_ok: bool,
                clash_factor: float = 0.6,
                bond_lo: float = 0.7, bond_hi: float = 2.6) -> dict:
    """1 配座の幾何健全性を量的に測る。

    妥当性ゲート（eval_ensemble.validity_of_conformer）と完全に同じ閾値・同じ
    nonbond_mask を用いるので、valid フラグはゲートの pass/fail と一致する。
    加えて「どれだけ」壊れているかを連続値で返す。
    """
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))              # (n, n)

    # ── clash（貫入深さ）──
    thr = (rvdw[:, None] + rvdw[None, :]) * clash_factor
    clash_mask = nonbond_mask & (dist < thr)
    n_clash = int(clash_mask.sum()) // 2               # 対称行列 → ペア数
    if n_clash > 0:
        pen = (thr - dist)[clash_mask]                 # 貫入深さ [Å]
        clash_max_pen = float(pen.max())
        clash_sum_pen = float(pen.sum()) / 2.0         # 対称の二重計上を補正
        clash_max_ratio = float((pen / thr[clash_mask]).max())
    else:
        clash_max_pen = clash_sum_pen = clash_max_ratio = 0.0

    # ── bond（境界からの逸脱量）──
    n_bad_bond = 0
    bond_max_ex = 0.0
    if bi.size > 0:
        bd = dist[bi, bj]
        ex = np.maximum(np.maximum(bond_lo - bd, bd - bond_hi), 0.0)  # >0=逸脱
        n_bad_bond = int((ex > 0).sum())
        bond_max_ex = float(ex.max()) if ex.size else 0.0

    fail_clash = n_clash > 0
    fail_bond = n_bad_bond > 0
    valid = (not fail_clash) and (not fail_bond) and sanitize_ok
    return {
        'n_clash': n_clash,
        'clash_max_pen': clash_max_pen,
        'clash_sum_pen': clash_sum_pen,
        'clash_max_ratio': clash_max_ratio,
        'n_bad_bond': n_bad_bond,
        'bond_max_ex': bond_max_ex,
        'fail_clash': fail_clash,
        'fail_bond': fail_bond,
        'valid': valid,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 生成（pos に加えて per-conformer 潜在ノルムも返す）
# ══════════════════════════════════════════════════════════════════════════════

def _chunks(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


@torch.no_grad()
def generate_pos_latent(gen_datas: list, mode: str, cond_encoder, vae, flow,
                        device, batch_size: int, n_steps: int):
    """gen_datas から生成座標と per-conformer 潜在ノルムを返す。

    戻り値: (pos_list, latnorm_list)
      pos_list     : 各配座 (n,3) fp32 cpu テンソル
      latnorm_list : 各配座の per-atom 潜在 L2 ノルムの平均（スカラー float）
    eval_ensemble.generate_positions と同じ生成経路。潜在 z を捕捉する点のみ拡張。
    """
    latent = vae.latent_dim
    pos_out, lat_out = [], []
    for chunk in _chunks(gen_datas, batch_size):
        batch = collate_fn(list(chunk)).to(device)
        _, e_cond, cond = cond_encoder(
            batch.atom_type_idx, batch.hyb_idx, batch.atom_cont,
            batch.bond_type_idx, batch.bond_cont, batch.edge_index,
            rwpe=getattr(batch, 'rwpe', None),
            lappe=getattr(batch, 'lappe', None), batch=batch.batch,
        )
        dm = getattr(batch, 'dist_mat', None)
        scaf = getattr(batch, 'init_scaffold', None)
        n_total = batch.num_nodes

        if mode == 'recon':
            mu, _ = vae.encoder(cond, batch.pos, batch.edge_index, e_cond,
                                batch.batch, dist_mat=dm)
            z = mu
        elif mode == 'prior':
            z = torch.randn(n_total, latent, device=device)
        elif mode == 'dit':
            z = flow.sample(n_atoms=n_total, cond=cond, batch=batch.batch,
                            n_steps=n_steps, edge_index=batch.edge_index,
                            device=device, dist_mat=dm)
        else:
            raise ValueError(f'unknown mode: {mode}')

        pos = vae.decode(z, cond, batch.edge_index, e_cond, batch.batch,
                         dist_mat=dm, init_scaffold=scaf).float()
        atom_norm = z.norm(dim=-1)                     # (n_total,) per-atom L2
        ptr = batch.ptr
        for m in range(ptr.numel() - 1):
            a, b = int(ptr[m]), int(ptr[m + 1])
            pos_out.append(pos[a:b].cpu())
            lat_out.append(float(atom_norm[a:b].mean().cpu()))
    return pos_out, lat_out


# ══════════════════════════════════════════════════════════════════════════════
# uuid 単位（3 モードを同一トポロジーで比較）
# ══════════════════════════════════════════════════════════════════════════════

MODES = ('recon', 'dit', 'prior')


def _mask_data(d) -> dict:
    """1 つの Data から妥当性判定に必要なマスク素材を抽出する。

    ★重要: PolyOmics は同一 uuid でも配座ごとに原子ラベリングが異なる（対称な
    水素などが入れ替わる）。したがってマスクは uuid で共有せず、生成配座を作った
    その Data 自身のトポロジー（＝デコード座標と同じラベリング）から作る。共有
    topo[0] を使うと 1 個の原子入れ替えで偽 clash / 偽 bad-bond が生じ、pass 率が
    崩壊する（診断で確認済み）。
    """
    an = d.atomic_nums.cpu().numpy()
    ei = d.edge_index
    und = ei[0] < ei[1]
    bi = ei[0][und].cpu().numpy()
    bj = ei[1][und].cpu().numpy()
    n = int(d.num_nodes)
    ei_np = ei.cpu().numpy()
    if hasattr(d, 'dist_mat'):
        nbm = d.dist_mat.cpu().numpy().astype(np.int16) >= 3
    else:
        nbm = np.ones((n, n), dtype=bool)
        nbm[ei_np[0], ei_np[1]] = False
        nbm[ei_np[1], ei_np[0]] = False
        np.fill_diagonal(nbm, False)
    rvdw = np.array([_PT.GetRvdw(int(z)) for z in an], dtype=np.float64)
    rdmol = build_rdmol(an, ei_np, d.bond_type_idx.cpu().numpy(),
                        d.atom_cont.cpu().numpy())
    return {'nbm': nbm, 'bi': bi, 'bj': bj, 'rvdw': rvdw,
            'sanitize_ok': rdmol is not None,
            'n_heavy': int((an > 1).sum())}


def eval_uuid(uuid: str, indices: list, ds: ConformerDataset,
              cond_encoder, vae, flow, device, args) -> list:
    """1 uuid につき 3 モードの per-conformer 健全性レコードのリストを返す。

    各生成配座は「その配座のトポロジー（自身のラベリング）」で妥当性判定する。
    """
    idxs = list(indices)
    n_gen = args.n_gen
    records = []
    for mode in MODES:
        # 生成元となる index 列（recon は先頭順、他は巡回）
        src = [idxs[k % len(idxs)] for k in range(n_gen)]
        # マスク素材は生成前に fresh 取得して確保（collate 消費と独立）
        masks = [_mask_data(ds[i]) for i in src]
        # 生成用は別 fetch（collate が topology を破壊消費するため）
        gen_datas = [ds[i] for i in src]

        pos_list, lat_list = generate_pos_latent(
            gen_datas, mode, cond_encoder, vae, flow,
            device, args.batch_size, args.n_steps)

        for p, ln, md in zip(pos_list, lat_list, masks):
            h = geom_health(p.cpu().numpy(), md['rvdw'], md['nbm'],
                            md['bi'], md['bj'], md['sanitize_ok'])
            h.update({'uuid': uuid, 'mode': mode,
                      'n_heavy': md['n_heavy'], 'lat_norm': ln})
            records.append(h)
    return records


# ══════════════════════════════════════════════════════════════════════════════
# 集計
# ══════════════════════════════════════════════════════════════════════════════

_SIZE_BINS = [0, 30, 50, 70, 10_000]


def _q(vals, q):
    a = np.asarray([v for v in vals if v == v], dtype=np.float64)
    return float(np.quantile(a, q)) if a.size else float('nan')


def _summarize(recs: list) -> dict:
    """レコード群（同一 mode）を集計。"""
    n = len(recs)
    if n == 0:
        return {'n': 0}
    valid = [r for r in recs if r['valid']]
    fc = [r for r in recs if r['fail_clash']]
    fb = [r for r in recs if r['fail_bond']]
    # clash 貫入は「clash が発生した配座のみ」で分布を見る（0 埋めしない）
    clash_pen_nz = [r['clash_max_pen'] for r in recs if r['n_clash'] > 0]
    bond_ex_nz = [r['bond_max_ex'] for r in recs if r['n_bad_bond'] > 0]
    return {
        'n': n,
        'pass_rate': len(valid) / n,
        'fail_clash_rate': len(fc) / n,
        'fail_bond_rate': len(fb) / n,
        'frac_with_clash': sum(1 for r in recs if r['n_clash'] > 0) / n,
        'clash_pen_median': _q(clash_pen_nz, 0.5),
        'clash_pen_p90': _q(clash_pen_nz, 0.9),
        'n_clash_median_all': _q([r['n_clash'] for r in recs], 0.5),
        'bond_ex_median': _q(bond_ex_nz, 0.5),
        'bond_ex_p90': _q(bond_ex_nz, 0.9),
        'lat_norm_median': _q([r['lat_norm'] for r in recs], 0.5),
        'lat_norm_mean': float(np.mean([r['lat_norm'] for r in recs])),
    }


def aggregate(all_recs: list) -> dict:
    out = {'overall': {}, 'by_size': {}}
    for mode in MODES:
        out['overall'][mode] = _summarize([r for r in all_recs if r['mode'] == mode])
    for i in range(len(_SIZE_BINS) - 1):
        lo, hi = _SIZE_BINS[i], _SIZE_BINS[i + 1]
        label = f'[{lo},{hi if hi < 10_000 else "+"})'
        band = {}
        for mode in MODES:
            sub = [r for r in all_recs
                   if r['mode'] == mode and lo <= r['n_heavy'] < hi]
            if sub:
                band[mode] = _summarize(sub)
        if band:
            out['by_size'][label] = band
    return out


def _print_report(agg: dict):
    print(f'\n{"=" * 78}')
    print('  生成ペナルティ切り分け診断  |  recon(天井) / dit(生成) / prior(下限)')
    print(f'{"=" * 78}')
    cols = ['pass_rate', 'fail_clash_rate', 'fail_bond_rate',
            'clash_pen_median', 'bond_ex_median', 'lat_norm_median']
    hdr = f'  {"mode":<7}' + ''.join(f'{c:>17}' for c in cols)
    print(hdr)
    print('  ' + '-' * 74)
    for mode in MODES:
        s = agg['overall'].get(mode, {})
        if not s or s.get('n', 0) == 0:
            continue
        line = f'  {mode:<7}'
        for c in cols:
            v = s.get(c, float('nan'))
            line += f'{v:>17.4f}'
        print(line)
    print(f'\n  ── サイズ帯別 pass率 / clash失敗率 ──')
    print(f'  {"帯":>10} {"n_uuidの原子帯":>4}'
          f'   recon(pass/clashfail)  dit(pass/clashfail)  prior(pass/clashfail)')
    for label, band in agg['by_size'].items():
        cells = []
        for mode in MODES:
            s = band.get(mode)
            if s:
                cells.append(f'{s["pass_rate"]:.2f}/{s["fail_clash_rate"]:.2f}')
            else:
                cells.append('  -  ')
        print(f'  {label:>10}          '
              f'{cells[0]:>18} {cells[1]:>18} {cells[2]:>18}')
    print()


# ══════════════════════════════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='生成ペナルティの clash/bond/潜在 切り分け')
    p.add_argument('--checkpoint', required=True, help='VAE ckpt (v3c)')
    p.add_argument('--dit_checkpoint', required=True, help='DiT ckpt (dit_v2)')
    p.add_argument('--val_lmdb', required=True)
    p.add_argument('--max_smiles', type=int, default=0, help='uuid 数上限 (0=全)')
    p.add_argument('--n_gen', type=int, default=50, help='mode ごとの生成本数')
    p.add_argument('--max_atoms', type=int, default=288)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_steps', type=int, default=100)
    p.add_argument('--device', default='auto')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default=None)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if args.device == 'auto' else torch.device(args.device))

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cond_encoder, vae, margs = _load_models(ckpt, device)

    from poly3d.model.builder import build_dit
    from poly3d.model.flow_matching import FlowMatching
    dck = torch.load(args.dit_checkpoint, map_location=device, weights_only=False)
    dargs = argparse.Namespace(**dck['args'])
    dit = build_dit(dargs).to(device)
    dit.load_state_dict(dck['flow'])
    flow = FlowMatching(dit, t_max=dargs.t_max)
    flow.eval()

    print(f'VAE  : {args.checkpoint}')
    print(f'DiT  : {args.dit_checkpoint}')
    print(f'val  : {args.val_lmdb}   device={device}')
    groups = build_uuid_index(args.val_lmdb, args.max_smiles)
    print(f'対象 uuid 数: {len(groups)}   n_gen/mode={args.n_gen}')

    ds = ConformerDataset(
        args.val_lmdb, max_atoms=args.max_atoms, precompute_topology=True,
        mds_init=getattr(margs, 'mds_init', False), topology_cache_size=8192,
    )

    all_recs = []
    for ui, (uuid, indices) in enumerate(groups.items()):
        recs = eval_uuid(uuid, indices, ds, cond_encoder, vae, flow, device, args)
        all_recs.extend(recs)
        if recs:
            bym = {m: [r for r in recs if r['mode'] == m] for m in MODES}
            pr = {m: (sum(r['valid'] for r in bym[m]) / len(bym[m])
                      if bym[m] else float('nan')) for m in MODES}
            nh = recs[0]['n_heavy']
            print(f'  [{ui + 1}/{len(groups)}] {uuid[:12]}… nheavy={nh:>3}  '
                  f'pass recon={pr["recon"]:.2f} dit={pr["dit"]:.2f} '
                  f'prior={pr["prior"]:.2f}')

    ds.close()

    if not all_recs:
        print('レコードが空でした')
        return

    agg = aggregate(all_recs)
    _print_report(agg)

    if args.out:
        result = {
            'checkpoint': str(args.checkpoint),
            'dit_checkpoint': str(args.dit_checkpoint),
            'val_lmdb': str(args.val_lmdb),
            'n_gen': args.n_gen,
            'aggregate': agg,
            'per_conformer': all_recs,
        }
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                  encoding='utf-8')
        print(f'結果を保存: {args.out}')


if __name__ == '__main__':
    main()
