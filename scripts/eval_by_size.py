"""
VAE 再構築品質を分子サイズ（原子数）別に分解して表示する。

evaluate_vae.py と同じ μ デコード RMSD を、原子数ビンごとに集計する。
「小分子は再構築できるが大分子で崩れる」かどうかを可視化するための診断。

さらに、**末端（グラフ直径端）に関する評価指標**を併せて集計する。
OPoly26 の GT はポリマー文脈から切り出した配座で末端（連結点）が外向きだが、
末端は H キャップされ「どの原子が連結点か」はラベル化されていない。
そこでトポロジーから末端性を創発させる: 各分子のグラフ距離（結合ホップ）が
最大となる原子ペア＝**グラフ直径の両端**を「末端プロキシ」とみなし、その
再構築品質（末端間距離が GT どおり外に届いているか等）をラベル不要で定量化する。

  python scripts/eval_by_size.py --checkpoint <ckpt> --val_lmdb <lmdb> [--max_batches N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

from poly3d.data.dataset import make_dataloader
from poly3d.model.builder import build_cond_encoder, build_vae
from poly3d.model.pos_bias import compute_graph_distance

# evaluate_vae の単一分子指標・モデルロードを再利用
from evaluate_vae import _kabsch_rmsd_single, _bond_rmse, _load_models


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--val_lmdb', required=True)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--max_atoms', type=int, default=240)
    p.add_argument('--max_batches', type=int, default=0)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', type=str, default=None, help='結果 JSON の保存先')
    return p.parse_args()


def _kabsch_align_single(P: torch.Tensor, Q: torch.Tensor):
    """P を Q に最適回転・並進アライメントした後の (P_rot, Qc) を返す。

    P, Q: (n, 3) fp32。両者を重心原点へ揃えた座標系で比較する
    （evaluate_vae._kabsch_rmsd_single と同一の Kabsch 実装。こちらは RMSD の
    スカラーではなくアライン後座標を返し、特定原子の位置誤差を測れるようにする）。
    """
    Pc = P - P.mean(dim=0, keepdim=True)
    Qc = Q - Q.mean(dim=0, keepdim=True)
    H = Pc.transpose(0, 1) @ Qc                      # (3, 3)
    H = H + 1e-6 * torch.eye(3, device=P.device)     # 特異値縮退対策
    U, _S, Vh = torch.linalg.svd(H)
    d = torch.sign(torch.det(Vh.transpose(0, 1) @ U.transpose(0, 1)))
    D = torch.eye(3, device=P.device)
    D[2, 2] = d
    R = Vh.transpose(0, 1) @ D @ U.transpose(0, 1)   # (3, 3)  Q ≈ R P
    P_rot = Pc @ R.transpose(0, 1)
    return P_rot, Qc


def _endpoint_metrics(pos_pred, pos_gt, edge_index, s, e) -> dict:
    """分子 [s, e) の末端（グラフ直径端）指標を計算する。

    トポロジーのみで末端プロキシを特定する:
      - 分子ローカルの edge_index から非クランプ・グラフ距離行列 D(n×n) を求め、
        (i*, j*) = argmax D を「直径端＝末端プロキシ」とする。
      - argmax は行優先で最初の最大要素を採る → (小さい i*, 次に小さい j*) を
        決定的タイブレークとして採用する（同点でも実行間で結果がぶれない）。

    全原子（H を含む）で計算する: 末端は H キャップされているため連結点そのものは
    判別できないが、H キャップを含めた「鎖の両端領域」こそがグラフ距離的に最遠であり、
    その末端が外に届いているか（内に巻き込んでいないか）を測るには全原子での
    直径端が最も素直なプロキシとなる。重原子限定にすると末端 H が落ち、直径が
    連結点手前で切れて末端“外向き”の評価が鈍る。

    Returns
    -------
    dict: e2e_gt, e2e_pred, e2e_abs_err, e2e_ratio, endpoint_pos_err（いずれも Å）
    """
    n = e - s
    # 分子内エッジのみ抽出（両端点が同一分子内なので src での絞り込みで十分）→ ローカル化
    mask = (edge_index[0] >= s) & (edge_index[0] < e)
    local_ei = edge_index[:, mask] - s
    D = compute_graph_distance(local_ei, n, max_dist=None)   # 非クランプ完全ホップ距離

    flat = int(torch.argmax(D.reshape(-1)).item())
    i_star = flat // n
    j_star = flat % n
    gi, gj = s + i_star, s + j_star

    e2e_gt = (pos_gt[gi] - pos_gt[gj]).norm().item()
    e2e_pred = (pos_pred[gi] - pos_pred[gj]).norm().item()
    e2e_abs_err = abs(e2e_pred - e2e_gt)
    e2e_ratio = e2e_pred / e2e_gt if e2e_gt > 1e-8 else float('nan')

    # 末端“配置”の正しさ（rmsd と分離）: 全原子で Kabsch アライン後、末端 2 原子の位置誤差平均
    P_rot, Qc = _kabsch_align_single(pos_pred[s:e], pos_gt[s:e])
    endpoint_pos_err = (0.5 * ((P_rot[i_star] - Qc[i_star]).norm()
                               + (P_rot[j_star] - Qc[j_star]).norm())).item()

    return {
        'e2e_gt': e2e_gt,
        'e2e_pred': e2e_pred,
        'e2e_abs_err': e2e_abs_err,
        'e2e_ratio': e2e_ratio,
        'endpoint_pos_err': endpoint_pos_err,
    }


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if args.device == 'auto' else torch.device(args.device))

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cond_encoder, vae, margs = _load_models(ckpt, device)

    loader = make_dataloader(
        args.val_lmdb, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, max_atoms=args.max_atoms,
        precompute_topology=True,
        # MDS 学習モデルは評価も足場必須。旧 ckpt は margs.mds_init=False で従来経路
        # （_load_models が mds_init を setdefault(False) するため KeyError にならない）。
        mds_init=getattr(margs, 'mds_init', False),
    )

    rows = []     # (natoms, rmsd, bond_rmse)
    ep_rows = []  # (natoms, e2e_gt, e2e_pred, e2e_abs_err, e2e_ratio, endpoint_pos_err)
    for bi, batch in enumerate(loader):
        if args.max_batches and bi >= args.max_batches:
            break
        if batch is None:
            continue
        batch = batch.to(device)
        _, e_cond, cond = cond_encoder(
            batch.atom_type_idx, batch.hyb_idx, batch.atom_cont,
            batch.bond_type_idx, batch.bond_cont, batch.edge_index,
            rwpe=getattr(batch, 'rwpe', None),
            lappe=getattr(batch, 'lappe', None), batch=batch.batch,
        )
        mu, _ = vae.encoder(cond, batch.pos, batch.edge_index, e_cond, batch.batch,
                            dist_mat=getattr(batch, 'dist_mat', None))
        pos_mu = vae.decode(mu, cond, batch.edge_index, e_cond, batch.batch,
                            dist_mat=getattr(batch, 'dist_mat', None),
                            init_scaffold=getattr(batch, 'init_scaffold', None)).float()
        pos_gt = batch.pos.float()
        ptr = batch.ptr
        for m in range(ptr.numel() - 1):
            s, e = int(ptr[m]), int(ptr[m + 1])
            if e - s < 2:
                continue
            rmsd = _kabsch_rmsd_single(pos_mu[s:e], pos_gt[s:e])
            bond = _bond_rmse(pos_mu, pos_gt, batch.edge_index, s, e)
            rows.append((e - s, rmsd, bond))

            ep = _endpoint_metrics(pos_mu, pos_gt, batch.edge_index, s, e)
            ep_rows.append((e - s, ep['e2e_gt'], ep['e2e_pred'], ep['e2e_abs_err'],
                            ep['e2e_ratio'], ep['endpoint_pos_err']))

    if not rows:
        print('分子なし')
        return

    natoms = torch.tensor([r[0] for r in rows], dtype=torch.float64)
    rmsd = torch.tensor([r[1] for r in rows], dtype=torch.float64)
    bond = torch.tensor([r[2] for r in rows], dtype=torch.float64)

    # 末端指標（分子ごと）
    ep_natoms = torch.tensor([r[0] for r in ep_rows], dtype=torch.float64)
    e2e_gt = torch.tensor([r[1] for r in ep_rows], dtype=torch.float64)
    e2e_abs = torch.tensor([r[3] for r in ep_rows], dtype=torch.float64)
    e2e_ratio = torch.tensor([r[4] for r in ep_rows], dtype=torch.float64)
    endpoint_pos_err = torch.tensor([r[5] for r in ep_rows], dtype=torch.float64)

    print(f'checkpoint : {args.checkpoint}')
    print(f'val_lmdb   : {args.val_lmdb}')
    print(f'アーキ     : hidden={margs.vae_hidden_dim} enc={margs.enc_layers} '
          f'dec={margs.dec_layers} latent={margs.latent_dim}')
    print(f'分子数     : {len(rows)}')

    # ピアソン相関（原子数 vs 指標）
    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        denom = (a.norm() * b.norm())
        return (a @ b / denom).item() if denom > 0 else float('nan')
    print(f'\n相関 corr(natoms, rmsd) = {corr(natoms, rmsd):+.3f}')
    print(f'相関 corr(natoms, bond) = {corr(natoms, bond):+.3f}')

    bins = [0, 40, 60, 80, 100, 130, 170, 240, 10000]
    print(f'\n{"原子数帯":>12} {"n":>5} {"rmsd_med":>9} {"rmsd_p90":>9} '
          f'{"bond_med":>9} {"成功<1Å":>8}')
    print('  ' + '-' * 60)
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (natoms >= lo) & (natoms < hi)
        k = int(mask.sum())
        if k == 0:
            continue
        r = rmsd[mask]; bd = bond[mask]
        succ = (r < 1.0).float().mean().item() * 100
        label = f'[{lo},{hi if hi < 10000 else "+"})'
        print(f'{label:>12} {k:>5} {r.median():>9.3f} '
              f'{torch.quantile(r, 0.9):>9.3f} {bd.median():>9.3f} {succ:>7.0f}%')

    # ── 末端（グラフ直径端）指標 ─────────────────────────────────────────────
    print(f'\n相関 corr(natoms, e2e_abs_err) = {corr(ep_natoms, e2e_abs):+.3f}')
    print(f'\n末端（グラフ直径端＝末端プロキシ）指標  |  {len(ep_rows)} 分子')
    print(f'  e2e_abs_err      : median={e2e_abs.median():.3f} '
          f'mean={e2e_abs.mean():.3f} p90={torch.quantile(e2e_abs, 0.9):.3f} Å')
    print(f'  endpoint_pos_err : median={endpoint_pos_err.median():.3f} '
          f'mean={endpoint_pos_err.mean():.3f} p90={torch.quantile(endpoint_pos_err, 0.9):.3f} Å')
    print(f'  e2e_ratio        : median={e2e_ratio.median():.3f}  '
          f'(<1 = 末端が内に巻き＝外向き不足)')

    print(f'\n{"原子数帯":>12} {"n":>5} {"e2e_gt_med":>11} {"e2e_abs_med":>12} '
          f'{"e2e_ratio_med":>14} {"ep_pos_med":>11}')
    print('  ' + '-' * 72)
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (ep_natoms >= lo) & (ep_natoms < hi)
        k = int(mask.sum())
        if k == 0:
            continue
        label = f'[{lo},{hi if hi < 10000 else "+"})'
        print(f'{label:>12} {k:>5} {e2e_gt[mask].median():>11.3f} '
              f'{e2e_abs[mask].median():>12.3f} {e2e_ratio[mask].median():>14.3f} '
              f'{endpoint_pos_err[mask].median():>11.3f}')

    # ── JSON 出力（オプション）───────────────────────────────────────────────
    if args.out:
        def _bin_stats(nat, val, agg='median'):
            out = {}
            for i in range(len(bins) - 1):
                lo, hi = bins[i], bins[i + 1]
                mask = (nat >= lo) & (nat < hi)
                k = int(mask.sum())
                if k == 0:
                    continue
                label = f'[{lo},{hi if hi < 10000 else "+"})'
                v = val[mask]
                stat = v.median().item() if agg == 'median' else v.mean().item()
                out[label] = {'n': k, agg: stat}
            return out

        result = {
            'checkpoint': str(args.checkpoint),
            'val_lmdb': str(args.val_lmdb),
            'n_molecules': len(rows),
            'corr_natoms_rmsd': corr(natoms, rmsd),
            'corr_natoms_bond': corr(natoms, bond),
            'by_size': {},
            'endpoint': {
                'n_molecules': len(ep_rows),
                'e2e_abs_err': _to_list_agg(e2e_abs),
                'endpoint_pos_err': _to_list_agg(endpoint_pos_err),
                'e2e_ratio_median': e2e_ratio.median().item(),
                'corr_natoms_e2e_abs_err': corr(ep_natoms, e2e_abs),
                'by_size': {},
            },
        }
        # サイズ帯別（rmsd 系）
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            mask = (natoms >= lo) & (natoms < hi)
            k = int(mask.sum())
            if k == 0:
                continue
            label = f'[{lo},{hi if hi < 10000 else "+"})'
            r = rmsd[mask]; bd = bond[mask]
            result['by_size'][label] = {
                'n': k,
                'rmsd_median': r.median().item(),
                'rmsd_p90': torch.quantile(r, 0.9).item(),
                'bond_median': bd.median().item(),
                'success_rmsd<1A': (r < 1.0).float().mean().item(),
            }
        # サイズ帯別（末端系）
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            mask = (ep_natoms >= lo) & (ep_natoms < hi)
            k = int(mask.sum())
            if k == 0:
                continue
            label = f'[{lo},{hi if hi < 10000 else "+"})'
            result['endpoint']['by_size'][label] = {
                'n': k,
                'e2e_gt_median': e2e_gt[mask].median().item(),
                'e2e_abs_err_median': e2e_abs[mask].median().item(),
                'e2e_ratio_median': e2e_ratio[mask].median().item(),
                'endpoint_pos_err_median': endpoint_pos_err[mask].median().item(),
            }
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                  encoding='utf-8')
        print(f'\n結果を保存: {args.out}')


def _to_list_agg(t: torch.Tensor) -> dict:
    """torch テンソルから median/mean/p90 の dict を作る（nan は除外）。"""
    v = t[t == t]
    if v.numel() == 0:
        return {'median': float('nan'), 'mean': float('nan'), 'p90': float('nan')}
    return {
        'median': v.median().item(),
        'mean': v.mean().item(),
        'p90': torch.quantile(v, 0.9).item(),
    }


if __name__ == '__main__':
    main()
