"""
Structural VAE の復元品質評価スクリプト。

VAE そのものの構築能力（DiT を介さない再構築性能）を測定する。
Encoder で潜在変数の平均 μ を得て、**μ を決定論的にデコード**した座標を
主指標とする（サンプリングノイズを除いた「ベストケース再構築」）。

大域的な折り畳みに頑健な評価を行うため、指標を 2 系統に分離して集計する:

  ● 大域指標（分子全体の構造一致）
      - rmsd            : Kabsch アライメント後の RMSD [Å]
                          1 本の回転可能結合のズレで分子の半分が反転すると
                          爆発する（＝大域折り畳みに敏感）
  ● 局所指標（局所幾何の再現性・大域折り畳みに頑健）
      - local_dist_rmse : GT で近接する原子ペア（< cutoff Å）の距離 RMSE [Å]
      - bond_rmse       : 結合長 RMSE [Å]
      - angle_mae       : 結合角 MAE [deg]
      - dihedral_mae    : 二面角 MAE [deg]（(-π, π] に折り返し）

大域指標が悪くても局所指標が良ければ「局所幾何は再現できているが
大域折り畳みがずれている」と診断でき、モデル改善の方向づけに使える。

使い方:
    "C:/Users/shanu/anaconda3/envs/polygen/python.exe" scripts/evaluate_vae.py \
        --checkpoint ./runs/search_v1/A1_baseline/vae_best.pt \
        --val_lmdb   data/val.lmdb \
        --max_batches 50

    # サンプリング（z ~ q(z|x)）デコードも併せて評価
    ... --sample

    # 結果を JSON 保存
    ... --out eval_A1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Windows の cp932 コンソールでも Unicode を出力できるようにする
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

from poly3d.data.dataset import make_dataloader
from poly3d.model.builder import build_cond_encoder, build_vae
from poly3d.model.geo_losses import _angle_between, _dihedral, wrap_to_pi

RAD2DEG = 180.0 / 3.141592653589793


# ── 単一分子の幾何指標 ────────────────────────────────────────────────────────

def _kabsch_rmsd_single(P: torch.Tensor, Q: torch.Tensor) -> float:
    """P を Q に最適回転アライメントした後の RMSD [Å]。P, Q: (n, 3) fp32。"""
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
    return torch.sqrt((P_rot - Qc).pow(2).sum(dim=-1).mean()).item()


def _local_dist_rmse(P: torch.Tensor, Q: torch.Tensor, cutoff: float,
                     eps: float = 1e-8) -> float:
    """GT で近接する（< cutoff Å）原子ペアの距離 RMSE [Å]。大域折り畳みに頑健。"""
    d_pred = torch.cdist(P, P)
    d_gt = torch.cdist(Q, Q)
    local = (d_gt < cutoff) & (d_gt > eps)
    if not local.any():
        return float('nan')
    return torch.sqrt((d_pred[local] - d_gt[local]).pow(2).mean()).item()


def _bond_rmse(pos_pred, pos_gt, edge_index, s, e) -> float:
    """分子 [s, e) の結合長 RMSE [Å]。"""
    mask = (edge_index[0] >= s) & (edge_index[0] < e)
    if not mask.any():
        return float('nan')
    src = edge_index[0][mask]
    dst = edge_index[1][mask]
    d_pred = (pos_pred[src] - pos_pred[dst]).norm(dim=-1)
    d_gt = (pos_gt[src] - pos_gt[dst]).norm(dim=-1)
    return torch.sqrt((d_pred - d_gt).pow(2).mean()).item()


def _angle_mae_deg(pos_pred, pos_gt, triplets, s, e) -> float:
    """分子 [s, e) の結合角 MAE [deg]。"""
    if triplets is None or triplets.size(0) == 0:
        return float('nan')
    mask = (triplets[:, 0] >= s) & (triplets[:, 0] < e)
    if not mask.any():
        return float('nan')
    t = triplets[mask]
    i, j, k = t[:, 0], t[:, 1], t[:, 2]
    th_pred = _angle_between(pos_pred[i] - pos_pred[j], pos_pred[k] - pos_pred[j])
    th_gt = _angle_between(pos_gt[i] - pos_gt[j], pos_gt[k] - pos_gt[j])
    return (th_pred - th_gt).abs().mean().item() * RAD2DEG


def _dihedral_mae_deg(pos_pred, pos_gt, quartets, s, e) -> float:
    """分子 [s, e) の二面角 MAE [deg]。差は (-π, π] に折り返して評価。"""
    if quartets is None or quartets.size(0) == 0:
        return float('nan')
    mask = (quartets[:, 0] >= s) & (quartets[:, 0] < e)
    if not mask.any():
        return float('nan')
    q = quartets[mask]
    i, j, k, l = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    phi_pred = _dihedral(pos_pred[i], pos_pred[j], pos_pred[k], pos_pred[l])
    phi_gt = _dihedral(pos_gt[i], pos_gt[j], pos_gt[k], pos_gt[l])
    delta = wrap_to_pi(phi_pred - phi_gt)
    return delta.abs().mean().item() * RAD2DEG


def _per_mol_metrics(pos_pred, pos_gt, ptr, edge_index, triplets, quartets,
                     cutoff: float) -> list[dict]:
    """バッチ内の各分子について幾何指標を計算し、dict のリストを返す。"""
    pos_pred = pos_pred.float()
    pos_gt = pos_gt.float()
    out = []
    n_mol = ptr.numel() - 1
    for m in range(n_mol):
        s = int(ptr[m].item())
        e = int(ptr[m + 1].item())
        if e - s < 2:
            continue
        P = pos_pred[s:e]
        Q = pos_gt[s:e]
        out.append({
            'rmsd':            _kabsch_rmsd_single(P, Q),
            'local_dist_rmse': _local_dist_rmse(P, Q, cutoff),
            'bond_rmse':       _bond_rmse(pos_pred, pos_gt, edge_index, s, e),
            'angle_mae':       _angle_mae_deg(pos_pred, pos_gt, triplets, s, e),
            'dihedral_mae':    _dihedral_mae_deg(pos_pred, pos_gt, quartets, s, e),
        })
    return out


# ── 集計 ──────────────────────────────────────────────────────────────────────

def _agg(values: list[float]) -> dict:
    """nan を除外して median / mean / p90 を計算。"""
    t = torch.tensor([v for v in values if v == v], dtype=torch.float64)  # v==v で nan 除外
    if t.numel() == 0:
        return {'median': float('nan'), 'mean': float('nan'),
                'p90': float('nan'), 'n': 0}
    return {
        'median': torch.median(t).item(),
        'mean':   t.mean().item(),
        'p90':    torch.quantile(t, 0.90).item(),
        'n':      int(t.numel()),
    }


def _frac_below(values: list[float], thresh: float) -> float:
    t = torch.tensor([v for v in values if v == v], dtype=torch.float64)
    if t.numel() == 0:
        return float('nan')
    return (t < thresh).float().mean().item()


# ── メイン ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Structural VAE 復元品質評価（μ デコード主指標）')
    p.add_argument('--checkpoint', type=str, required=True,
                   help='VAE チェックポイント（vae_best.pt）')
    p.add_argument('--val_lmdb', type=str, required=True)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--max_atoms', type=int, default=240)
    p.add_argument('--max_batches', type=int, default=0,
                   help='評価するバッチ数の上限（0 = 全 val データ）')
    p.add_argument('--local_cutoff', type=float, default=5.0,
                   help='local_dist_rmse の近接ペア閾値 [Å]')
    p.add_argument('--rmsd_thresh', type=float, default=1.0,
                   help='成功率算出用の RMSD 閾値 [Å]')
    p.add_argument('--sample', action='store_true',
                   help='μ デコードに加え z~q(z|x) サンプリングデコードも評価')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', type=str, default=None, help='結果 JSON の保存先')
    return p.parse_args()


def _load_models(ckpt: dict, device: torch.device):
    ck_args = dict(ckpt['args'])
    # 後方互換: 後から追加された引数が無い旧チェックポイントへの対応
    ck_args.setdefault('use_rwpe', True)
    ck_args.setdefault('use_lappe', False)
    ck_args.setdefault('atom_emb_dim', 32)
    ck_args.setdefault('hyb_emb_dim', 16)
    ck_args.setdefault('bond_emb_dim', 16)
    model_args = argparse.Namespace(**ck_args)

    cond_encoder = build_cond_encoder(model_args).to(device)
    cond_encoder.load_state_dict(ckpt['cond_encoder'])
    cond_encoder.eval().requires_grad_(False)

    vae = build_vae(model_args).to(device)
    vae.load_state_dict(ckpt['vae'])
    vae.eval().requires_grad_(False)
    return cond_encoder, vae, model_args


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if args.device == 'auto' else torch.device(args.device))

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cond_encoder, vae, model_args = _load_models(ckpt, device)

    loader = make_dataloader(
        args.val_lmdb, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, max_atoms=args.max_atoms,
        precompute_topology=True,
    )

    mu_metrics: dict[str, list] = {k: [] for k in
                                   ('rmsd', 'local_dist_rmse', 'bond_rmse',
                                    'angle_mae', 'dihedral_mae')}
    sample_rmsd: list[float] = []

    print(f'checkpoint : {args.checkpoint}')
    print(f'val_lmdb   : {args.val_lmdb}')
    print(f'device     : {device}')
    print(f'アーキ     : hidden={model_args.vae_hidden_dim} '
          f'enc={model_args.enc_layers} dec={model_args.dec_layers} '
          f'latent={model_args.latent_dim}')
    print('評価中...')

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
            lappe=getattr(batch, 'lappe', None),
            batch=batch.batch,
        )
        mu, logvar = vae.encoder(cond, batch.pos, batch.edge_index, e_cond, batch.batch)

        # ── 主指標: μ を決定論的にデコード ──
        pos_mu = vae.decode(mu, cond, batch.edge_index, e_cond, batch.batch)

        ptr = batch.ptr
        triplets = getattr(batch, 'triplets', None)
        quartets = getattr(batch, 'quartets', None)
        rows = _per_mol_metrics(pos_mu, batch.pos, ptr, batch.edge_index,
                                triplets, quartets, args.local_cutoff)
        for r in rows:
            for k in mu_metrics:
                mu_metrics[k].append(r[k])

        # ── 参考: サンプリングデコード（z ~ q(z|x)）の RMSD のみ ──
        if args.sample:
            std = (0.5 * logvar).exp()
            z = mu + std * torch.randn_like(std)
            pos_s = vae.decode(z, cond, batch.edge_index, e_cond, batch.batch)
            pos_s = pos_s.float()
            pos_gt_f = batch.pos.float()
            for m in range(ptr.numel() - 1):
                s, e2 = int(ptr[m].item()), int(ptr[m + 1].item())
                if e2 - s < 2:
                    continue
                sample_rmsd.append(_kabsch_rmsd_single(pos_s[s:e2], pos_gt_f[s:e2]))

    # ── 集計 ──
    result = {
        'checkpoint': str(args.checkpoint),
        'n_molecules': len(mu_metrics['rmsd']),
        'local_cutoff': args.local_cutoff,
        'mu_decode': {k: _agg(v) for k, v in mu_metrics.items()},
        'success_rate': {
            f'rmsd<{args.rmsd_thresh}A': _frac_below(mu_metrics['rmsd'], args.rmsd_thresh),
            'local_dist_rmse<0.5A': _frac_below(mu_metrics['local_dist_rmse'], 0.5),
        },
    }
    if args.sample:
        result['sample_decode'] = {'rmsd': _agg(sample_rmsd)}

    _print_report(result)

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                  encoding='utf-8')
        print(f'\n結果を保存: {args.out}')
    return result


def _print_report(r: dict):
    units = {'rmsd': 'Å', 'local_dist_rmse': 'Å', 'bond_rmse': 'Å',
             'angle_mae': 'deg', 'dihedral_mae': 'deg'}
    labels = {
        'rmsd':            'rmsd (Kabsch, 大域)',
        'local_dist_rmse': 'local_dist_rmse (局所)',
        'bond_rmse':       'bond_rmse (局所)',
        'angle_mae':       'angle_mae (局所)',
        'dihedral_mae':    'dihedral_mae (局所)',
    }
    print(f'\n{"="*68}')
    print(f'  VAE 復元品質  |  μ デコード（主指標）  |  {r["n_molecules"]} 分子')
    print(f'{"="*68}')
    print(f'  {"指標":<26} {"median":>9} {"mean":>9} {"p90":>9}  単位')
    print(f'  {"-"*64}')
    for k in ('rmsd', 'local_dist_rmse', 'bond_rmse', 'angle_mae', 'dihedral_mae'):
        a = r['mu_decode'][k]
        print(f'  {labels[k]:<26} {a["median"]:>9.4f} {a["mean"]:>9.4f} '
              f'{a["p90"]:>9.4f}  {units[k]}')
    print(f'  {"-"*64}')
    for name, val in r['success_rate'].items():
        print(f'  成功率 {name:<22}: {val*100:>6.1f}%')
    if 'sample_decode' in r:
        a = r['sample_decode']['rmsd']
        print(f'\n  [参考] サンプリングデコード rmsd: '
              f'median={a["median"]:.4f} mean={a["mean"]:.4f} p90={a["p90"]:.4f} Å')
    print()


if __name__ == '__main__':
    evaluate(parse_args())
