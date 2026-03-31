"""
PolyGen 学習スクリプト（2段階）

Stage 1: Structural VAE (ConditionalEncoder + VAE)
Stage 2: Latent DiT (Flow Matching)

実行例:
  # Stage 1
  "C:/Users/shanu/anaconda3/envs/polygen/python.exe" train.py \\
      --stage vae \\
      --train_lmdb D:/Dataset/OMol_base/OPoly26/processed/train.lmdb \\
      --val_lmdb   D:/Dataset/OMol_base/OPoly26/processed/val.lmdb \\
      --out_dir    ./runs/polygen_v1 --epochs 300

  # Stage 2
  "C:/Users/shanu/anaconda3/envs/polygen/python.exe" train.py \\
      --stage dit \\
      --train_lmdb D:/Dataset/OMol_base/OPoly26/processed/train.lmdb \\
      --val_lmdb   D:/Dataset/OMol_base/OPoly26/processed/val.lmdb \\
      --out_dir    ./runs/polygen_v1 --epochs 600 \\
      --vae_checkpoint ./runs/polygen_v1/vae_best.pt
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from poly3d.data.dataset import make_dataloader
from poly3d.model.cond_encoder import ConditionalEncoder
from poly3d.model.vae import StructuralVAE
from poly3d.model.dit import LatentDiT
from poly3d.model.flow_matching import FlowMatching
from poly3d.model.vae_loss import vae_loss
from poly3d.model.geo_losses import build_angle_triplets, build_dihedral_quartets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['vae', 'dit'], required=True)
    p.add_argument('--train_lmdb', type=str, required=True)
    p.add_argument('--val_lmdb', type=str, required=True)
    p.add_argument('--out_dir', type=str, default='./runs/polygen')
    p.add_argument('--vae_checkpoint', type=str, default=None,
                   help='Stage 2 時に使う VAE チェックポイント')

    # ConditionalEncoder
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--edge_dim', type=int, default=64)
    p.add_argument('--cond_layers', type=int, default=4)
    p.add_argument('--atom_emb_dim', type=int, default=32)
    p.add_argument('--hyb_emb_dim', type=int, default=16)
    p.add_argument('--bond_emb_dim', type=int, default=16)
    p.add_argument('--use_rwpe', action='store_true', default=True)
    p.add_argument('--use_lappe', action='store_true', default=False)

    # VAE
    p.add_argument('--latent_dim', type=int, default=16)
    p.add_argument('--vae_hidden_dim', type=int, default=128)
    p.add_argument('--enc_layers', type=int, default=4)
    p.add_argument('--dec_layers', type=int, default=4)
    p.add_argument('--beta_start', type=float, default=0.0)
    p.add_argument('--beta_end', type=float, default=1.0)
    p.add_argument('--beta_warmup_epochs', type=int, default=50)
    p.add_argument('--w_pos', type=float, default=1.0)
    p.add_argument('--w_bond', type=float, default=1.0)
    p.add_argument('--w_angle', type=float, default=0.5)
    p.add_argument('--w_dihedral', type=float, default=0.1)

    # DiT
    p.add_argument('--dit_hidden_dim', type=int, default=256)
    p.add_argument('--dit_n_heads', type=int, default=8)
    p.add_argument('--dit_n_layers', type=int, default=6)
    p.add_argument('--time_dim', type=int, default=64)
    p.add_argument('--t_max', type=float, default=0.9)
    p.add_argument('--p_selfcond', type=float, default=0.5)
    p.add_argument('--use_pos_bias', action='store_true', default=True)

    # 学習共通
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--max_atoms', type=int, default=300)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--save_every', type=int, default=10)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def build_cond_encoder(args: argparse.Namespace) -> ConditionalEncoder:
    return ConditionalEncoder(
        hidden_dim=args.hidden_dim,
        edge_dim=args.edge_dim,
        n_layers=args.cond_layers,
        atom_emb_dim=args.atom_emb_dim,
        hyb_emb_dim=args.hyb_emb_dim,
        bond_emb_dim=args.bond_emb_dim,
        use_rwpe=args.use_rwpe,
        use_lappe=args.use_lappe,
    )


def build_vae(args: argparse.Namespace) -> StructuralVAE:
    return StructuralVAE(
        cond_dim=args.hidden_dim,
        edge_dim=args.edge_dim,
        hidden_dim=args.vae_hidden_dim,
        latent_dim=args.latent_dim,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
    )


def build_dit(args: argparse.Namespace) -> LatentDiT:
    return LatentDiT(
        latent_dim=args.latent_dim,
        cond_dim=args.hidden_dim,
        time_dim=args.time_dim,
        hidden_dim=args.dit_hidden_dim,
        n_heads=args.dit_n_heads,
        n_layers=args.dit_n_layers,
        use_pos_bias=args.use_pos_bias,
    )


# ── VAE Trainer ────────────────────────────────────────────────────────────────

class VAETrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = _get_device(args.device)
        torch.manual_seed(args.seed)

        self.cond_encoder = build_cond_encoder(args).to(self.device)
        self.vae = build_vae(args).to(self.device)

        params = list(self.cond_encoder.parameters()) + list(self.vae.parameters())
        n_params = sum(p.numel() for p in params if p.requires_grad)
        print(f'VAE パラメータ数: {n_params:,}')

        self.optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs)

        self.train_loader = make_dataloader(
            args.train_lmdb, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, max_atoms=args.max_atoms,
        )
        self.val_loader = make_dataloader(
            args.val_lmdb, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, max_atoms=args.max_atoms,
        )

        self.start_epoch = 1
        self.best_val_loss = float('inf')
        if args.resume:
            self._load(args.resume)

    def _get_beta(self, epoch: int) -> float:
        a = self.args
        if a.beta_warmup_epochs <= 0:
            return a.beta_end
        progress = min(1.0, (epoch - 1) / a.beta_warmup_epochs)
        return a.beta_start + (a.beta_end - a.beta_start) * progress

    def _load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.cond_encoder.load_state_dict(ckpt['cond_encoder'])
        self.vae.load_state_dict(ckpt['vae'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f'Resume: {path} (epoch {ckpt["epoch"]})')

    def _save(self, epoch: int, val_loss: float):
        ckpt = {
            'epoch': epoch,
            'cond_encoder': self.cond_encoder.state_dict(),
            'vae': self.vae.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'args': vars(self.args),
        }
        torch.save(ckpt, self.out_dir / f'vae_epoch{epoch:04d}.pt')
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(ckpt, self.out_dir / 'vae_best.pt')
            print(f'  → best 更新 (val_loss: {val_loss:.4f})')

    def _run_epoch(self, loader, train: bool, beta: float) -> dict:
        self.cond_encoder.train(train)
        self.vae.train(train)
        sums: dict = {}
        n = 0

        pbar = tqdm(loader, desc='Train' if train else 'Val', leave=False, dynamic_ncols=True)
        for batch in pbar:
            if batch is None:
                continue
            batch = batch.to(self.device)

            with torch.set_grad_enabled(train):
                h_cond, e_cond, cond = self.cond_encoder(
                    batch.atom_type_idx, batch.hyb_idx, batch.atom_cont,
                    batch.bond_type_idx, batch.bond_cont, batch.edge_index,
                    rwpe=getattr(batch, 'rwpe', None),
                    lappe=getattr(batch, 'lappe', None),
                    batch=batch.batch,
                )
                pos_pred, mu, logvar = self.vae(
                    cond, batch.pos, batch.edge_index, e_cond, batch.batch
                )

                # グラフトポロジーはバッチ内全ノードで edge_index を共有
                # triplets/quartets は各分子ごとに計算するとコストが高いので
                # バッチレベルで一括計算（結果はバッチ全体の loss に使う）
                num_nodes = batch.pos.size(0)
                loss, ld = vae_loss(
                    pos_pred, batch.pos, mu, logvar,
                    batch.edge_index, num_nodes,
                    beta=beta,
                    w_pos=self.args.w_pos,
                    w_bond=self.args.w_bond,
                    w_angle=self.args.w_angle,
                    w_dihedral=self.args.w_dihedral,
                )

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.cond_encoder.parameters()) + list(self.vae.parameters()),
                    self.args.grad_clip
                )
                self.optimizer.step()

            for k, v in ld.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
            pbar.set_postfix({k: f'{v:.4f}' for k, v in ld.items()})

        return {k: v / max(n, 1) for k, v in sums.items()}

    def run(self):
        log = self.out_dir / 'vae_log.csv'
        if self.start_epoch == 1:
            with open(log, 'w') as f:
                f.write('epoch,beta,train_total,val_total,val_pos,val_kl,lr,elapsed\n')

        for epoch in range(self.start_epoch, self.args.epochs + 1):
            beta = self._get_beta(epoch)
            t0 = time.time()
            tr = self._run_epoch(self.train_loader, True, beta)
            va = self._run_epoch(self.val_loader, False, beta)
            self.scheduler.step()

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]['lr']
            print(
                f'[VAE] Epoch {epoch:4d}/{self.args.epochs} β={beta:.3f} | '
                f'train={tr["total"]:.4f} | val={va["total"]:.4f} '
                f'(pos={va.get("pos",0):.4f} kl={va.get("kl",0):.4f}) | '
                f'lr={lr:.2e} | {elapsed:.1f}s'
            )
            with open(log, 'a') as f:
                f.write(f'{epoch},{beta:.4f},{tr.get("total",0):.6f},'
                        f'{va.get("total",0):.6f},{va.get("pos",0):.6f},'
                        f'{va.get("kl",0):.6f},{lr:.2e},{elapsed:.1f}\n')

            if epoch % self.args.save_every == 0 or epoch == self.args.epochs:
                self._save(epoch, va['total'])


# ── DiT Trainer ────────────────────────────────────────────────────────────────

class DiTTrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = _get_device(args.device)
        torch.manual_seed(args.seed)

        # ConditionalEncoder + VAE Encoder を凍結ロード
        if args.vae_checkpoint is None:
            raise ValueError('--vae_checkpoint を指定してください')
        vae_ckpt = torch.load(args.vae_checkpoint, map_location=self.device, weights_only=False)
        vae_args = argparse.Namespace(**vae_ckpt['args'])

        self.cond_encoder = build_cond_encoder(vae_args).to(self.device)
        self.cond_encoder.load_state_dict(vae_ckpt['cond_encoder'])
        self.cond_encoder.eval().requires_grad_(False)

        self.vae = build_vae(vae_args).to(self.device)
        self.vae.load_state_dict(vae_ckpt['vae'])
        self.vae.eval().requires_grad_(False)

        # DiT (学習対象)
        dit = build_dit(args)
        self.flow = FlowMatching(dit, t_max=args.t_max, p_selfcond=args.p_selfcond).to(self.device)

        n_params = sum(p.numel() for p in dit.parameters() if p.requires_grad)
        print(f'DiT パラメータ数: {n_params:,}')

        self.optimizer = AdamW(dit.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs)

        self.train_loader = make_dataloader(
            args.train_lmdb, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, max_atoms=args.max_atoms,
        )
        self.val_loader = make_dataloader(
            args.val_lmdb, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, max_atoms=args.max_atoms,
        )

        self.start_epoch = 1
        self.best_val_loss = float('inf')
        if args.resume:
            self._load(args.resume)

    def _load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.flow.load_state_dict(ckpt['flow'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f'Resume DiT: {path} (epoch {ckpt["epoch"]})')

    def _save(self, epoch: int, val_loss: float):
        ckpt = {
            'epoch': epoch,
            'flow': self.flow.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'args': vars(self.args),
        }
        torch.save(ckpt, self.out_dir / f'dit_epoch{epoch:04d}.pt')
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(ckpt, self.out_dir / 'dit_best.pt')
            print(f'  → best 更新 (val_loss: {val_loss:.4f})')

    def _run_epoch(self, loader, train: bool) -> dict:
        self.flow.train(train)
        sums: dict = {}
        n = 0

        pbar = tqdm(loader, desc='Train' if train else 'Val', leave=False, dynamic_ncols=True)
        for batch in pbar:
            if batch is None:
                continue
            batch = batch.to(self.device)

            with torch.no_grad():
                _, e_cond, cond = self.cond_encoder(
                    batch.atom_type_idx, batch.hyb_idx, batch.atom_cont,
                    batch.bond_type_idx, batch.bond_cont, batch.edge_index,
                    rwpe=getattr(batch, 'rwpe', None),
                    lappe=getattr(batch, 'lappe', None),
                    batch=batch.batch,
                )
                # DiT 学習用: mu を使う（reparameterize のノイズは不要）
                mu, _ = self.vae.encoder(
                    cond, batch.pos, batch.edge_index, e_cond, batch.batch
                )
                z0 = mu  # deterministic latent

            with torch.set_grad_enabled(train):
                loss, ld = self.flow.loss(
                    z0, cond, batch.batch, batch.edge_index
                )

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.flow.parameters(), self.args.grad_clip)
                self.optimizer.step()

            for k, v in ld.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
            pbar.set_postfix({k: f'{v:.4f}' for k, v in ld.items()})

        return {k: v / max(n, 1) for k, v in sums.items()}

    def run(self):
        log = self.out_dir / 'dit_log.csv'
        if self.start_epoch == 1:
            with open(log, 'w') as f:
                f.write('epoch,train_flow,val_flow,lr,elapsed\n')

        for epoch in range(self.start_epoch, self.args.epochs + 1):
            t0 = time.time()
            tr = self._run_epoch(self.train_loader, True)
            va = self._run_epoch(self.val_loader, False)
            self.scheduler.step()

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]['lr']
            print(
                f'[DiT] Epoch {epoch:4d}/{self.args.epochs} | '
                f'train={tr["flow"]:.4f} | val={va["flow"]:.4f} | '
                f'lr={lr:.2e} | {elapsed:.1f}s'
            )
            with open(log, 'a') as f:
                f.write(f'{epoch},{tr.get("flow",0):.6f},{va.get("flow",0):.6f},'
                        f'{lr:.2e},{elapsed:.1f}\n')

            if epoch % self.args.save_every == 0 or epoch == self.args.epochs:
                self._save(epoch, va['flow'])


# ── utils ───────────────────────────────────────────────────────────────────────

def _get_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


if __name__ == '__main__':
    args = parse_args()
    print(f'Device: {_get_device(args.device)}')

    if args.stage == 'vae':
        VAETrainer(args).run()
    else:
        DiTTrainer(args).run()
