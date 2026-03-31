"""
PolyGen 学習スクリプト（2段階）

Stage 1: Structural VAE (ConditionalEncoder + VAE)
Stage 2: Latent DiT (Flow Matching)

【シングル GPU】
  "C:/Users/shanu/anaconda3/envs/polygen/python.exe" scripts/train.py \
      --stage vae \
      --train_lmdb D:/Dataset/OMol_base/OPoly26/processed/train.lmdb \
      --val_lmdb   D:/Dataset/OMol_base/OPoly26/processed/val.lmdb \
      --out_dir    ./runs/polygen_v1 --epochs 300

【マルチ GPU（単一ノード、4GPU の例）】
  torchrun --nproc_per_node=4 scripts/train.py \
      --stage vae ...

【マルチノード（2ノード × 4GPU = 8GPU の例）】
  # node0 (master)
  torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
      --master_addr=<ip> --master_port=29500 scripts/train.py --stage vae ...
  # node1
  torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
      --master_addr=<ip> --master_port=29500 scripts/train.py --stage vae ...

【TensorBoard】
  tensorboard --logdir ./runs/polygen_v1
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from poly3d.data.dataset import make_dataloader
from poly3d.model.cond_encoder import ConditionalEncoder
from poly3d.model.vae import StructuralVAE
from poly3d.model.dit import LatentDiT
from poly3d.model.flow_matching import FlowMatching
from poly3d.model.vae_loss import vae_loss


# ── 分散学習ユーティリティ ─────────────────────────────────────────────────────

def init_dist() -> tuple[int, int, int]:
    """
    torchrun が設定する環境変数から rank 情報を読み取り、
    NCCL プロセスグループを初期化する。

    Returns:
        (local_rank, global_rank, world_size)
        分散モードでない場合は (0, 0, 1)
    """
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank < 0:
        return 0, 0, 1   # シングル GPU / CPU

    dist.init_process_group(backend='nccl')
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size


def cleanup_dist() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def _unwrap(model: nn.Module) -> nn.Module:
    """DDP ラップを外して元のモジュールを返す。"""
    return model.module if isinstance(model, DDP) else model


def _all_reduce_dict(d: dict[str, float], n: int, device: torch.device) -> tuple[dict[str, float], int]:
    """
    各ランクの (sums_dict, n_batches) を全ランクで集計し、
    グローバル合計を返す。分散モードでなければ入力をそのまま返す。
    """
    if not dist.is_initialized():
        return d, n

    keys = sorted(d.keys())
    vals = torch.tensor([d[k] for k in keys] + [float(n)], device=device)
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    total_n = int(vals[-1].item())
    result = {k: vals[i].item() for i, k in enumerate(keys)}
    return result, total_n


# ── 引数解析 ──────────────────────────────────────────────────────────────────

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


# ── モデル生成ファクトリ ────────────────────────────────────────────────────────

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
    def __init__(self, args: argparse.Namespace, local_rank: int):
        self.args = args
        self.local_rank = local_rank
        self.out_dir = Path(args.out_dir)
        if is_main_process():
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = _resolve_device(args.device, local_rank)
        torch.manual_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))

        self.cond_encoder = build_cond_encoder(args).to(self.device)
        self.vae = build_vae(args).to(self.device)

        # DDP ラップ（分散モード時のみ）
        if dist.is_initialized():
            self.cond_encoder = DDP(self.cond_encoder, device_ids=[local_rank])
            self.vae = DDP(self.vae, device_ids=[local_rank])

        params = (list(self.cond_encoder.parameters())
                  + list(self.vae.parameters()))
        if is_main_process():
            n_params = sum(p.numel() for p in params if p.requires_grad)
            print(f'VAE パラメータ数: {n_params:,}')

        self.optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs)

        # DistributedSampler（分散モード時）
        self.train_sampler: Optional[DistributedSampler] = None
        if dist.is_initialized():
            from poly3d.data.dataset import ConformerDataset
            _ds = ConformerDataset(args.train_lmdb, max_atoms=args.max_atoms)
            self.train_sampler = DistributedSampler(_ds, shuffle=True, seed=args.seed)

        self.train_loader = make_dataloader(
            args.train_lmdb, batch_size=args.batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=args.num_workers, max_atoms=args.max_atoms,
            sampler=self.train_sampler,
        )
        self.val_loader = make_dataloader(
            args.val_lmdb, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, max_atoms=args.max_atoms,
        )

        self.start_epoch = 1
        self.best_val_loss = float('inf')
        if args.resume:
            self._load(args.resume)

        # TensorBoard（main process のみ）
        self.writer: Optional[SummaryWriter] = None
        if is_main_process():
            self.writer = SummaryWriter(log_dir=str(self.out_dir / 'tb_vae'))

    def _get_beta(self, epoch: int) -> float:
        a = self.args
        if a.beta_warmup_epochs <= 0:
            return a.beta_end
        progress = min(1.0, (epoch - 1) / a.beta_warmup_epochs)
        return a.beta_start + (a.beta_end - a.beta_start) * progress

    def _load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        _unwrap(self.cond_encoder).load_state_dict(ckpt['cond_encoder'])
        _unwrap(self.vae).load_state_dict(ckpt['vae'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_val_loss = ckpt.get('val_loss', float('inf'))
        if is_main_process():
            print(f'Resume: {path} (epoch {ckpt["epoch"]})')

    def _save(self, epoch: int, val_loss: float):
        if not is_main_process():
            return
        ckpt = {
            'epoch': epoch,
            'cond_encoder': _unwrap(self.cond_encoder).state_dict(),
            'vae': _unwrap(self.vae).state_dict(),
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

        pbar = tqdm(loader, desc='Train' if train else 'Val',
                    leave=False, dynamic_ncols=True, disable=not is_main_process())
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
                    self.args.grad_clip,
                )
                self.optimizer.step()

            for k, v in ld.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
            if is_main_process():
                pbar.set_postfix({k: f'{v:.4f}' for k, v in ld.items()})

        # 全ランクの sums / n を集計してグローバル平均を返す
        sums, n = _all_reduce_dict(sums, n, self.device)
        return {k: v / max(n, 1) for k, v in sums.items()}

    def run(self):
        log = self.out_dir / 'vae_log.csv'
        if is_main_process() and self.start_epoch == 1:
            with open(log, 'w') as f:
                f.write('epoch,beta,train_total,val_total,val_pos,val_kl,lr,elapsed\n')

        for epoch in range(self.start_epoch, self.args.epochs + 1):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            beta = self._get_beta(epoch)
            t0 = time.time()
            tr = self._run_epoch(self.train_loader, True, beta)
            va = self._run_epoch(self.val_loader, False, beta)
            self.scheduler.step()

            if is_main_process():
                elapsed = time.time() - t0
                lr = self.optimizer.param_groups[0]['lr']
                print(
                    f'[VAE] Epoch {epoch:4d}/{self.args.epochs} β={beta:.3f} | '
                    f'train={tr["total"]:.4f} | val={va["total"]:.4f} '
                    f'(pos={va.get("pos",0):.4f} kl={va.get("kl",0):.4f}) | '
                    f'lr={lr:.2e} | {elapsed:.1f}s'
                )

                # CSV ログ
                with open(log, 'a') as f:
                    f.write(f'{epoch},{beta:.4f},{tr.get("total",0):.6f},'
                            f'{va.get("total",0):.6f},{va.get("pos",0):.6f},'
                            f'{va.get("kl",0):.6f},{lr:.2e},{elapsed:.1f}\n')

                # TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar('Loss/train_total', tr.get('total', 0), epoch)
                    self.writer.add_scalar('Loss/val_total', va.get('total', 0), epoch)
                    for key in ('pos', 'bond', 'angle', 'dihedral', 'kl'):
                        if key in va:
                            self.writer.add_scalar(f'Loss/val_{key}', va[key], epoch)
                    self.writer.add_scalar('Params/lr', lr, epoch)
                    self.writer.add_scalar('Params/beta', beta, epoch)

                if epoch % self.args.save_every == 0 or epoch == self.args.epochs:
                    self._save(epoch, va['total'])

        if self.writer is not None:
            self.writer.close()


# ── DiT Trainer ────────────────────────────────────────────────────────────────

class DiTTrainer:
    def __init__(self, args: argparse.Namespace, local_rank: int):
        self.args = args
        self.local_rank = local_rank
        self.out_dir = Path(args.out_dir)
        if is_main_process():
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = _resolve_device(args.device, local_rank)
        torch.manual_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))

        # ConditionalEncoder + VAE Encoder を凍結ロード（DDP 不要）
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
        dit = build_dit(args).to(self.device)
        self.flow = FlowMatching(dit, t_max=args.t_max, p_selfcond=args.p_selfcond)

        # DDP は LatentDiT のみに適用。
        # FlowMatching.loss() 内で self.model(...)（DDP経由）を呼ぶため
        # gradient sync が正しく発動する。
        if dist.is_initialized():
            self.flow.model = DDP(dit, device_ids=[local_rank])

        if is_main_process():
            n_params = sum(p.numel() for p in dit.parameters() if p.requires_grad)
            print(f'DiT パラメータ数: {n_params:,}')

        self.optimizer = AdamW(self.flow.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs)

        # DistributedSampler
        self.train_sampler: Optional[DistributedSampler] = None
        if dist.is_initialized():
            from poly3d.data.dataset import ConformerDataset
            _ds = ConformerDataset(args.train_lmdb, max_atoms=args.max_atoms)
            self.train_sampler = DistributedSampler(_ds, shuffle=True, seed=args.seed)

        self.train_loader = make_dataloader(
            args.train_lmdb, batch_size=args.batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=args.num_workers, max_atoms=args.max_atoms,
            sampler=self.train_sampler,
        )
        self.val_loader = make_dataloader(
            args.val_lmdb, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, max_atoms=args.max_atoms,
        )

        self.start_epoch = 1
        self.best_val_loss = float('inf')
        if args.resume:
            self._load(args.resume)

        # TensorBoard（main process のみ）
        self.writer: Optional[SummaryWriter] = None
        if is_main_process():
            self.writer = SummaryWriter(log_dir=str(self.out_dir / 'tb_dit'))

    def _load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        _unwrap(self.flow.model).load_state_dict(ckpt['flow'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_val_loss = ckpt.get('val_loss', float('inf'))
        if is_main_process():
            print(f'Resume DiT: {path} (epoch {ckpt["epoch"]})')

    def _save(self, epoch: int, val_loss: float):
        if not is_main_process():
            return
        # DDP ラップを外してから保存（sample.py 側の load_state_dict と形式を合わせる）
        ckpt = {
            'epoch': epoch,
            'flow': _unwrap(self.flow.model).state_dict(),   # LatentDiT の weights のみ
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

        pbar = tqdm(loader, desc='Train' if train else 'Val',
                    leave=False, dynamic_ncols=True, disable=not is_main_process())
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
                z0 = mu   # deterministic latent

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
            if is_main_process():
                pbar.set_postfix({k: f'{v:.4f}' for k, v in ld.items()})

        sums, n = _all_reduce_dict(sums, n, self.device)
        return {k: v / max(n, 1) for k, v in sums.items()}

    def run(self):
        log = self.out_dir / 'dit_log.csv'
        if is_main_process() and self.start_epoch == 1:
            with open(log, 'w') as f:
                f.write('epoch,train_flow,val_flow,lr,elapsed\n')

        for epoch in range(self.start_epoch, self.args.epochs + 1):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            t0 = time.time()
            tr = self._run_epoch(self.train_loader, True)
            va = self._run_epoch(self.val_loader, False)
            self.scheduler.step()

            if is_main_process():
                elapsed = time.time() - t0
                lr = self.optimizer.param_groups[0]['lr']
                print(
                    f'[DiT] Epoch {epoch:4d}/{self.args.epochs} | '
                    f'train={tr["flow"]:.4f} | val={va["flow"]:.4f} | '
                    f'lr={lr:.2e} | {elapsed:.1f}s'
                )

                # CSV ログ
                with open(log, 'a') as f:
                    f.write(f'{epoch},{tr.get("flow",0):.6f},{va.get("flow",0):.6f},'
                            f'{lr:.2e},{elapsed:.1f}\n')

                # TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar('Loss/train_flow', tr.get('flow', 0), epoch)
                    self.writer.add_scalar('Loss/val_flow', va.get('flow', 0), epoch)
                    self.writer.add_scalar('Params/lr', lr, epoch)

                if epoch % self.args.save_every == 0 or epoch == self.args.epochs:
                    self._save(epoch, va['flow'])

        if self.writer is not None:
            self.writer.close()


# ── utils ───────────────────────────────────────────────────────────────────────

def _resolve_device(device_str: str, local_rank: int) -> torch.device:
    if dist.is_initialized():
        return torch.device(f'cuda:{local_rank}')
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


if __name__ == '__main__':
    local_rank, global_rank, world_size = init_dist()
    args = parse_args()

    if is_main_process():
        mode = f'{world_size} GPU(s)' if world_size > 1 else 'シングル GPU'
        device = _resolve_device(args.device, local_rank)
        print(f'Device: {device}  ({mode})')

    try:
        if args.stage == 'vae':
            VAETrainer(args, local_rank).run()
        else:
            DiTTrainer(args, local_rank).run()
    finally:
        cleanup_dist()
