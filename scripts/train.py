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

【推奨設定】
  RTX4060Ti 16GB / Ryzen9 7900X:
    --batch_size 64 --num_workers 8 --grad_accum 2
    (VAE: batch_size 64, DiT: batch_size 32 --grad_accum 4)

  RTX5000 Ada 32GB / Xeon Platinum 8558:
    --batch_size 128 --num_workers 16 --grad_accum 2
    (VAE: batch_size 128, DiT: batch_size 64 --grad_accum 2)
"""
from __future__ import annotations

import argparse
import os
import time
from contextlib import nullcontext
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
from poly3d.data.latent_dataset import make_latent_dataloader
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
    p.add_argument('--latent_lmdb', type=str, default=None,
                   help='precompute_latents.py で生成した事前エンコード済み LMDB。'
                        '指定すると DiT 学習時に ConditionalEncoder/VAE Encoder をスキップ')
    p.add_argument('--latent_val_lmdb', type=str, default=None,
                   help='val 用の事前エンコード済み LMDB')

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
    p.add_argument('--num_workers', type=int, default=8,
                   help='DataLoader ワーカー数。Ryzen9 7900X: 8-10、Xeon 8558: 16-20 推奨')
    p.add_argument('--save_every', type=int, default=10)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--seed', type=int, default=42)

    # パフォーマンス
    p.add_argument('--no_amp', action='store_true', default=False,
                   help='混合精度 (bf16) を無効化。デフォルト: AMP 有効')
    p.add_argument('--grad_accum', type=int, default=1,
                   help='勾配累積ステップ数。実効 batch_size = batch_size × grad_accum')
    p.add_argument('--prefetch_factor', type=int, default=4,
                   help='DataLoader ワーカーあたりのプリフェッチバッチ数')
    p.add_argument('--val_every', type=int, default=1,
                   help='何エポックごとに validation を実行するか（デフォルト: 毎エポック）')
    p.add_argument('--compile', action='store_true', default=False,
                   help='torch.compile でモデルをコンパイル（PyTorch 2.0+）')
    p.add_argument('--benchmark', type=int, default=0, metavar='N',
                   help='N バッチでベンチマークを実行して終了。セクション別タイミングを出力')
    p.add_argument('--tb_log_every', type=int, default=100,
                   help='TensorBoard に途中経過を書き込むステップ間隔（0 = エポック末のみ）')

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

        # 混合精度設定（CUDA 時のみ有効）
        self.amp_enabled = (not args.no_amp) and (self.device.type == 'cuda')
        self.amp_dtype = torch.bfloat16   # bf16: GradScaler 不要、数値安定

        self.cond_encoder = build_cond_encoder(args).to(self.device)
        self.vae = build_vae(args).to(self.device)

        # DDP ラップ（分散モード時のみ）
        if dist.is_initialized():
            self.cond_encoder = DDP(self.cond_encoder, device_ids=[local_rank])
            self.vae = DDP(self.vae, device_ids=[local_rank])

        params = (list(self.cond_encoder.parameters())
                  + list(self.vae.parameters()))
        # torch.compile（DDP 前に適用）
        if args.compile and hasattr(torch, 'compile'):
            self.cond_encoder = torch.compile(self.cond_encoder, dynamic=True)
            self.vae = torch.compile(self.vae, dynamic=True)
            if is_main_process():
                print('torch.compile: cond_encoder + vae をコンパイル')

        if is_main_process():
            n_params = sum(p.numel() for p in params if p.requires_grad)
            print(f'VAE パラメータ数: {n_params:,}')
            print(f'AMP: {"bf16 有効" if self.amp_enabled else "無効（fp32）"}')
            print(f'勾配累積: {args.grad_accum} step(s)  '
                  f'実効 batch_size = {args.batch_size * args.grad_accum}')
            print(f'Validation: {args.val_every} エポックごと')

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
            prefetch_factor=args.prefetch_factor,
        )
        self.val_loader = make_dataloader(
            args.val_lmdb, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, max_atoms=args.max_atoms,
            prefetch_factor=args.prefetch_factor,
        )

        self.start_epoch = 1
        self.best_val_loss = float('inf')
        self.global_step = 0
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
        n = 0   # 処理済みバッチ数
        n_batches = len(loader)
        accum = self.args.grad_accum
        tb_every = self.args.tb_log_every

        if train:
            self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(loader, desc='Train' if train else 'Val',
                    leave=False, dynamic_ncols=True, disable=not is_main_process())

        for step, batch in enumerate(pbar):
            if batch is None:
                continue
            batch = batch.to(self.device)

            with torch.set_grad_enabled(train):
                with torch.autocast(device_type=self.device.type,
                                    dtype=self.amp_dtype, enabled=self.amp_enabled):
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
                        # ワーカーで事前計算済み（None なら vae_loss 内でオンザフライ計算）
                        triplets=getattr(batch, 'triplets', None),
                        quartets=getattr(batch, 'quartets', None),
                    )

            if train:
                # 勾配累積: accum ステップで 1 回最適化
                (loss / accum).backward()

                is_last_step = (step + 1 >= n_batches)
                if (step + 1) % accum == 0 or is_last_step:
                    nn.utils.clip_grad_norm_(
                        list(self.cond_encoder.parameters()) + list(self.vae.parameters()),
                        self.args.grad_clip,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            # detach テンソル → float 変換（バッチ末でまとめて sync）
            for k, v in ld.items():
                v_f = v.item() if isinstance(v, torch.Tensor) else v
                sums[k] = sums.get(k, 0.0) + v_f
            n += 1
            if is_main_process():
                pbar.set_postfix({k: f'{sums[k]/n:.4f}' for k in sums})

            # TensorBoard 途中経過（train のみ）
            if train and is_main_process() and tb_every > 0 and n % tb_every == 0:
                self.global_step += tb_every
                if self.writer is not None:
                    for k in sums:
                        self.writer.add_scalar(
                            f'Step/train_{k}', sums[k] / n, self.global_step
                        )
                    self.writer.flush()

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

            run_val = (epoch % self.args.val_every == 0) or (epoch == self.args.epochs)
            va = self._run_epoch(self.val_loader, False, beta) if run_val else {}
            self.scheduler.step()

            if is_main_process():
                elapsed = time.time() - t0
                lr = self.optimizer.param_groups[0]['lr']
                val_str = (f'{va["total"]:.4f} '
                           f'(pos={va.get("pos",0):.4f} kl={va.get("kl",0):.4f})'
                           if run_val else '(skip)')
                print(
                    f'[VAE] Epoch {epoch:4d}/{self.args.epochs} β={beta:.3f} | '
                    f'train={tr["total"]:.4f} | val={val_str} | '
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
                    if run_val:
                        self.writer.add_scalar('Loss/val_total', va.get('total', 0), epoch)
                        for key in ('pos', 'bond', 'angle', 'dihedral', 'kl'):
                            if key in va:
                                self.writer.add_scalar(f'Loss/val_{key}', va[key], epoch)
                    self.writer.add_scalar('Params/lr', lr, epoch)
                    self.writer.add_scalar('Params/beta', beta, epoch)
                    self.writer.flush()   # エポックごとにディスクへ書き込み

                if (epoch % self.args.save_every == 0 or epoch == self.args.epochs) and run_val:
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

        # 混合精度設定
        self.amp_enabled = (not args.no_amp) and (self.device.type == 'cuda')
        self.amp_dtype = torch.bfloat16

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

        # torch.compile（DDP 前に適用）
        if args.compile and hasattr(torch, 'compile'):
            dit = torch.compile(dit, dynamic=True)
            if is_main_process():
                print('torch.compile: LatentDiT をコンパイル')

        self.flow = FlowMatching(dit, t_max=args.t_max, p_selfcond=args.p_selfcond)

        # DDP は LatentDiT のみに適用。
        # FlowMatching.loss() 内で self.model(...)（DDP経由）を呼ぶため
        # gradient sync が正しく発動する。
        if dist.is_initialized():
            self.flow.model = DDP(dit, device_ids=[local_rank])

        if is_main_process():
            n_params = sum(p.numel() for p in dit.parameters() if p.requires_grad)
            print(f'DiT パラメータ数: {n_params:,}')
            print(f'AMP: {"bf16 有効" if self.amp_enabled else "無効（fp32）"}')
            print(f'勾配累積: {args.grad_accum} step(s)  '
                  f'実効 batch_size = {args.batch_size * args.grad_accum}')

        self.optimizer = AdamW(self.flow.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs)

        # 事前エンコード済み LMDB が指定された場合は LatentDataset を使用
        self.use_latent = (args.latent_lmdb is not None)
        if self.use_latent and is_main_process():
            print(f'LatentDataset モード: ConditionalEncoder/VAE Encoder スキップ')
            print(f'  train: {args.latent_lmdb}')
            print(f'  val  : {args.latent_val_lmdb or args.val_lmdb}')

        # DistributedSampler
        self.train_sampler: Optional[DistributedSampler] = None
        if dist.is_initialized():
            if self.use_latent:
                from poly3d.data.latent_dataset import LatentDataset
                _ds = LatentDataset(args.latent_lmdb)
            else:
                from poly3d.data.dataset import ConformerDataset
                _ds = ConformerDataset(args.train_lmdb, max_atoms=args.max_atoms)
            self.train_sampler = DistributedSampler(_ds, shuffle=True, seed=args.seed)

        if self.use_latent:
            self.train_loader = make_latent_dataloader(
                args.latent_lmdb, batch_size=args.batch_size,
                shuffle=(self.train_sampler is None),
                num_workers=args.num_workers,
                sampler=self.train_sampler,
                prefetch_factor=args.prefetch_factor,
            )
            val_latent = args.latent_val_lmdb or args.latent_lmdb
            self.val_loader = make_latent_dataloader(
                val_latent, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
            )
        else:
            self.train_loader = make_dataloader(
                args.train_lmdb, batch_size=args.batch_size,
                shuffle=(self.train_sampler is None),
                num_workers=args.num_workers, max_atoms=args.max_atoms,
                sampler=self.train_sampler,
                prefetch_factor=args.prefetch_factor,
            )
            self.val_loader = make_dataloader(
                args.val_lmdb, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, max_atoms=args.max_atoms,
                prefetch_factor=args.prefetch_factor,
            )

        self.start_epoch = 1
        self.best_val_loss = float('inf')
        self.global_step = 0
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
        n_batches = len(loader)
        accum = self.args.grad_accum
        tb_every = self.args.tb_log_every

        if train:
            self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(loader, desc='Train' if train else 'Val',
                    leave=False, dynamic_ncols=True, disable=not is_main_process())

        for step, batch in enumerate(pbar):
            if batch is None:
                continue
            batch = batch.to(self.device)

            # dist_mat: ワーカーで事前計算済みのブロック対角距離行列
            dist_mat = getattr(batch, 'dist_mat', None)

            if self.use_latent:
                # LatentDataset モード: エンコード済みデータを直接使用
                z0   = batch.z0      # (N, latent_dim)
                cond = batch.cond    # (N, cond_dim)
                # e_cond はバッチ内で使用しない（FlowMatching は cond のみ使用）
            else:
                # 凍結モデル（勾配不要）: bf16 で推論し高速化
                with torch.no_grad():
                    with torch.autocast(device_type=self.device.type,
                                        dtype=self.amp_dtype, enabled=self.amp_enabled):
                        _, e_cond, cond = self.cond_encoder(
                            batch.atom_type_idx, batch.hyb_idx, batch.atom_cont,
                            batch.bond_type_idx, batch.bond_cont, batch.edge_index,
                            rwpe=getattr(batch, 'rwpe', None),
                            lappe=getattr(batch, 'lappe', None),
                            batch=batch.batch,
                        )
                        mu, _ = self.vae.encoder(
                            cond, batch.pos, batch.edge_index, e_cond, batch.batch
                        )
                        z0 = mu

            with torch.set_grad_enabled(train):
                with torch.autocast(device_type=self.device.type,
                                    dtype=self.amp_dtype, enabled=self.amp_enabled):
                    loss, ld = self.flow.loss(
                        z0, cond, batch.batch, batch.edge_index,
                        dist_mat=dist_mat,
                    )

            if train:
                (loss / accum).backward()

                is_last_step = (step + 1 >= n_batches)
                if (step + 1) % accum == 0 or is_last_step:
                    nn.utils.clip_grad_norm_(self.flow.parameters(), self.args.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            for k, v in ld.items():
                v_f = v.item() if isinstance(v, torch.Tensor) else v
                sums[k] = sums.get(k, 0.0) + v_f
            n += 1
            if is_main_process():
                pbar.set_postfix({k: f'{sums[k]/n:.4f}' for k in sums})

            # TensorBoard 途中経過（train のみ）
            if train and is_main_process() and tb_every > 0 and n % tb_every == 0:
                self.global_step += tb_every
                if self.writer is not None:
                    for k in sums:
                        self.writer.add_scalar(
                            f'Step/train_{k}', sums[k] / n, self.global_step
                        )
                    self.writer.flush()

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

            run_val = (epoch % self.args.val_every == 0) or (epoch == self.args.epochs)
            va = self._run_epoch(self.val_loader, False) if run_val else {}
            self.scheduler.step()

            if is_main_process():
                elapsed = time.time() - t0
                lr = self.optimizer.param_groups[0]['lr']
                val_str = f'{va["flow"]:.4f}' if run_val else '(skip)'
                print(
                    f'[DiT] Epoch {epoch:4d}/{self.args.epochs} | '
                    f'train={tr["flow"]:.4f} | val={val_str} | '
                    f'lr={lr:.2e} | {elapsed:.1f}s'
                )

                # CSV ログ
                with open(log, 'a') as f:
                    f.write(f'{epoch},{tr.get("flow",0):.6f},{va.get("flow",0):.6f},'
                            f'{lr:.2e},{elapsed:.1f}\n')

                # TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar('Loss/train_flow', tr.get('flow', 0), epoch)
                    if run_val:
                        self.writer.add_scalar('Loss/val_flow', va.get('flow', 0), epoch)
                    self.writer.add_scalar('Params/lr', lr, epoch)
                    self.writer.flush()   # エポックごとにディスクへ書き込み

                if (epoch % self.args.save_every == 0 or epoch == self.args.epochs) and run_val:
                    self._save(epoch, va['flow'])

        if self.writer is not None:
            self.writer.close()


# ── ベンチマーク ──────────────────────────────────────────────────────────────────

def _benchmark_vae(trainer: VAETrainer, n_batches: int):
    """VAE のセクション別タイミング計測。"""
    device = trainer.device
    use_cuda = device.type == 'cuda'

    trainer.cond_encoder.train()
    trainer.vae.train()

    beta = trainer._get_beta(1)
    accum = trainer.args.grad_accum

    # タイミング収集用
    timings = {
        'data_load': [], 'to_device': [], 'cond_enc': [],
        'vae_fwd': [], 'loss_calc': [], 'backward': [],
        'optim_step': [], 'total': [],
    }

    if use_cuda:
        torch.cuda.synchronize(device)

    warmup = 3
    loader_iter = iter(trainer.train_loader)
    trainer.optimizer.zero_grad(set_to_none=True)

    for i in range(warmup + n_batches):
        is_warmup = i < warmup

        t_start = time.perf_counter()

        # ── Data load ──
        batch = None
        while batch is None:
            batch = next(loader_iter)
        t_data = time.perf_counter()

        # ── To device ──
        batch = batch.to(device)
        if use_cuda:
            torch.cuda.synchronize(device)
        t_todev = time.perf_counter()

        # ── ConditionalEncoder ──
        with torch.autocast(device_type=device.type, dtype=trainer.amp_dtype,
                            enabled=trainer.amp_enabled):
            h_cond, e_cond, cond = trainer.cond_encoder(
                batch.atom_type_idx, batch.hyb_idx, batch.atom_cont,
                batch.bond_type_idx, batch.bond_cont, batch.edge_index,
                rwpe=getattr(batch, 'rwpe', None),
                lappe=getattr(batch, 'lappe', None),
                batch=batch.batch,
            )
        if use_cuda:
            torch.cuda.synchronize(device)
        t_cond = time.perf_counter()

        # ── VAE forward ──
        with torch.autocast(device_type=device.type, dtype=trainer.amp_dtype,
                            enabled=trainer.amp_enabled):
            pos_pred, mu, logvar = trainer.vae(
                cond, batch.pos, batch.edge_index, e_cond, batch.batch
            )
        if use_cuda:
            torch.cuda.synchronize(device)
        t_vae = time.perf_counter()

        # ── Loss ──
        with torch.autocast(device_type=device.type, dtype=trainer.amp_dtype,
                            enabled=trainer.amp_enabled):
            num_nodes = batch.pos.size(0)
            loss, ld = vae_loss(
                pos_pred, batch.pos, mu, logvar,
                batch.edge_index, num_nodes, beta=beta,
                w_pos=trainer.args.w_pos, w_bond=trainer.args.w_bond,
                w_angle=trainer.args.w_angle, w_dihedral=trainer.args.w_dihedral,
                triplets=getattr(batch, 'triplets', None),
                quartets=getattr(batch, 'quartets', None),
            )
        if use_cuda:
            torch.cuda.synchronize(device)
        t_loss = time.perf_counter()

        # ── Backward ──
        (loss / accum).backward()
        if use_cuda:
            torch.cuda.synchronize(device)
        t_bwd = time.perf_counter()

        # ── Optimizer step ──
        t_opt = t_bwd
        if (i + 1) % accum == 0:
            nn.utils.clip_grad_norm_(
                list(trainer.cond_encoder.parameters()) + list(trainer.vae.parameters()),
                trainer.args.grad_clip,
            )
            trainer.optimizer.step()
            trainer.optimizer.zero_grad(set_to_none=True)
            if use_cuda:
                torch.cuda.synchronize(device)
            t_opt = time.perf_counter()

        t_end = time.perf_counter()

        if not is_warmup:
            timings['data_load'].append(t_data - t_start)
            timings['to_device'].append(t_todev - t_data)
            timings['cond_enc'].append(t_cond - t_todev)
            timings['vae_fwd'].append(t_vae - t_cond)
            timings['loss_calc'].append(t_loss - t_vae)
            timings['backward'].append(t_bwd - t_loss)
            timings['optim_step'].append(t_opt - t_bwd)
            timings['total'].append(t_end - t_start)

    return timings


def _benchmark_dit(trainer: DiTTrainer, n_batches: int):
    """DiT のセクション別タイミング計測。"""
    device = trainer.device
    use_cuda = device.type == 'cuda'

    trainer.flow.train(True)
    accum = trainer.args.grad_accum

    timings = {
        'data_load': [], 'to_device': [],
    }
    if not trainer.use_latent:
        timings['cond_enc'] = []
        timings['vae_enc'] = []
    timings.update({'flow_fwd': [], 'backward': [], 'optim_step': [], 'total': []})

    if use_cuda:
        torch.cuda.synchronize(device)

    warmup = 3
    loader_iter = iter(trainer.train_loader)
    trainer.optimizer.zero_grad(set_to_none=True)

    for i in range(warmup + n_batches):
        is_warmup = i < warmup

        t_start = time.perf_counter()

        batch = None
        while batch is None:
            batch = next(loader_iter)
        t_data = time.perf_counter()

        batch = batch.to(device)
        if use_cuda:
            torch.cuda.synchronize(device)
        t_todev = time.perf_counter()

        dist_mat = getattr(batch, 'dist_mat', None)

        if trainer.use_latent:
            z0 = batch.z0
            cond = batch.cond
            t_enc = t_todev
        else:
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=trainer.amp_dtype,
                                    enabled=trainer.amp_enabled):
                    _, e_cond, cond = trainer.cond_encoder(
                        batch.atom_type_idx, batch.hyb_idx, batch.atom_cont,
                        batch.bond_type_idx, batch.bond_cont, batch.edge_index,
                        rwpe=getattr(batch, 'rwpe', None),
                        lappe=getattr(batch, 'lappe', None),
                        batch=batch.batch,
                    )
            if use_cuda:
                torch.cuda.synchronize(device)
            t_cond = time.perf_counter()

            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=trainer.amp_dtype,
                                    enabled=trainer.amp_enabled):
                    mu, _ = trainer.vae.encoder(
                        cond, batch.pos, batch.edge_index, e_cond, batch.batch
                    )
                    z0 = mu
            if use_cuda:
                torch.cuda.synchronize(device)
            t_enc = time.perf_counter()

        # ── Flow forward ──
        with torch.autocast(device_type=device.type, dtype=trainer.amp_dtype,
                            enabled=trainer.amp_enabled):
            loss, ld = trainer.flow.loss(
                z0, cond, batch.batch, batch.edge_index, dist_mat=dist_mat,
            )
        if use_cuda:
            torch.cuda.synchronize(device)
        t_fwd = time.perf_counter()

        # ── Backward ──
        (loss / accum).backward()
        if use_cuda:
            torch.cuda.synchronize(device)
        t_bwd = time.perf_counter()

        # ── Optimizer step ──
        t_opt = t_bwd
        if (i + 1) % accum == 0:
            nn.utils.clip_grad_norm_(trainer.flow.parameters(), trainer.args.grad_clip)
            trainer.optimizer.step()
            trainer.optimizer.zero_grad(set_to_none=True)
            if use_cuda:
                torch.cuda.synchronize(device)
            t_opt = time.perf_counter()

        t_end = time.perf_counter()

        if not is_warmup:
            timings['data_load'].append(t_data - t_start)
            timings['to_device'].append(t_todev - t_data)
            if not trainer.use_latent:
                timings['cond_enc'].append(t_cond - t_todev)
                timings['vae_enc'].append(t_enc - t_cond)
            timings['flow_fwd'].append(t_fwd - t_enc)
            timings['backward'].append(t_bwd - t_fwd)
            timings['optim_step'].append(t_opt - t_bwd)
            timings['total'].append(t_end - t_start)

    return timings


def _print_benchmark(timings: dict, n_batches: int, total_batches_per_epoch: int, stage: str):
    """ベンチマーク結果のフォーマット出力。"""
    print(f'\n{"="*60}')
    print(f'  Benchmark: {stage} ({n_batches} batches, warmup=3)')
    print(f'{"="*60}')

    total_time = sum(timings['total'])
    avg_per_batch = total_time / n_batches

    print(f'\n{"セクション":<16} {"合計(s)":>10} {"平均(ms)":>10} {"割合":>8}')
    print(f'{"-"*46}')
    for key, vals in timings.items():
        s = sum(vals)
        avg_ms = (s / len(vals)) * 1000
        pct = s / total_time * 100
        label = {
            'data_load': 'Data Load',
            'to_device': 'To Device',
            'cond_enc': 'CondEncoder',
            'vae_fwd': 'VAE Forward',
            'vae_enc': 'VAE Encoder',
            'loss_calc': 'Loss Calc',
            'flow_fwd': 'Flow Forward',
            'backward': 'Backward',
            'optim_step': 'Optim Step',
            'total': 'TOTAL',
        }.get(key, key)
        marker = ' <<<' if key != 'total' and pct > 25 else ''
        print(f'{label:<16} {s:>10.3f} {avg_ms:>10.1f} {pct:>7.1f}%{marker}')

    est_epoch = avg_per_batch * total_batches_per_epoch
    est_h = est_epoch / 3600
    print(f'\n  平均バッチ時間: {avg_per_batch*1000:.1f} ms')
    print(f'  1エポック推定 : {est_epoch:.0f}s ({est_h:.1f}h)')
    print(f'  (全 {total_batches_per_epoch} バッチ)')
    print(f'{"="*60}\n')


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
            trainer = VAETrainer(args, local_rank)
            if args.benchmark > 0:
                timings = _benchmark_vae(trainer, args.benchmark)
                _print_benchmark(timings, args.benchmark,
                                 len(trainer.train_loader), 'VAE')
            else:
                trainer.run()
        else:
            trainer = DiTTrainer(args, local_rank)
            if args.benchmark > 0:
                timings = _benchmark_dit(trainer, args.benchmark)
                _print_benchmark(timings, args.benchmark,
                                 len(trainer.train_loader), 'DiT')
            else:
                trainer.run()
    finally:
        cleanup_dist()
