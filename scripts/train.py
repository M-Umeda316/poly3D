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
import gc
import os
import random
import sys
import time

# CUDA アロケータの断片化緩和（torch import より前に設定）
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
from contextlib import ExitStack, nullcontext
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
from poly3d.model.builder import build_cond_encoder, build_vae, build_dit
from poly3d.model.flow_matching import FlowMatching
from poly3d.model.vae_loss import vae_loss


# ── vdW 半径テーブル（clash 損失用） ───────────────────────────────────────────
_RVDW_TABLE: Optional[torch.Tensor] = None


def get_rvdw_table(device: torch.device, max_z: int = 100) -> torch.Tensor:
    """原子番号 Z でインデックスする van der Waals 半径テーブル（(max_z,) float32）。

    eval_ensemble の妥当性ゲートと同じ RDKit GetRvdw を使うことで clash 判定を一致させる。
    プロセス内で 1 回だけ構築してキャッシュ（デバイスが変われば作り直す）。index 0 は未使用。
    """
    global _RVDW_TABLE
    if _RVDW_TABLE is None or _RVDW_TABLE.device != device:
        from rdkit import Chem
        pt = Chem.GetPeriodicTable()
        vals = [0.0] + [float(pt.GetRvdw(z)) for z in range(1, max_z)]
        _RVDW_TABLE = torch.tensor(vals, dtype=torch.float32, device=device)
    return _RVDW_TABLE


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
    各ランクの (sums_dict, count) を全ランクで集計し、グローバル合計を返す。
    分散モードでなければ入力をそのまま返す（シングル GPU の挙動を完全維持）。

    ここで d[k] は「各バッチの損失（バッチ内平均）× count(=1) の総和」、
    n は count の総和（＝処理バッチ数）に相当する。したがって all_reduce SUM 後に
    呼び出し側で `d[k] / n` とすると、

        Σ_ranks Σ_batches loss_k    sum_of_(loss * count)
        ─────────────────────────  =  ─────────────────────
        Σ_ranks (バッチ数)              sum_of_count

    となり、各ランクの処理バッチ数が異なっても正しい加重平均になる（各バッチ重み1）。
    val を DistributedSampler で分割するとランクごとにバッチ数が変わり得るが、
    この形（sum_of_(loss*count) と sum_of_count をそれぞれ all_reduce してから割る）
    なら単一 GPU 時と同じグローバル平均に一致する。

    注意: DistributedSampler は割り切れるようサンプルを重複パディングするため、
    分割 val ではパディング由来の軽微なバイアスが乗る（drop_last=False のまま許容）。
    厳密さが必要なら padding 分を除外する集計を別途実装すること。
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
    p.add_argument('--egt_every', type=int, default=0,
                   help='k>0 で Decoder の k 層ごとに EGNN 層を EGT（大域 attention）に置換。0=無効（後方互換）')
    p.add_argument('--enc_egt_every', type=int, default=0,
                   help='k>0 で Encoder の k 層ごとに EGNN 層を EGT に置換。0=無効。'
                        '潜在に大域構造を符号化させたい場合に使う')
    p.add_argument('--beta_start', type=float, default=0.0)
    p.add_argument('--beta_end', type=float, default=1.0)
    p.add_argument('--beta_warmup_epochs', type=int, default=50)
    p.add_argument('--w_pos', type=float, default=1.0)
    p.add_argument('--w_bond', type=float, default=1.0)
    p.add_argument('--w_angle', type=float, default=0.5)
    p.add_argument('--w_dihedral', type=float, default=0.1)
    p.add_argument('--pos_loss_type', type=str, default='kabsch',
                   choices=['kabsch', 'distmat', 'local_distmat', 'multiscale_distmat'],
                   help='座標損失の種類: kabsch（Kabsch RMSD）/ distmat（距離行列 MSE）/ '
                        'local_distmat（近接ペアのみ、大域折り畳みに頑健）/ '
                        'multiscale_distmat（局所＋long-range 距離、大域折れに滑らかな勾配）')
    p.add_argument('--w_local', type=float, default=1.0,
                   help='multiscale_distmat の局所距離損失重み')
    p.add_argument('--w_global', type=float, default=1.0,
                   help='multiscale_distmat の long-range 距離損失重み')
    p.add_argument('--local_cutoff', type=float, default=5.0,
                   help='local_distmat の近接ペア閾値（Å）')
    # clash（立体衝突）ガードレール損失。eval の妥当性ゲートの clash 判定を鏡写しにする。
    p.add_argument('--w_clash', type=float, default=0.0,
                   help='clash ガードレール損失の重み（0=無効・後方互換）。'
                        'グラフ距離>=3 のペアが (rvdw_i+rvdw_j)*clash_factor に食い込んだ量を罰する')
    p.add_argument('--clash_factor', type=float, default=0.6,
                   help='clash 閾値係数（eval の妥当性ゲートと合わせ 0.6）')
    p.add_argument('--clash_min_graph_dist', type=int, default=3,
                   help='clash 対象とみなすグラフ距離の下限（eval と合わせ 3）')
    p.add_argument('--clash_max_pairs', type=int, default=512,
                   help='clash 損失で 1 分子あたりサンプリングする最大ペア数')
    p.add_argument('--mds_init', action='store_true', default=False,
                   help='デコーダの初期座標に MDS 大域足場（トポロジー由来）を用いる。'
                        '0=無効（後方互換、per-atom MLP のみ）')

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
    p.add_argument('--lr_min', type=float, default=0.0,
                   help='CosineAnnealingLR の eta_min（末尾での LR 下限）。'
                        '0 だと末尾で LR=0 まで減衰し枯渇するため、短〜中 epoch では lr*0.05 程度を推奨')
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--max_atoms', type=int, default=240)
    p.add_argument('--num_workers', type=int, default=8,
                   help='DataLoader ワーカー数。Ryzen9 7900X: 8-10、Xeon 8558: 16-20 推奨')
    p.add_argument('--save_every', type=int, default=10)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--init_weights', type=str, default=None,
                   help='チェックポイントから cond_encoder + vae の重みだけを読み込んで'
                        '学習を新規開始する（optimizer/scheduler/epoch は fresh）。'
                        '既存モデルを土台に別の損失・LRで fine-tune する用途。'
                        '--resume が指定された場合はそちらが優先される（本フラグは無視）')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--seed', type=int, default=42)

    # パフォーマンス
    p.add_argument('--no_amp', action='store_true', default=False,
                   help='混合精度 (bf16) を無効化。デフォルト: AMP 有効')
    p.add_argument('--grad_accum', type=int, default=1,
                   help='勾配累積ステップ数。実効 batch_size = batch_size × grad_accum')
    p.add_argument('--oom_max_skips', type=int, default=0,
                   help='単一GPU時、OOM でバッチを破棄して継続することを許す回数'
                        '（0=初回 OOM で即停止＝fail-fast, 既定）。破棄されるのは'
                        '常に最も重いバッチ＝巨大分子(240+)を含むバッチ＝サイズ別'
                        '評価で測ろうとしている当の信号なので、握りつぶすと実験が'
                        '静かに歪む。save_every 1 なら再起動で直前 epoch から '
                        'resume できるので停止コストは小さい。DDP は従来どおり'
                        '常に即 raise（NCCL ハング回避）。stage=vae のみ有効。')
    p.add_argument('--warmup_steps', type=int, default=0,
                   help='LR warmup の長さ（**optimizer step** 単位。0=無効＝完全後方'
                        '互換）。lr を warmup_start_factor 倍から線形に定格 lr へ'
                        '上げる。epoch 単位でなく step 単位なのは、幅256で実測した'
                        '勾配爆発が最初の ~40 optimizer step で起きるため（epoch '
                        '単位では 1 epoch = 数千 step で粗すぎて効かない）。'
                        'stage=vae のみ有効。')
    p.add_argument('--warmup_start_factor', type=float, default=0.01,
                   help='warmup 開始時の lr 倍率（--warmup_steps > 0 のときのみ）。')
    p.add_argument('--gnorm_log_every', type=int, default=0,
                   help='N optimizer step ごとに [GNORM] 行（クリップ前の勾配ノルム・'
                        'クリップ率）を出力（0=無効, epoch 毎の集計は常に出力）。'
                        '計算には非干渉。実測: C(hidden128)/E(hidden256) とも収束後で'
                        'mean||g||=47/73・clipped=100% ＝常時クリップ。')
    p.add_argument('--empty_cache_every', type=int, default=0,
                   help='N step ごとに torch.cuda.empty_cache() で予約メモリの断片化を'
                        'リセット（0=無効）。Windows は expandable_segments 非対応で'
                        '可変分子サイズだと reserved が肥大するため、実使用が小さいのに'
                        '天井に張り付く場合の対策。計算には非干渉。')
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

    # ハイパーパラメータ探索用
    p.add_argument('--subset_ratio', type=float, default=1.0,
                   help='訓練データのサブセット比率 (0 < r ≤ 1.0)')
    p.add_argument('--val_subset_ratio', type=float, default=1.0,
                   help='学習中の val loss 監視用サブセット比率 (0 < r ≤ 1.0)。'
                        'best ckpt 選択・監視用の指標なので小さくてよい（本評価は eval_by_size.py で別途実施）。'
                        'デフォルト 1.0 は全 val を使用（従来動作）')

    # 大分子オーバーサンプリング（VAE stage 専用）
    p.add_argument('--oversample_alpha', type=float, default=0.0,
                   help='>0 で原子数^alpha に比例した重みで大分子を過抽出する'
                        '（WeightedRandomSampler, 復元抽出）。VAE stage 専用・単一GPUのみ。'
                        '事前に scripts/build_size_index.py で <train_lmdb>.sizes.npy を作ること。'
                        '0.0（デフォルト）で無効（従来動作を一切変えない）')

    return p.parse_args()


# ── サブセットユーティリティ ──────────────────────────────────────────────────────

def _make_subset_idx(n: int, ratio: float, seed: int) -> Optional[list]:
    """比率に応じたランダムサブセットインデックスを返す。ratio >= 1.0 なら None。"""
    if ratio >= 1.0:
        return None
    k = max(1, round(n * ratio))
    return random.Random(seed).sample(range(n), k)


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
        # torch.compile（DDP ラップ後でも PyTorch 2.x では動作する）
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
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=args.lr_min)

        # サブセットインデックスの計算（subset_ratio < 1.0 の場合のみ）
        subset_idx: Optional[list] = None
        if args.subset_ratio < 1.0:
            from poly3d.data.dataset import ConformerDataset as _CDS
            _n = len(_CDS(args.train_lmdb, max_atoms=args.max_atoms))
            subset_idx = _make_subset_idx(_n, args.subset_ratio, args.seed)
            if is_main_process():
                print(f'サブセット: {len(subset_idx):,} / {_n:,} サンプル ({args.subset_ratio:.0%})')

        # DistributedSampler（分散モード時）
        self.train_sampler: Optional[torch.utils.data.Sampler] = None
        if dist.is_initialized():
            from poly3d.data.dataset import ConformerDataset
            _ds = ConformerDataset(args.train_lmdb, max_atoms=args.max_atoms)
            if subset_idx is not None:
                from torch.utils.data import Subset
                _ds = Subset(_ds, subset_idx)
            self.train_sampler = DistributedSampler(_ds, shuffle=True, seed=args.seed)

        # 大分子オーバーサンプリング（VAE stage 専用・単一GPUのみ）。
        # alpha=0（デフォルト）では一切この経路に入らず従来動作（train_sampler=None /
        # DDP時 DistributedSampler）を完全に維持する。
        if args.oversample_alpha > 0:
            self._setup_oversampler(args, subset_idx)

        self.train_loader = make_dataloader(
            args.train_lmdb, batch_size=args.batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=args.num_workers, max_atoms=args.max_atoms,
            sampler=self.train_sampler,
            prefetch_factor=args.prefetch_factor,
            subset_indices=subset_idx,
            mds_init=args.mds_init,
        )
        # val も分散時は DistributedSampler で分割（全ランク重複処理の無駄を排除）。
        # shuffle=False。各ランクのバッチ数差は _all_reduce_dict の加重平均で吸収される。
        # val 監視用サブセット（best ckpt 選択・監視用。本評価は eval_by_size.py で別途）
        val_subset_idx: Optional[list] = None
        if args.val_subset_ratio < 1.0:
            from poly3d.data.dataset import ConformerDataset as _CDS
            _vn = len(_CDS(args.val_lmdb, max_atoms=args.max_atoms))
            val_subset_idx = _make_subset_idx(_vn, args.val_subset_ratio, args.seed)
            if is_main_process():
                print(f'val サブセット: {len(val_subset_idx):,} / {_vn:,} サンプル ({args.val_subset_ratio:.0%})')

        self.val_sampler: Optional[DistributedSampler] = None
        if dist.is_initialized() and dist.get_world_size() > 1:
            from poly3d.data.dataset import ConformerDataset
            _val_ds = ConformerDataset(args.val_lmdb, max_atoms=args.max_atoms)
            if val_subset_idx is not None:
                from torch.utils.data import Subset
                _val_ds = Subset(_val_ds, val_subset_idx)
            self.val_sampler = DistributedSampler(_val_ds, shuffle=False)
        self.val_loader = make_dataloader(
            args.val_lmdb, batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers, max_atoms=args.max_atoms,
            sampler=self.val_sampler,
            prefetch_factor=args.prefetch_factor,
            subset_indices=val_subset_idx,
            mds_init=args.mds_init,
        )

        self.start_epoch = 1
        self.best_val_loss = float('inf')
        self.global_step = 0
        # LR warmup 用。opt_step は optimizer.step() の通算回数（global_step は
        # tb_log_every 刻みで進む別物なので流用不可）。_sched_lrs は「その epoch で
        # scheduler が設定した lr」＝ warmup のスケール基準。
        self.opt_step = 0
        self._sched_lrs: Optional[list] = None
        # OOM で破棄したバッチ数（run 通算。epoch ごとにリセットしない＝1件でも
        # 出たらサイズ別評価は汚染されているため）。
        self._oom_skips = 0
        if args.resume:
            self._load(args.resume)
        elif args.init_weights:
            self._load_weights_only(args.init_weights)

        if is_main_process() and args.warmup_steps > 0:
            print(f'LR warmup: {args.warmup_steps} optimizer step かけて '
                  f'lr {args.lr * args.warmup_start_factor:.2e} → {args.lr:.2e} '
                  f'（開始 opt_step={self.opt_step}）')

        # TensorBoard（main process のみ）
        self.writer: Optional[SummaryWriter] = None
        if is_main_process():
            self.writer = SummaryWriter(log_dir=str(self.out_dir / 'tb_vae'))

    def _setup_oversampler(self, args: argparse.Namespace, subset_idx: Optional[list]) -> None:
        """
        原子数^alpha に比例した重みの WeightedRandomSampler を構築して
        self.train_sampler に設定する（大分子オーバーサンプリング）。

        重み設計:
          - 全 idx に対し w_i = size_i^alpha
          - size_i <= 0（欠損）または size_i > max_atoms は w_i = 0.0（抽出対象外）
          - subset を使う場合は subset_idx の順序に整列した重みを作る
            （WeightedRandomSampler は Subset のローカル添字 0..k-1 を返すため整合）

        DDP とは併用不可（コレクティブ通信・DistributedSampler と競合するため）。
        """
        import numpy as np
        from torch.utils.data import WeightedRandomSampler

        if dist.is_initialized():
            raise NotImplementedError('oversampling は単一GPUのみ対応')

        alpha = float(args.oversample_alpha)
        sizes_path = Path(str(args.train_lmdb) + '.sizes.npy')
        if not sizes_path.exists():
            raise FileNotFoundError(
                f'サイズインデックスが見つかりません: {sizes_path}\n'
                f'先に以下を実行してください:\n'
                f'  "{__import__("sys").executable}" scripts/build_size_index.py '
                f'--src {args.train_lmdb}'
            )
        sizes = np.load(sizes_path)

        # 全 idx の重み（max_atoms 超・欠損は 0.0）
        w_all = np.where(
            (sizes > 0) & (sizes <= args.max_atoms),
            sizes.astype(np.float64) ** alpha,
            0.0,
        )

        if subset_idx is not None:
            idx_arr = np.asarray(subset_idx, dtype=np.int64)
            weights = w_all[idx_arr]          # subset_idx 順に整列
            sub_sizes = sizes[idx_arr]
            num_samples = len(subset_idx)
        else:
            weights = w_all
            sub_sizes = sizes
            num_samples = len(sizes)

        w_sum = float(weights.sum())
        if w_sum <= 0:
            raise ValueError(
                'オーバーサンプリング重みの総和が 0 です。'
                f'プール内に 0 < 原子数 <= max_atoms({args.max_atoms}) の分子がありません。'
            )

        self.train_sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=num_samples,
            replacement=True,
        )   # generator は渡さない（グローバル RNG / torch.manual_seed 済み）

        # 要約（main プロセスのみ）
        if is_main_process():
            pool_mask = weights > 0
            n_pool = int(pool_mask.sum())
            large_mask = sub_sizes >= 130
            frac_large_pool = (
                float((large_mask & pool_mask).sum()) / n_pool if n_pool > 0 else 0.0
            )
            exp_large = float(weights[large_mask].sum()) / w_sum
            print(f'オーバーサンプリング: alpha={alpha:.2f}  '
                  f'プール件数={n_pool:,} / {len(weights):,}')
            print(f'  >=130 原子の割合: プール内 {frac_large_pool*100:.1f}% '
                  f'→ 抽出期待 {exp_large*100:.1f}%')

    def _get_beta(self, epoch: int) -> float:
        a = self.args
        if a.beta_warmup_epochs <= 0:
            return a.beta_end
        progress = min(1.0, (epoch - 1) / a.beta_warmup_epochs)
        return a.beta_start + (a.beta_end - a.beta_start) * progress

    def _apply_warmup(self) -> None:
        """LR warmup（opt-in, --warmup_steps > 0 のみ）。

        基準 (_sched_lrs) は「その epoch 頭に scheduler が設定した lr」。そこに
        線形の倍率を掛けるだけで、cosine 本体には一切触らない。opt_step ==
        warmup_steps でちょうど倍率 1.0（＝定格 lr）に着地し、以降は何もしない。

        warmup_steps=0 なら即 return ＝ param_group['lr'] を一度も書き換えない
        ので、既存ランと**ビット単位で同一**。
        """
        w = self.args.warmup_steps
        if w <= 0 or self._sched_lrs is None or self.opt_step > w:
            return
        f = self.args.warmup_start_factor
        scale = f + (1.0 - f) * (self.opt_step / w)
        for g, base in zip(self.optimizer.param_groups, self._sched_lrs):
            g['lr'] = base * scale

    def _load_weights_only(self, path: str):
        """cond_encoder + vae の重みだけを読み込み、optimizer/scheduler/epoch は fresh の
        まま学習を新規開始する（別損失・別 LR での warm-start / fine-tune 用）。"""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        _unwrap(self.cond_encoder).load_state_dict(ckpt['cond_encoder'])
        _unwrap(self.vae).load_state_dict(ckpt['vae'])
        if is_main_process():
            print(f'重みのみロード（fine-tune 起点）: {path} '
                  f'(元 epoch={ckpt.get("epoch", "?")}, val_loss={ckpt.get("val_loss", "?")}) '
                  f'→ optimizer/scheduler/epoch は fresh で開始')

    def _load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        _unwrap(self.cond_encoder).load_state_dict(ckpt['cond_encoder'])
        _unwrap(self.vae).load_state_dict(ckpt['vae'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_val_loss = ckpt.get('val_loss', float('inf'))
        self.global_step = ckpt.get('global_step', 0)
        # 旧 ckpt には無い → 0。warmup 無しで作られた ckpt なので 0 で正しい。
        self.opt_step = ckpt.get('opt_step', 0)
        if is_main_process():
            print(f'Resume: {path} (epoch {ckpt["epoch"]}, step {self.global_step})')

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
            'global_step': self.global_step,
            'opt_step': self.opt_step,
            'args': vars(self.args),
        }
        torch.save(ckpt, self.out_dir / f'vae_epoch{epoch:04d}.pt')
        if epoch > self.args.save_every:
            old = self.out_dir / f'vae_epoch{epoch-self.args.save_every:04d}.pt'
            if old.exists():   # resume 境界で存在しない場合があるためガード
                os.remove(old)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(ckpt, self.out_dir / 'vae_best.pt')
            print(f'  → best 更新 (val_loss: {val_loss:.4f})')

    def _run_epoch(self, loader, train: bool, beta: float) -> dict[str, float]:
        self.cond_encoder.train(train)
        self.vae.train(train)
        sums: dict = {}
        n = 0   # 処理済みバッチ数
        n_batches = len(loader)
        accum = self.args.grad_accum
        tb_every = self.args.tb_log_every
        self._gnorm_sum = 0.0
        self._gnorm_n = 0
        self._gnorm_clipped = 0

        if train:
            # warmup のスケール基準＝この epoch の scheduler 設定値を退避。
            self._sched_lrs = [g['lr'] for g in self.optimizer.param_groups]
            self.optimizer.zero_grad(set_to_none=True)

        # stderr がファイル等の非TTYへリダイレクトされている場合はバーを自動無効化
        # （ログにCR更新スパムを残さない。コンソール実行時のみ表示）。
        _tty = bool(getattr(sys.stderr, 'isatty', lambda: False)())
        pbar = tqdm(loader, desc='Train' if train else 'Val',
                    leave=False, dynamic_ncols=True,
                    disable=(not is_main_process()) or not _tty)

        for step, batch in enumerate(pbar):
            if batch is None:
                continue

            try:
                # .to(device) も try 内に置く（大きなバッチ転送時の OOM を捕捉して
                # スキップ・継続できるようにする。外に置くと未捕捉でプロセスが落ちる）
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
                            cond, batch.pos, batch.edge_index, e_cond, batch.batch,
                            dist_mat=getattr(batch, 'dist_mat', None),
                            init_scaffold=getattr(batch, 'init_scaffold', None),
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
                            pos_loss_type=self.args.pos_loss_type,
                            local_cutoff=self.args.local_cutoff,
                            batch=batch.batch,
                            # multiscale_distmat の long-range 項・clash 項に必要
                            dist_mat=getattr(batch, 'dist_mat', None),
                            ptr=getattr(batch, 'ptr', None),
                            w_local=self.args.w_local,
                            w_global=self.args.w_global,
                            # clash ガードレール損失
                            w_clash=self.args.w_clash,
                            rvdw=(get_rvdw_table(self.device)[batch.atomic_nums]
                                  if self.args.w_clash > 0.0 else None),
                            clash_factor=self.args.clash_factor,
                            clash_min_graph_dist=self.args.clash_min_graph_dist,
                            clash_max_pairs=self.args.clash_max_pairs,
                        )

                if train:
                    # 勾配累積: accum ステップで 1 回最適化
                    is_last_step = (step + 1 >= n_batches)
                    is_accum_step = ((step + 1) % accum == 0) or is_last_step
                    # accum 中間ステップ（optimizer.step() しない回）は DDP の勾配
                    # all-reduce を no_sync で抑制し、accum 回分の通信を最終ステップに集約する。
                    # 最終ステップで一括同期されるため数値は完全に不変（通信最適化のみ）。
                    # 単一 GPU では生モジュール（no_sync を持たない）→ 何もしない＝挙動不変。
                    with ExitStack() as stack:
                        if not is_accum_step:
                            for m in (self.cond_encoder, self.vae):
                                if hasattr(m, 'no_sync'):
                                    stack.enter_context(m.no_sync())
                        (loss / accum).backward()

                    if is_accum_step:
                        # clip_grad_norm_ はクリップ「前」の総ノルムを返す（追加計算なし）。
                        # 実測（2026-07-17）: C(hidden128) mean||g||=47.4 / E(hidden256)
                        # 73.4、いずれも clipped=100% ＝全ステップでクリップ発火し、
                        # optimizer には常に単位ノルム g/||g|| しか渡っていない。
                        # 注意: optimizer は AdamW。Adam の更新 m/(sqrt(v)+eps) は g の
                        # 「定数倍」に不変なので、常時クリップでも 1 step の大きさは
                        # ほぼ lr のまま＝SGD 的な「実効 LR が縮む」話にはならない。
                        # 効くのは別筋で、クリップ係数 1/||g_t|| が step ごとに変動する
                        # ため step 間の相対的な大きさが消える: 巨大分子を含む重い
                        # バッチ（||g||~97 を実測）が、易しいバッチ（~3）と同じ「1
                        # 単位」として m/v に入る＝希少で難しい例が本来持つはずの
                        # 大きな寄与を失う。1 step の総ノルム 1.0 を全分子で奪い合う
                        # ゼロサム構造にもなる（巨大分子を押すと小分子が痩せる）。
                        gnorm = nn.utils.clip_grad_norm_(
                            list(self.cond_encoder.parameters()) + list(self.vae.parameters()),
                            self.args.grad_clip,
                        )
                        self._gnorm_sum += float(gnorm)
                        self._gnorm_n += 1
                        self._gnorm_clipped += int(float(gnorm) > self.args.grad_clip)
                        self.opt_step += 1
                        self._apply_warmup()
                        # step 単位でも出す（epoch 単位だけだと幅256で1行27分かかり、
                        # 短時間プローブができない）。0=無効。lr も出すのは warmup が
                        # 実際に効いているかをここで検証するため。
                        if (is_main_process() and self.args.gnorm_log_every > 0
                                and self._gnorm_n % self.args.gnorm_log_every == 0):
                            print(f'[GNORM] optstep{self._gnorm_n} '
                                  f'norm={float(gnorm):.3f} '
                                  f'mean={self._gnorm_sum/self._gnorm_n:.3f} '
                                  f'clip={self.args.grad_clip} '
                                  f'clipped={100*self._gnorm_clipped/self._gnorm_n:.1f}% '
                                  f'lr={self.optimizer.param_groups[0]["lr"]:.2e}',
                                  flush=True)
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
            except (torch.cuda.OutOfMemoryError, torch.AcceleratorError) as e:
                if not isinstance(e, torch.cuda.OutOfMemoryError) and 'out of memory' not in str(e).lower():
                    raise
                n_atoms = int(batch.pos.size(0)) if 'batch' in locals() and batch is not None else -1
                self._oom_skips += 1
                if is_main_process():
                    print(f'[OOM] step={step} n_atoms={n_atoms} — バッチを破棄'
                          f'（この run で {self._oom_skips} 件目）')
                self.optimizer.zero_grad(set_to_none=True)
                # autograd graph を保持する中間テンソルを全て解放
                batch = None
                loss = None
                ld = None
                pos_pred = None
                mu = None
                logvar = None
                h_cond = None
                e_cond = None
                cond = None
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception as _ec:
                    if is_main_process():
                        print(f'[OOM] empty_cache も失敗: {_ec} — 次バッチで再試行')
                # マルチ GPU では、あるランクだけが OOM で continue すると
                # DDP backward の勾配 all-reduce（コレクティブ通信）が不整合になり、
                # 他ランクが完了待ちで NCCL ハングする（原因不明のまま停止）。
                # ハングするより明示的にクラッシュさせる方が安全なので、
                # メモリ解放後に元例外を re-raise する。シングル GPU では従来通り continue。
                if dist.is_initialized() and dist.get_world_size() > 1:
                    raise
                # 単一 GPU: fail-fast（既定 --oom_max_skips 0）。
                # 握りつぶして continue すると、捨てられるのは必ず「最も重い
                # バッチ」＝巨大分子(240+)を含むバッチ＝サイズ別評価で我々が測ろう
                # としている当の信号。静かに間引かれた結果は「240+ は改善しなかった」
                # と読めてしまい、実験が仮説に不利な方向へサイレントに歪む。
                # 実害の前例: run_D は wedged allocator のまま 2.5h・132,286 行の
                # OOM スキップを空回りし、1 epoch も完了せず終わった。
                # save_every 1 なら再起動で直前 epoch から resume できるので、
                # 落ちるコストは小さい（黙って壊れた結果を出すより遥かに安い）。
                if self._oom_skips > self.args.oom_max_skips:
                    raise RuntimeError(
                        f'OOM でバッチを破棄しました（step={step}, '
                        f'n_atoms={n_atoms}, この run で {self._oom_skips} 件目）。'
                        f'--oom_max_skips={self.args.oom_max_skips} を超えたので停止します。\n'
                        f'  破棄されるのは最も重い＝巨大分子を含むバッチなので、'
                        f'続行するとサイズ別評価が静かに歪みます。\n'
                        f'  対処: --batch_size を半分かつ --grad_accum を倍'
                        f'（実効 batch 維持＝比較性は保たれる）／'
                        f'--empty_cache_every を小さく／VRAM に余裕のある機で。\n'
                        f'  save_every 1 なら再起動で直前 epoch から resume されます。'
                    ) from e
                continue

            # detach テンソル → float 変換（バッチ末でまとめて sync）
            for k, v in ld.items():
                v_f = v.item() if isinstance(v, torch.Tensor) else v
                sums[k] = sums.get(k, 0.0) + v_f
            n += 1
            if is_main_process():
                pbar.set_postfix({k: f'{sums[k]/n:.4f}' for k in sums})
                if train and torch.cuda.is_available() and n % 200 == 0:
                    print(f'[VRAM] step{n} '
                          f'alloc={torch.cuda.max_memory_allocated()/1e9:.2f}GB '
                          f'reserved={torch.cuda.max_memory_reserved()/1e9:.2f}GB',
                          flush=True)
            if (train and torch.cuda.is_available()
                    and self.args.empty_cache_every > 0
                    and n % self.args.empty_cache_every == 0):
                torch.cuda.empty_cache()

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
            # DistributedSampler は set_epoch で shuffle 再シードが必要。
            # WeightedRandomSampler は set_epoch を持たない（グローバル RNG で毎回変動）。
            if self.train_sampler is not None and hasattr(self.train_sampler, 'set_epoch'):
                self.train_sampler.set_epoch(epoch)

            beta = self._get_beta(epoch)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            tr = self._run_epoch(self.train_loader, True, beta)
            if is_main_process() and torch.cuda.is_available():
                print(f'[VRAM] ep{epoch} train_peak '
                      f'alloc={torch.cuda.max_memory_allocated()/1e9:.2f}GB '
                      f'reserved={torch.cuda.max_memory_reserved()/1e9:.2f}GB',
                      flush=True)
            # clipped が常時 ~100% だと optimizer には単位ノルムの g/||g|| しか
            # 渡らず、step 間の勾配の大小が消える（詳細は clip 実行箇所のコメント）。
            if is_main_process() and self._gnorm_n > 0:
                print(f'[GNORM] ep{epoch} '
                      f'mean={self._gnorm_sum/self._gnorm_n:.3f} '
                      f'clip={self.args.grad_clip} '
                      f'clipped={100*self._gnorm_clipped/self._gnorm_n:.1f}%',
                      flush=True)

            run_val = (epoch % self.args.val_every == 0) or (epoch == self.args.epochs)
            va = self._run_epoch(self.val_loader, False, beta) if run_val else {}
            # warmup 中は param_group['lr'] を書き換えているので、scheduler が
            # 自分の設定値を読めるよう必ず戻してから step する。
            # CosineAnnealingLR.get_lr() は group['lr'] から**再帰的に**次の lr を
            # 計算するため、書き換えたまま step するとコサイン曲線自体が壊れる。
            # warmup 無効時は _sched_lrs == 現在値なので代入は no-op（＝挙動不変）。
            if self._sched_lrs is not None:
                for g, base in zip(self.optimizer.param_groups, self._sched_lrs):
                    g['lr'] = base
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

            # エポック末尾でメモリ掃除（fragmentation 対策）
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=args.lr_min)

        # 事前エンコード済み LMDB が指定された場合は LatentDataset を使用
        self.use_latent = (args.latent_lmdb is not None)
        if self.use_latent and is_main_process():
            print(f'LatentDataset モード: ConditionalEncoder/VAE Encoder スキップ')
            print(f'  train: {args.latent_lmdb}')
            print(f'  val  : {args.latent_val_lmdb or args.val_lmdb}')

        # サブセットインデックスの計算
        subset_idx: Optional[list] = None
        if args.subset_ratio < 1.0:
            if self.use_latent:
                from poly3d.data.latent_dataset import LatentDataset as _LDS
                _n = len(_LDS(args.latent_lmdb))
            else:
                from poly3d.data.dataset import ConformerDataset as _CDS
                _n = len(_CDS(args.train_lmdb, max_atoms=args.max_atoms))
            subset_idx = _make_subset_idx(_n, args.subset_ratio, args.seed)
            if is_main_process():
                print(f'サブセット: {len(subset_idx):,} / {_n:,} サンプル ({args.subset_ratio:.0%})')

        # DistributedSampler
        self.train_sampler: Optional[DistributedSampler] = None
        if dist.is_initialized():
            if self.use_latent:
                from poly3d.data.latent_dataset import LatentDataset
                _ds = LatentDataset(args.latent_lmdb)
            else:
                from poly3d.data.dataset import ConformerDataset
                _ds = ConformerDataset(args.train_lmdb, max_atoms=args.max_atoms)
            if subset_idx is not None:
                from torch.utils.data import Subset
                _ds = Subset(_ds, subset_idx)
            self.train_sampler = DistributedSampler(_ds, shuffle=True, seed=args.seed)

        # val も分散時は DistributedSampler で分割（全ランク重複処理の無駄を排除）。
        # shuffle=False。各ランクのバッチ数差は _all_reduce_dict の加重平均で吸収される。
        self.val_sampler: Optional[DistributedSampler] = None
        _distributed = dist.is_initialized() and dist.get_world_size() > 1
        if self.use_latent:
            self.train_loader = make_latent_dataloader(
                args.latent_lmdb, batch_size=args.batch_size,
                shuffle=(self.train_sampler is None),
                num_workers=args.num_workers,
                sampler=self.train_sampler,
                prefetch_factor=args.prefetch_factor,
                subset_indices=subset_idx,
            )
            val_latent = args.latent_val_lmdb or args.latent_lmdb
            if _distributed:
                from poly3d.data.latent_dataset import LatentDataset
                self.val_sampler = DistributedSampler(
                    LatentDataset(val_latent), shuffle=False)
            self.val_loader = make_latent_dataloader(
                val_latent, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers,
                sampler=self.val_sampler,
                prefetch_factor=args.prefetch_factor,
            )
        else:
            self.train_loader = make_dataloader(
                args.train_lmdb, batch_size=args.batch_size,
                shuffle=(self.train_sampler is None),
                num_workers=args.num_workers, max_atoms=args.max_atoms,
                sampler=self.train_sampler,
                prefetch_factor=args.prefetch_factor,
                subset_indices=subset_idx,
            )
            if _distributed:
                from poly3d.data.dataset import ConformerDataset
                self.val_sampler = DistributedSampler(
                    ConformerDataset(args.val_lmdb, max_atoms=args.max_atoms),
                    shuffle=False)
            self.val_loader = make_dataloader(
                args.val_lmdb, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, max_atoms=args.max_atoms,
                sampler=self.val_sampler,
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
        self.global_step = ckpt.get('global_step', 0)
        if is_main_process():
            print(f'Resume DiT: {path} (epoch {ckpt["epoch"]}, step {self.global_step})')

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
            'global_step': self.global_step,
            'args': vars(self.args),
        }
        torch.save(ckpt, self.out_dir / f'dit_epoch{epoch:04d}.pt')
        if epoch > self.args.save_every:
            os.remove(self.out_dir / f'dit_epoch{epoch-self.args.save_every:04d}.pt')
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(ckpt, self.out_dir / 'dit_best.pt')
            print(f'  → best 更新 (val_loss: {val_loss:.4f})')

    def _run_epoch(self, loader, train: bool) -> dict[str, float]:
        self.flow.train(train)
        sums: dict = {}
        n = 0
        n_batches = len(loader)
        accum = self.args.grad_accum
        tb_every = self.args.tb_log_every

        if train:
            self.optimizer.zero_grad(set_to_none=True)

        # stderr がファイル等の非TTYへリダイレクトされている場合はバーを自動無効化
        # （ログにCR更新スパムを残さない。コンソール実行時のみ表示）。
        _tty = bool(getattr(sys.stderr, 'isatty', lambda: False)())
        pbar = tqdm(loader, desc='Train' if train else 'Val',
                    leave=False, dynamic_ncols=True,
                    disable=(not is_main_process()) or not _tty)

        for step, batch in enumerate(pbar):
            if batch is None:
                continue

            try:
                # .to(device) も try 内に置く（OOM を捕捉してスキップ・継続するため）
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
                    is_last_step = (step + 1 >= n_batches)
                    is_accum_step = ((step + 1) % accum == 0) or is_last_step
                    # accum 中間ステップ（optimizer.step() しない回）は DDP の勾配
                    # all-reduce を no_sync で抑制し、accum 回分の通信を最終ステップに集約する。
                    # DiT stage は flow.model のみ DDP ラップ。最終ステップで一括同期され
                    # 数値は完全に不変（通信最適化のみ）。単一 GPU では生モジュール
                    # （no_sync を持たない）→ 何もしない＝挙動不変。
                    with ExitStack() as stack:
                        if not is_accum_step:
                            m = self.flow.model
                            if hasattr(m, 'no_sync'):
                                stack.enter_context(m.no_sync())
                        (loss / accum).backward()

                    if is_accum_step:
                        nn.utils.clip_grad_norm_(self.flow.parameters(), self.args.grad_clip)
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
            except (torch.cuda.OutOfMemoryError, torch.AcceleratorError) as e:
                if not isinstance(e, torch.cuda.OutOfMemoryError) and 'out of memory' not in str(e).lower():
                    raise
                if is_main_process():
                    n_atoms = (int(batch.pos.size(0))
                               if 'batch' in locals() and batch is not None and hasattr(batch, 'pos')
                               else -1)
                    print(f'[OOM] DiT step skip step={step} n_atoms={n_atoms} — バッチを破棄')
                self.optimizer.zero_grad(set_to_none=True)
                # autograd graph を保持する中間テンソルを全て解放
                batch = None
                loss = None
                ld = None
                z0 = None
                cond = None
                e_cond = None
                mu = None
                dist_mat = None
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception as _ec:
                    if is_main_process():
                        print(f'[OOM] empty_cache も失敗: {_ec} — 次バッチで再試行')
                # マルチ GPU では、あるランクだけが OOM で continue すると
                # DDP backward の勾配 all-reduce（コレクティブ通信）が不整合になり、
                # 他ランクが完了待ちで NCCL ハングする（原因不明のまま停止）。
                # ハングするより明示的にクラッシュさせる方が安全なので、
                # メモリ解放後に元例外を re-raise する。シングル GPU では従来通り continue。
                if dist.is_initialized() and dist.get_world_size() > 1:
                    raise
                continue

            for k, v in ld.items():
                v_f = v.item() if isinstance(v, torch.Tensor) else v
                sums[k] = sums.get(k, 0.0) + v_f
            n += 1
            if is_main_process():
                pbar.set_postfix({k: f'{sums[k]/n:.4f}' for k in sums})
                if train and torch.cuda.is_available() and n % 200 == 0:
                    print(f'[VRAM] step{n} '
                          f'alloc={torch.cuda.max_memory_allocated()/1e9:.2f}GB '
                          f'reserved={torch.cuda.max_memory_reserved()/1e9:.2f}GB',
                          flush=True)
            if (train and torch.cuda.is_available()
                    and self.args.empty_cache_every > 0
                    and n % self.args.empty_cache_every == 0):
                torch.cuda.empty_cache()

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

            # エポック末尾でメモリ掃除（fragmentation 対策）
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
                pos_loss_type=trainer.args.pos_loss_type,
                local_cutoff=trainer.args.local_cutoff,
                batch=batch.batch,
                dist_mat=getattr(batch, 'dist_mat', None),
                ptr=getattr(batch, 'ptr', None),
                w_local=trainer.args.w_local,
                w_global=trainer.args.w_global,
                w_clash=trainer.args.w_clash,
                rvdw=(get_rvdw_table(device)[batch.atomic_nums]
                      if trainer.args.w_clash > 0.0 else None),
                clash_factor=trainer.args.clash_factor,
                clash_min_graph_dist=trainer.args.clash_min_graph_dist,
                clash_max_pairs=trainer.args.clash_max_pairs,
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
