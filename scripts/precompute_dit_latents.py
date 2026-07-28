"""
VAE デコーダ fine-tune（DiT consistency, 方針1b）用に、各訓練レコードの
実 DiT 潜在 z_dit を事前計算して LMDB に保存するスクリプト。

なぜ事前計算するか
------------------
v3e（方針1b）は fine-tune 中に毎バッチ `flow.sample(...)`（ODE ~20-100 step）で
実 DiT 潜在を生成し、`vae.forward(extra_latent=z_dit)` でデコード→clash+bond_range
ガードレールを課していた。これが大単位妥当性を上げたが、毎バッチ ODE で ~64 分/epoch と遅い。

エンコーダ（cond_encoder + vae.encoder）は凍結なので cond は決定的であり、DiT 潜在は
デコーダに依存しない。したがって各レコードの z_dit を「そのレコード自身の原子順で」
1 回だけ事前生成して保存すれば、学習は読むだけで済み（~30 分/epoch）、プールは再利用できる。
レコードごとに自分の順序で作るのでリマップ不要・構造的に正しい（PolyOmics の配座間
ラベリング不整合の影響を受けない）。

キー規約・保存形状
------------------
  キー   : f'{idx:09d}'（ConformerDataset / dataset.py __getitem__ と同一規約）
  値     : pickle 化した numpy 配列 (K, N_i, latent_dim) float32
           K = --n_samples（既定 1 なら (1, N, 16)）
  __len__: 元 src_lmdb の件数（dataset 規約に合わせる）
  max_atoms 超過で None になるレコードはキーを作らない（学習側は存在チェックで
  フォールバック）。

冪等性
------
out_lmdb に既にあるキーはスキップして再開できる（resume 可能）。

実行例（本番: train, ~2.8h 想定）:
  "C:/Users/shanu/anaconda3/envs/polygen/python.exe" scripts/precompute_dit_latents.py \
      --vae_checkpoint runs/polyomics_vae_v3c/vae_best.pt \
      --dit_checkpoint runs/polyomics_dit_v2/dit_best.pt \
      --src_lmdb data/polyomics_PG_train.lmdb \
      --out_lmdb data/polyomics_PG_train_ditlatents.lmdb \
      --n_steps 100 --n_samples 1 --batch_size 64
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Optional

import lmdb
import numpy as np
import torch

from poly3d.data.dataset import ConformerDataset, collate_fn, worker_init_fn
from poly3d.model.builder import build_cond_encoder, build_vae, build_dit
from poly3d.model.flow_matching import FlowMatching


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='各訓練レコードの実 DiT 潜在を事前計算して LMDB に保存する')
    p.add_argument('--vae_checkpoint', type=str,
                   default='runs/polyomics_vae_v3c/vae_best.pt',
                   help='cond_encoder を供給する VAE チェックポイント（凍結）')
    p.add_argument('--dit_checkpoint', type=str,
                   default='runs/polyomics_dit_v2/dit_best.pt',
                   help='潜在生成に使う DiT チェックポイント（凍結）')
    p.add_argument('--src_lmdb', type=str, required=True,
                   help='前処理済み LMDB（ConformerDataset が読む形式・通常 train）')
    p.add_argument('--out_lmdb', type=str, required=True,
                   help='出力先 DiT 潜在 LMDB（例 data/polyomics_PG_train_ditlatents.lmdb）')
    p.add_argument('--n_steps', type=int, default=100,
                   help='flow.sample の Euler ODE ステップ数')
    p.add_argument('--n_samples', type=int, default=1,
                   help='1 レコードあたり生成する潜在サンプル数 K（既定 1）')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--max_atoms', type=int, default=288)
    p.add_argument('--map_size_gb', type=float, default=6.0,
                   help='出力 LMDB のマップサイズ（GB）。K=1 実データ ~1-2GB 想定なので 6 で十分。'
                        'Windows は非 sparse で実確保するため過大にしないこと')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--max_records', type=int, default=0,
                   help='>0 で先頭からこの件数までしか処理しない（小規模検証用）。0=全件')
    p.add_argument('--log_every', type=int, default=50,
                   help='N バッチごとに進捗（件数・経過秒）を print')
    return p.parse_args()


class _IndexedDataset(torch.utils.data.Dataset):
    """ConformerDataset を包み (idx, Data or None) を返すラッパ。

    レコード index の対応が崩れないよう、__getitem__ は元 idx を必ず一緒に返す。
    collate（identity）で main プロセスに (idx, Data) のリストが渡るので、None を
    除外したうえで元 idx と生成潜在を確実に対応づけられる（max_atoms 超過で None に
    なったレコードはキーを作らない＝スキップ）。

    worker_init_fn は `.dataset` 属性を辿って ConformerDataset の lmdb env を
    worker ごとにリセットする（num_workers>0 対応）。そのため base を `.dataset`
    という名前で保持する。
    """

    def __init__(self, base: ConformerDataset, indices: list):
        self.dataset = base       # worker_init_fn が辿る属性名
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, j: int):
        idx = self.indices[j]
        return idx, self.dataset[idx]


def _identity_collate(batch: list) -> list:
    """(idx, Data or None) のリストをそのまま返す（main プロセスで手動 collate する）。"""
    return batch


def _existing_keys(out_path: Path, map_size: int) -> set:
    """out_lmdb に既に書かれている idx キー（整数）の集合を返す（resume 用）。"""
    if not out_path.exists():
        return set()
    env = lmdb.open(str(out_path), subdir=False, map_size=map_size,
                    readonly=False, lock=False)
    keys = set()
    with env.begin() as txn:
        cur = txn.cursor()
        for k, _ in cur:
            if k == b'__len__':
                continue
            try:
                keys.add(int(k.decode('ascii')))
            except ValueError:
                pass
    env.close()
    return keys


@torch.no_grad()
def main():
    args = parse_args()

    device = torch.device(
        'cuda' if args.device == 'auto' and torch.cuda.is_available()
        else (args.device if args.device != 'auto' else 'cpu')
    )
    print(f'Device: {device}')

    # ── VAE チェックポイントから cond_encoder をロード（凍結）──
    vck = torch.load(args.vae_checkpoint, map_location=device, weights_only=False)
    vae_args = argparse.Namespace(**vck['args'])
    cond_encoder = build_cond_encoder(vae_args).to(device)
    cond_encoder.load_state_dict(vck['cond_encoder'])
    cond_encoder.eval().requires_grad_(False)
    # latent_dim の確認用に vae も構築（重みは cond のみ使うが latent_dim を参照）
    vae = build_vae(vae_args).to(device)
    vae.load_state_dict(vck['vae'])
    vae.eval().requires_grad_(False)
    latent_dim = vae.latent_dim

    # ── DiT チェックポイントから FlowMatching をロード（凍結）──
    dck = torch.load(args.dit_checkpoint, map_location=device, weights_only=False)
    dargs = argparse.Namespace(**dck['args'])
    dit = build_dit(dargs).to(device)
    dit.load_state_dict(dck['flow'])
    flow = FlowMatching(dit, t_max=dargs.t_max)
    flow.eval()
    flow.requires_grad_(False)
    print(f'モデルロード完了  vae={args.vae_checkpoint}  dit={args.dit_checkpoint}  '
          f'latent_dim={latent_dim}  n_steps={args.n_steps}  K={args.n_samples}')

    # ── Dataset（cond 計算は precompute とアーキ整合のため margs に追従）──
    base_ds = ConformerDataset(
        args.src_lmdb, max_atoms=args.max_atoms, precompute_topology=True,
        mds_init=getattr(vae_args, 'mds_init', False),
    )
    total = len(base_ds)
    if args.max_records > 0:
        total = min(total, args.max_records)
    print(f'src レコード数: {len(base_ds)}  処理対象: {total}')

    # ── 出力 LMDB を開く（resume: 既存キーをスキップ）──
    out_path = Path(args.out_lmdb)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    map_size = int(args.map_size_gb * 1024 ** 3)
    done_keys = _existing_keys(out_path, map_size)
    if done_keys:
        print(f'resume: 既存キー {len(done_keys)} 件をスキップ')

    todo = [i for i in range(total) if i not in done_keys]
    if not todo:
        print('全レコード処理済み。何もすることはありません。')
        return

    env_out = lmdb.open(str(out_path), subdir=False, map_size=map_size,
                        readonly=False)

    idx_ds = _IndexedDataset(base_ds, todo)
    loader = torch.utils.data.DataLoader(
        idx_ds,
        batch_size=args.batch_size,
        shuffle=False,                 # 元 idx を保持（連番対応）
        num_workers=args.num_workers,
        collate_fn=_identity_collate,
        worker_init_fn=worker_init_fn,
        pin_memory=False,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                             enabled=(device.type == 'cuda'))

    n_saved = len(done_keys)
    n_skipped_none = 0
    t0 = time.time()

    for bi, raw in enumerate(loader):
        # raw: [(idx, Data or None), ...]  None（max_atoms 超過）を除外
        pairs = [(idx, d) for (idx, d) in raw if d is not None]
        n_skipped_none += len(raw) - len(pairs)
        if not pairs:
            continue

        kept_idx = [idx for (idx, _) in pairs]
        datas = [d for (_, d) in pairs]
        # collate_fn は topology 属性を破壊消費するので使い捨て（毎バッチ fresh 取得済み）
        batch = collate_fn(datas)
        if batch is None:
            continue
        batch = batch.to(device)

        n_total = batch.num_nodes
        dm = getattr(batch, 'dist_mat', None)

        with amp_ctx:
            _, _e_cond, cond = cond_encoder(
                batch.atom_type_idx, batch.hyb_idx, batch.atom_cont,
                batch.bond_type_idx, batch.bond_cont, batch.edge_index,
                rwpe=getattr(batch, 'rwpe', None),
                lappe=getattr(batch, 'lappe', None),
                batch=batch.batch,
            )

        # ── K 回サンプリング（各回 (n_total, latent_dim)）→ (K, n_total, latent_dim) ──
        # flow.sample は autocast 外（内部で必要な演算を fp32 で行う。eval_ensemble 準拠）。
        samples = []
        for _ in range(args.n_samples):
            z = flow.sample(
                n_atoms=n_total, cond=cond, batch=batch.batch,
                n_steps=args.n_steps, edge_index=batch.edge_index,
                device=device, dist_mat=dm,
            )
            samples.append(z.float().cpu())
        z_all = torch.stack(samples, dim=0)   # (K, n_total, latent_dim)

        # ── ptr でレコード単位に分割して保存（元 idx をキーに使う）──
        ptr = batch.ptr
        with env_out.begin(write=True) as txn:
            for m, idx in enumerate(kept_idx):
                a = int(ptr[m])
                b = int(ptr[m + 1])
                z_rec = z_all[:, a:b, :].contiguous().numpy().astype(np.float32)
                # (K, N_i, latent_dim)
                key = f'{idx:09d}'.encode('ascii')
                txn.put(key, pickle.dumps(z_rec))
                n_saved += 1

        if (bi + 1) % args.log_every == 0:
            dt = time.time() - t0
            print(f'  batch {bi + 1}: 保存 {n_saved} 件  '
                  f'（None スキップ {n_skipped_none}）  経過 {dt:.1f}s',
                  flush=True)

    # __len__ を保存（dataset 規約：元 src の件数）
    with env_out.begin(write=True) as txn:
        txn.put(b'__len__', str(len(base_ds)).encode('ascii'))

    env_out.close()
    base_ds.close()
    dt = time.time() - t0
    print(f'\n完了: 保存 {n_saved} 件（None スキップ {n_skipped_none}）  '
          f'経過 {dt:.1f}s → {args.out_lmdb}')


if __name__ == '__main__':
    main()
