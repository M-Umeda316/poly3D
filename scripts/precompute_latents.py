"""
DiT 学習用の潜在変数を事前エンコードして LMDB に保存するスクリプト。

ConditionalEncoder と VAE Encoder の出力（cond, e_cond, z0=mu）を
全サンプルについて一度だけ計算してキャッシュする。

DiT 学習時はこの事前エンコード済み LMDB を使うことで、
毎バッチの凍結モデル実行コスト（4 MPNN 層 + 4 EGNN 層）をゼロにできる。

実行例:
  "C:/Users/shanu/anaconda3/envs/polygen/python.exe" scripts/precompute_latents.py \
      --vae_checkpoint ./runs/polygen_v1/vae_best.pt \
      --src_lmdb D:/Dataset/OMol_base/OPoly26/processed/train.lmdb \
      --out_lmdb D:/Dataset/OMol_base/OPoly26/latents/train.lmdb \
      --batch_size 256 --num_workers 8

  # val も同様に
  "C:/Users/shanu/anaconda3/envs/polygen/python.exe" scripts/precompute_latents.py \
      --vae_checkpoint ./runs/polygen_v1/vae_best.pt \
      --src_lmdb D:/Dataset/OMol_base/OPoly26/processed/val.lmdb \
      --out_lmdb D:/Dataset/OMol_base/OPoly26/latents/val.lmdb \
      --batch_size 256 --num_workers 8
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import lmdb
import torch
import torch.nn as nn
from tqdm import tqdm

from poly3d.data.dataset import make_dataloader
from poly3d.model.cond_encoder import ConditionalEncoder
from poly3d.model.vae import StructuralVAE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--vae_checkpoint', type=str, required=True)
    p.add_argument('--src_lmdb', type=str, required=True,
                   help='前処理済み LMDB (ConformerDataset が読む形式)')
    p.add_argument('--out_lmdb', type=str, required=True,
                   help='出力先 LMDB (LatentDataset が読む形式)')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--max_atoms', type=int, default=300)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--map_size_gb', type=float, default=40.0,
                   help='出力 LMDB のマップサイズ (GB)')
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        'cuda' if args.device == 'auto' and torch.cuda.is_available()
        else (args.device if args.device != 'auto' else 'cpu')
    )
    print(f'Device: {device}')

    # VAE チェックポイントから ConditionalEncoder と VAE をロード
    ckpt = torch.load(args.vae_checkpoint, map_location=device, weights_only=False)
    vae_args = argparse.Namespace(**ckpt['args'])

    cond_encoder = ConditionalEncoder(
        hidden_dim=vae_args.hidden_dim,
        edge_dim=vae_args.edge_dim,
        n_layers=vae_args.cond_layers,
        atom_emb_dim=vae_args.atom_emb_dim,
        hyb_emb_dim=vae_args.hyb_emb_dim,
        bond_emb_dim=vae_args.bond_emb_dim,
        use_rwpe=vae_args.use_rwpe,
        use_lappe=vae_args.use_lappe,
    ).to(device)
    cond_encoder.load_state_dict(ckpt['cond_encoder'])
    cond_encoder.eval().requires_grad_(False)

    vae = StructuralVAE(
        cond_dim=vae_args.hidden_dim,
        edge_dim=vae_args.edge_dim,
        hidden_dim=vae_args.vae_hidden_dim,
        latent_dim=vae_args.latent_dim,
        enc_layers=vae_args.enc_layers,
        dec_layers=vae_args.dec_layers,
    ).to(device)
    vae.load_state_dict(ckpt['vae'])
    vae.eval().requires_grad_(False)

    print('モデルロード完了')

    # DataLoader（トポロジー事前計算込み）
    loader = make_dataloader(
        args.src_lmdb,
        batch_size=args.batch_size,
        shuffle=False,    # 順番を保持して idx を LMDB キーに使う
        num_workers=args.num_workers,
        max_atoms=args.max_atoms,
        precompute_topology=True,
    )

    # 出力先 LMDB を開く
    out_path = Path(args.out_lmdb)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    map_size = int(args.map_size_gb * 1024 ** 3)
    env_out = lmdb.open(str(out_path), subdir=False, map_size=map_size, readonly=False)

    total_saved = 0
    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                              enabled=(device.type == 'cuda'))

    with torch.no_grad():
        for batch in tqdm(loader, desc='Encoding'):
            if batch is None:
                continue
            batch = batch.to(device)

            with amp_ctx:
                _, e_cond, cond = cond_encoder(
                    batch.atom_type_idx, batch.hyb_idx, batch.atom_cont,
                    batch.bond_type_idx, batch.bond_cont, batch.edge_index,
                    rwpe=getattr(batch, 'rwpe', None),
                    lappe=getattr(batch, 'lappe', None),
                    batch=batch.batch,
                )
                mu, _ = vae.encoder(cond, batch.pos, batch.edge_index, e_cond, batch.batch)

            # float32 に戻してから保存（後でそのまま使えるように）
            cond_fp32   = cond.float().cpu()
            e_cond_fp32 = e_cond.float().cpu()
            z0_fp32     = mu.float().cpu()
            batch_idx   = batch.batch.cpu()
            edge_index  = batch.edge_index.cpu()
            dist_mat    = getattr(batch, 'dist_mat', None)
            if dist_mat is not None:
                dist_mat = dist_mat.cpu()

            # 分子ごとに分解して保存
            B = int(batch_idx.max().item()) + 1
            with env_out.begin(write=True) as txn:
                for b in range(B):
                    node_mask = (batch_idx == b)
                    n = node_mask.sum().item()
                    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
                    offset = node_mask.nonzero(as_tuple=True)[0][0].item()

                    entry = {
                        'z0':        z0_fp32[node_mask].numpy(),      # (N, latent_dim)
                        'cond':      cond_fp32[node_mask].numpy(),     # (N, cond_dim)
                        'e_cond':    e_cond_fp32[edge_mask].numpy(),   # (E, edge_dim)
                        'edge_index': (edge_index[:, edge_mask] - offset).numpy(),  # (2, E)
                        'num_nodes': n,
                    }
                    if dist_mat is not None:
                        # 分子内のブロックを切り出す
                        idx = node_mask.nonzero(as_tuple=True)[0]
                        entry['dist_mat'] = dist_mat[idx][:, idx].numpy()   # (N, N) int8

                    key = f'{total_saved:09d}'.encode('ascii')
                    txn.put(key, pickle.dumps(entry))
                    total_saved += 1

    # __len__ を保存
    with env_out.begin(write=True) as txn:
        txn.put(b'__len__', str(total_saved).encode('ascii'))

    env_out.close()
    print(f'\n完了: {total_saved} サンプル → {args.out_lmdb}')


if __name__ == '__main__':
    main()
