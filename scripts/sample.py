"""
SMILES からポリマー繰り返し単位の 3D コンフォーマーを生成するスクリプト

パイプライン:
  SMILES → ConditionalEncoder → Ci
  Z1 ~ N(0,I) → DiT (flow matching ODE) → Z0
  Z0 + Ci → VAE Decoder → 3D 座標

実行例:
  "C:/Users/shanu/anaconda3/envs/polygen/python.exe" sample.py \\
      --vae_checkpoint ./runs/polygen_v1/vae_best.pt \\
      --dit_checkpoint ./runs/polygen_v1/dit_best.pt \\
      --smiles "CC(C)c1ccc(CC)cc1" \\
      --out conformers.sdf \\
      --n_steps 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

from poly3d.model.builder import build_cond_encoder, build_vae, build_dit
from poly3d.model.flow_matching import FlowMatching
from poly3d.model.features import smiles_to_data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--vae_checkpoint', type=str, required=True)
    p.add_argument('--dit_checkpoint', type=str, required=True)
    p.add_argument('--smiles', type=str, default=None)
    p.add_argument('--smiles_file', type=str, default=None)
    p.add_argument('--out', type=str, default='conformers.sdf')
    p.add_argument('--n_conf', type=int, default=1)
    p.add_argument('--add_h', action='store_true', default=True)
    p.add_argument('--n_steps', type=int, default=100)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def load_models(vae_path: str, dit_path: str, device: torch.device):
    # VAE ロード
    vae_ckpt = torch.load(vae_path, map_location=device, weights_only=False)
    vae_args = argparse.Namespace(**vae_ckpt['args'])

    cond_encoder = build_cond_encoder(vae_args).to(device)
    cond_encoder.load_state_dict(vae_ckpt['cond_encoder'])
    cond_encoder.eval()

    vae = build_vae(vae_args).to(device)
    vae.load_state_dict(vae_ckpt['vae'])
    vae.eval()

    # DiT ロード
    dit_ckpt = torch.load(dit_path, map_location=device, weights_only=False)
    dit_args = argparse.Namespace(**dit_ckpt['args'])

    dit = build_dit(dit_args).to(device)
    # チェックポイントには LatentDiT の weights のみ保存されている
    dit.load_state_dict(dit_ckpt['flow'])
    flow = FlowMatching(dit, t_max=dit_args.t_max)
    flow.eval()

    return cond_encoder, vae, flow


def smiles_to_pyg(smiles: str, add_h: bool, use_rwpe: bool, use_lappe: bool,
                  device: torch.device) -> Data:
    d = smiles_to_data(smiles, add_h=add_h, use_rwpe=use_rwpe, use_lappe=use_lappe)
    n = d['atom_type_idx'].shape[0]
    kwargs = dict(
        atom_type_idx=torch.from_numpy(d['atom_type_idx'].astype(np.int64)).to(device),
        hyb_idx=torch.from_numpy(d['hyb_idx'].astype(np.int64)).to(device),
        atom_cont=torch.from_numpy(d['atom_cont']).to(device),
        bond_type_idx=torch.from_numpy(d['bond_type_idx'].astype(np.int64)).to(device),
        bond_cont=torch.from_numpy(d['bond_cont']).to(device),
        edge_index=torch.from_numpy(d['edge_index']).to(device),
        batch=torch.zeros(n, dtype=torch.long, device=device),
        num_nodes=n,
    )
    if 'rwpe' in d:
        kwargs['rwpe'] = torch.from_numpy(d['rwpe']).to(device)
    if 'lappe' in d:
        kwargs['lappe'] = torch.from_numpy(d['lappe']).to(device)
    return Data(**kwargs)


@torch.no_grad()
def generate_conformers(
    smiles: str,
    cond_encoder: ConditionalEncoder,
    vae: StructuralVAE,
    flow: FlowMatching,
    device: torch.device,
    n_conf: int = 1,
    add_h: bool = True,
    n_steps: int = 100,
) -> list:
    use_rwpe = cond_encoder.use_rwpe
    use_lappe = cond_encoder.use_lappe

    data = smiles_to_pyg(smiles, add_h, use_rwpe, use_lappe, device)
    n_atoms = data.atom_type_idx.size(0)
    batch_idx = data.batch

    # Conditioning
    _, e_cond, cond = cond_encoder(
        data.atom_type_idx, data.hyb_idx, data.atom_cont,
        data.bond_type_idx, data.bond_cont, data.edge_index,
        rwpe=getattr(data, 'rwpe', None),
        lappe=getattr(data, 'lappe', None),
        batch=batch_idx,
    )

    rdmol = Chem.MolFromSmiles(smiles)
    if rdmol is None:
        return []
    if add_h:
        rdmol = Chem.AddHs(rdmol)

    conformers = []
    for conf_idx in range(n_conf):
        # DiT ODE サンプリング → Z0
        z0 = flow.sample(
            n_atoms=n_atoms,
            cond=cond,
            batch=batch_idx,
            n_steps=n_steps,
            edge_index=data.edge_index,
            device=device,
        )

        # VAE Decoder → 座標
        pos = vae.decode(z0, cond, data.edge_index, e_cond, batch_idx)  # (N, 3)
        pos_np = pos.cpu().numpy().astype(float)

        mol = Chem.RWMol(rdmol)
        conf = Chem.Conformer(n_atoms)
        for i, (x, y, z) in enumerate(pos_np):
            conf.SetAtomPosition(i, (x, y, z))
        mol.AddConformer(conf, assignId=True)
        mol.SetProp('SMILES', smiles)
        mol.SetProp('conf_id', str(conf_idx))
        conformers.append(mol.GetMol())

    return conformers


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(
        'cuda' if args.device == 'auto' and torch.cuda.is_available()
        else args.device if args.device != 'auto' else 'cpu'
    )
    print(f'Device: {device}')

    cond_encoder, vae, flow = load_models(args.vae_checkpoint, args.dit_checkpoint, device)
    print('モデルをロード完了')

    if args.smiles:
        smiles_list = [args.smiles]
    elif args.smiles_file:
        with open(args.smiles_file) as f:
            smiles_list = [line.strip() for line in f if line.strip()]
    else:
        print('--smiles または --smiles_file を指定してください', file=sys.stderr)
        sys.exit(1)

    writer = Chem.SDWriter(args.out)
    total_ok = 0

    for smi in smiles_list:
        print(f'生成中: {smi}')
        try:
            mols = generate_conformers(
                smi, cond_encoder, vae, flow, device,
                n_conf=args.n_conf, add_h=args.add_h, n_steps=args.n_steps,
            )
            for mol in mols:
                writer.write(mol)
                total_ok += 1
        except Exception as e:
            print(f'  エラー ({smi}): {e}')

    writer.close()
    print(f'\n完了: {total_ok} 件 → {args.out}')


if __name__ == '__main__':
    main()
