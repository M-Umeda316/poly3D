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
from poly3d.model.cond_encoder import ConditionalEncoder
from poly3d.model.flow_matching import FlowMatching
from poly3d.model.features import smiles_to_data
from poly3d.model.pos_bias import compute_graph_distance, mds_init_coords
from poly3d.model.relax import relax_coords
from poly3d.model.vae import StructuralVAE


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
    # ── 幾何緩和（後処理）: decode 座標に clash+bond のヒンジエネルギーを掛けて局所違反を解消 ──
    p.add_argument('--relax', action='store_true',
                   help='decode 後の座標に clash+bond の幾何緩和（後処理）を掛ける')
    p.add_argument('--relax_steps', type=int, default=20, help='緩和の最適化ステップ数')
    p.add_argument('--relax_lr', type=float, default=0.02, help='緩和 Adam の学習率')
    p.add_argument('--relax_w_clash', type=float, default=1.0, help='clash ヒンジ項の重み')
    p.add_argument('--relax_w_bond', type=float, default=1.0, help='bond ヒンジ項の重み')
    p.add_argument('--relax_w_anchor', type=float, default=0.5, help='元座標アンカー項の重み')
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

    # vae_args（学習時のモデル引数）も返す。EGT/MDS の有無を推論側でゲートするため
    # generate_conformers が参照する（getattr デフォルトで旧 ckpt にも後方互換）。
    return cond_encoder, vae, flow, vae_args


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
        atomic_nums=torch.from_numpy(d['atomic_nums'].astype(np.int64)).to(device),
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
    margs: argparse.Namespace,
    n_conf: int = 1,
    add_h: bool = True,
    n_steps: int = 100,
    relax: bool = False,
    relax_steps: int = 20,
    relax_lr: float = 0.02,
    relax_w_clash: float = 1.0,
    relax_w_bond: float = 1.0,
    relax_w_anchor: float = 0.5,
) -> list:
    """
    同一分子の n_conf 本のコンフォーマーを、ノードをブロック対角に複製した
    1 つのバッチにまとめ、flow.sample() を 1 回だけ呼び出して生成する。

    高速化ポイント:
      - グラフ距離 BFS（compute_graph_distance）は分子 1 本分のみ計算し、
        torch.block_diag で n_conf 個をブロック対角に複製した dist_mat を
        flow.sample() に渡す（BFS を n_conf 回 → 1 回に削減）。
      - flow.sample() 自体の呼び出しも n_conf 回 → 1 回に削減（ODE ステップ内の
        DiT forward が n_conf 分子分をまとめて 1 回の attention で処理される）。
      - 各コンフォーマーは flow.sample() 内部で独立にサンプリングされる
        Z1 ~ N(0,I) から出発するため、逐次実装と同じく互いに異なる乱数となる。
    """
    use_rwpe = cond_encoder.use_rwpe
    use_lappe = cond_encoder.use_lappe

    data = smiles_to_pyg(smiles, add_h, use_rwpe, use_lappe, device)
    n_atoms = data.atom_type_idx.size(0)
    batch_idx = data.batch

    # Conditioning（分子 1 本分のみ計算すればよい）
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

    if n_atoms == 0:
        return []

    # ── conformer バッチ化: n_conf 回のノード複製をブロック対角に構成 ──
    edge_index = data.edge_index                      # (2, E)
    total_atoms = n_atoms * n_conf

    # cond / e_cond は分子構造に依存するだけなので単純にタイル
    cond_rep = cond.repeat(n_conf, 1)                  # (n_conf*n_atoms, cond_dim)
    e_cond_rep = e_cond.repeat(n_conf, 1) if e_cond.numel() > 0 else e_cond

    # edge_index はブロックごとに n_atoms オフセットして連結
    if edge_index.numel() > 0:
        offsets = (torch.arange(n_conf, device=device) * n_atoms).repeat_interleave(
            edge_index.size(1)
        )
        edge_index_rep = edge_index.repeat(1, n_conf) + offsets.unsqueeze(0)
    else:
        edge_index_rep = edge_index.new_empty((2, 0))

    # batch テンソル: どのコンフォーマー（ブロック）に属するかを示す
    batch_rep = torch.arange(n_conf, device=device).repeat_interleave(n_atoms)

    # グラフ距離 BFS は分子 1 本分だけ計算し、block_diag でブロック対角に複製。
    # dist_mat を flow.sample() に渡すことで、DiT 側の BFS 再計算（n_conf 回）
    # を完全にスキップできる（異ブロック間は block-diagonal mask で -1e9 に
    # 上書きされるため block_diag の off-diagonal 値は 0 で問題ない）。
    dist_single = compute_graph_distance(edge_index, n_atoms)   # (n_atoms, n_atoms)
    dist_mat_rep = torch.block_diag(*([dist_single] * n_conf)) if n_conf > 1 else dist_single

    # DiT ODE サンプリング → Z0（n_conf 本まとめて 1 回だけ呼ぶ）
    z0 = flow.sample(
        n_atoms=total_atoms,
        cond=cond_rep,
        batch=batch_rep,
        n_steps=n_steps,
        edge_index=edge_index_rep,
        device=device,
        dist_mat=dist_mat_rep,
    )

    # ── 学習時設定に応じた decode 入力のゲート（旧 ckpt 完全後方互換）──
    # margs から EGT/MDS の有無を読む。旧 ckpt はこれらのキーが無いため
    # getattr のデフォルト（0 / False）で「従来と同一」の decode 経路になる。
    dec_uses_egt = getattr(margs, 'egt_every', 0) > 0   # decoder に EGT 層があるか
    use_mds = getattr(margs, 'mds_init', False)          # MDS 大域足場で学習したか

    # dist_mat: decoder が EGT を使う場合のみ渡す。EGTLayer は分子内ブロック
    # dm[nl][:,nl] のみ参照するため、dist_mat_rep（block_diag, 各対角ブロック＝
    # compute_graph_distance(edge_index, n_atoms, max_dist=4)）は dataset.py:157 の
    # 学習時 per-molecule dist_mat と値が一致する（off-diagonal は未参照で無害）。
    # EGT 非使用の旧 ckpt では None を渡し従来と完全一致。
    decode_dist_mat = dist_mat_rep if dec_uses_egt else None

    # init_scaffold: mds_init 学習時のみ。MDS 足場は純トポロジー量なので
    # 分子 1 本分を mds_init_coords(local edge_index, n_atoms) で計算し、
    # n_conf 個を単純 cat で縦連結する（dataset.collate と同じ規約：オフセット不要、
    # pos/cond_rep/batch_rep と同一の分子順・原子順）。旧 ckpt では None（従来経路）。
    if use_mds:
        scaffold_single = mds_init_coords(edge_index, n_atoms)   # (n_atoms, 3)
        init_scaffold = (torch.cat([scaffold_single] * n_conf, dim=0)
                         if n_conf > 1 else scaffold_single)      # (total_atoms, 3)
    else:
        init_scaffold = None

    # VAE Decoder → 座標（block-diagonal のまま一括デコード）
    pos = vae.decode(z0, cond_rep, edge_index_rep, e_cond_rep, batch_rep,
                     dist_mat=decode_dist_mat, init_scaffold=init_scaffold)  # (total_atoms, 3)

    # ── 幾何緩和（後処理）: decode 座標に clash+bond のヒンジエネルギーを掛けて
    #    局所違反（食い込み・結合長逸脱）を解消する。block-diagonal の
    #    edge_index_rep / dist_mat_rep / batch_rep をそのまま流用する。
    #    @torch.no_grad() の内側なので必ず enable_grad で包む。
    if relax:
        atomic_nums_rep = data.atomic_nums.repeat(n_conf)   # (total_atoms,)
        with torch.enable_grad():
            pos = relax_coords(
                pos, edge_index_rep, dist_mat_rep, atomic_nums_rep,
                batch_rep, ptr=None,
                steps=relax_steps, lr=relax_lr,
                w_clash=relax_w_clash, w_bond=relax_w_bond,
                w_anchor=relax_w_anchor)

    pos_np = pos.cpu().numpy().astype(float)

    conformers = []
    for conf_idx in range(n_conf):
        sl = slice(conf_idx * n_atoms, (conf_idx + 1) * n_atoms)
        pos_i = pos_np[sl]

        mol = Chem.RWMol(rdmol)
        conf = Chem.Conformer(n_atoms)
        for i, (x, y, z) in enumerate(pos_i):
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

    cond_encoder, vae, flow, margs = load_models(args.vae_checkpoint, args.dit_checkpoint, device)
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
                smi, cond_encoder, vae, flow, device, margs,
                n_conf=args.n_conf, add_h=args.add_h, n_steps=args.n_steps,
                relax=args.relax, relax_steps=args.relax_steps,
                relax_lr=args.relax_lr, relax_w_clash=args.relax_w_clash,
                relax_w_bond=args.relax_w_bond, relax_w_anchor=args.relax_w_anchor,
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
