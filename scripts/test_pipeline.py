"""
パイプライン全体の動作確認（VAE + DiT アーキテクチャ）

実行:
  cd C:/Users/shanu/Documents/Python/poly3D
  "C:/Users/shanu/anaconda3/envs/polygen/python.exe" scripts/test_pipeline.py
"""
import sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL = '[ OK ]', '[FAIL]'


def test_lmdb_reader():
    from poly3d.preprocess.lmdb_reader import iter_lmdb, read_molecule, count_entries
    path = r'D:\Dataset\OMol_base\OPoly26\val\data0000.aselmdb'
    n = count_entries(path)
    assert n > 0
    for i, rec in enumerate(iter_lmdb(path)):
        nums, pos, charge, sid = read_molecule(rec)
        assert pos.shape == (len(nums), 3)
        if i >= 2: break
    print(f'  n_entries={n}, last n_atoms={len(nums)}, charge={charge}')
    return True


def test_features():
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from poly3d.model.features import (
        mol_to_data, smiles_to_data,
        ATOM_TYPE_VOCAB, HYBRIDIZATION_VOCAB, BOND_TYPE_VOCAB,
        ATOM_CONT_DIM, BOND_CONT_DIM, RWPE_DIM,
    )

    smi = 'CC(=O)Nc1ccc(O)cc1'
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    d = mol_to_data(mol, use_rwpe=True)
    n = mol.GetNumAtoms()
    assert d['atom_type_idx'].shape == (n,)
    assert d['hyb_idx'].shape == (n,)
    assert d['atom_cont'].shape == (n, ATOM_CONT_DIM)
    assert d['rwpe'].shape == (n, RWPE_DIM)
    assert abs(d['pos'].mean(axis=0)).max() < 1e-5, '重心ゼロでない'
    assert d['atom_type_idx'].max() < ATOM_TYPE_VOCAB
    assert d['hyb_idx'].max() < HYBRIDIZATION_VOCAB
    assert d['bond_type_idx'].max() < BOND_TYPE_VOCAB

    sg = smiles_to_data(smi, use_rwpe=True)
    assert sg['rwpe'].shape == (sg['atom_type_idx'].shape[0], RWPE_DIM)
    print(f'  n={n}, ATOM_TYPE_VOCAB={ATOM_TYPE_VOCAB}, rwpe OK')
    return True


def test_cond_encoder():
    import torch
    from poly3d.model.cond_encoder import ConditionalEncoder
    from poly3d.model.features import ATOM_CONT_DIM, BOND_CONT_DIM, RWPE_DIM

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = ConditionalEncoder(hidden_dim=32, edge_dim=16, n_layers=2).to(device)

    N, E, B = 15, 20, 2
    batch = torch.zeros(N, dtype=torch.long, device=device)
    batch[8:] = 1  # 2分子

    h, e, cond = enc(
        atom_type_idx=torch.randint(0, 17, (N,), device=device),
        hyb_idx=torch.randint(0, 6, (N,), device=device),
        atom_cont=torch.randn(N, ATOM_CONT_DIM, device=device),
        bond_type_idx=torch.randint(0, 5, (E,), device=device),
        bond_cont=torch.randn(E, BOND_CONT_DIM, device=device),
        edge_index=torch.randint(0, N, (2, E), device=device),
        rwpe=torch.randn(N, RWPE_DIM, device=device),
        batch=batch,
    )
    assert h.shape == (N, 32)
    assert e.shape == (E, 16)
    assert cond.shape == (N, 32), f'cond shape: {cond.shape}'
    print(f'  h={h.shape}, e={e.shape}, cond={cond.shape}')
    return True


def test_vae():
    import torch
    from poly3d.model.cond_encoder import ConditionalEncoder
    from poly3d.model.vae import StructuralVAE
    from poly3d.model.vae_loss import vae_loss
    from poly3d.model.features import ATOM_CONT_DIM, BOND_CONT_DIM, RWPE_DIM

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    enc = ConditionalEncoder(hidden_dim=32, edge_dim=16, n_layers=2).to(device)
    vae = StructuralVAE(
        cond_dim=32, edge_dim=16,
        hidden_dim=32, latent_dim=8,
        enc_layers=2, dec_layers=2,
    ).to(device)

    N, E = 12, 18
    batch = torch.tensor([0]*7 + [1]*5, dtype=torch.long, device=device)
    ei = torch.randint(0, N, (2, E), device=device)
    pos_gt = torch.randn(N, 3, device=device)

    h, e_cond, cond = enc(
        atom_type_idx=torch.randint(0, 17, (N,), device=device),
        hyb_idx=torch.randint(0, 6, (N,), device=device),
        atom_cont=torch.randn(N, ATOM_CONT_DIM, device=device),
        bond_type_idx=torch.randint(0, 5, (E,), device=device),
        bond_cont=torch.randn(E, BOND_CONT_DIM, device=device),
        edge_index=ei,
        rwpe=torch.randn(N, RWPE_DIM, device=device),
        batch=batch,
    )

    pos_pred, mu, logvar, _, _ = vae(cond, pos_gt, ei, e_cond, batch)
    assert pos_pred.shape == (N, 3)
    assert mu.shape == (N, 8)

    loss, ld = vae_loss(pos_pred, pos_gt, mu, logvar, ei, N, beta=0.01, batch=batch)
    assert not torch.isnan(loss), f'loss NaN'
    print(f'  pos_pred={pos_pred.shape}, loss={loss.item():.4f}, keys={list(ld.keys())}')
    return True


def test_dit_flow():
    import torch
    from poly3d.model.dit import LatentDiT
    from poly3d.model.flow_matching import FlowMatching

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dit = LatentDiT(
        latent_dim=8, cond_dim=32, time_dim=16,
        hidden_dim=32, n_heads=4, n_layers=2,
        use_pos_bias=True,
    ).to(device)
    flow = FlowMatching(dit, t_max=0.9, p_selfcond=0.5).to(device)

    # 2 分子バッチ (N1=6, N2=4)
    N = 10
    batch = torch.tensor([0]*6 + [1]*4, dtype=torch.long, device=device)
    # edge_index: 分子内のみ（PyG Batch と同じ形式）
    ei = torch.tensor([
        [0,1,1,2,2,3, 6,7,7,8],
        [1,0,2,1,3,2, 7,6,8,7],
    ], dtype=torch.long, device=device)

    z0 = torch.randn(N, 8, device=device)
    cond = torch.randn(N, 32, device=device)

    # 損失計算
    loss, ld = flow.loss(z0, cond, batch, edge_index=ei)
    assert not torch.isnan(loss), f'loss NaN: {loss}'
    print(f'  flow loss={loss.item():.4f}')

    # サンプリング
    with torch.no_grad():
        z0_sampled = flow.sample(N, cond, batch, n_steps=5,
                                 edge_index=ei, device=device)
    assert z0_sampled.shape == (N, 8)
    print(f'  sample shape={z0_sampled.shape}')
    return True


def test_single_mol_preprocess():
    from poly3d.preprocess.lmdb_reader import iter_lmdb, read_molecule
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    from poly3d.model.features import mol_to_data

    path = r'D:\Dataset\OMol_base\OPoly26\val\data0000.aselmdb'
    ok = fail = 0
    for rec in iter_lmdb(path):
        nums, pos, charge, sid = read_molecule(rec)
        n = len(nums)
        from rdkit.Chem import GetPeriodicTable
        ptable = GetPeriodicTable()
        try:
            sym_list = [ptable.GetElementSymbol(int(z)) for z in nums]
        except Exception:
            fail += 1
            continue
        lines = [str(n), ''] + [f'{s} {x:.4f} {y:.4f} {z:.4f}'
                                  for s, (x, y, z) in zip(sym_list, pos)]
        try:
            mol = Chem.MolFromXYZBlock('\n'.join(lines))
            rdDetermineBonds.DetermineBonds(mol, charge=charge, maxIterations=2000)
            frags = Chem.GetMolFrags(mol, asMols=True)
            mol = max(frags, key=lambda m: m.GetNumAtoms())
            d = mol_to_data(mol, sid=sid, use_rwpe=True)
            assert d['rwpe'].shape[0] == d['atom_type_idx'].shape[0]
            ok += 1
        except Exception:
            fail += 1
        if ok + fail >= 20:
            break
    print(f'  {ok} 成功 / {fail} 失敗 (20 件中)')
    assert ok > 0
    return True


def run(name, fn):
    print(f'\n{name}')
    try:
        fn()
        print(f'{PASS} {name}')
        return True
    except Exception as e:
        print(f'{FAIL} {name}: {e}')
        traceback.print_exc()
        return False


if __name__ == '__main__':
    results = [
        run('lmdb_reader', test_lmdb_reader),
        run('features (embedding idx + RWPE)', test_features),
        run('cond_encoder (global pooling + Ci)', test_cond_encoder),
        run('vae (encode + decode + loss)', test_vae),
        run('dit + flow_matching (loss + sample)', test_dit_flow),
        run('single_mol_preprocess', test_single_mol_preprocess),
    ]
    print(f'\n{"="*50}')
    print(f'結果: {sum(results)}/{len(results)} テスト通過')
    sys.exit(0 if all(results) else 1)
