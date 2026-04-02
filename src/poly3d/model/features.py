"""
分子特徴量の定義とデータ変換。

カテゴリカル特徴（atom type, hybridization, bond type）は
nn.Embedding で学習可能な埋め込みに変換するため、インデックスとして返す。
連続値特徴は別途返す。

RWPE (Random Walk Positional Encoding) と LapPE (Laplacian PE) の計算も担当。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from rdkit.Chem import Atom, Bond, Mol

# ── 原子タイプ辞書 ────────────────────────────────────────────────────────────
# インデックス 0〜15: known elements, 16: "other"
ATOM_TYPES: List[int] = [
    1,   # H
    5,   # B
    6,   # C
    7,   # N
    8,   # O
    9,   # F
    14,  # Si
    15,  # P
    16,  # S
    17,  # Cl
    32,  # Ge
    34,  # Se
    35,  # Br
    50,  # Sn
    52,  # Te
    53,  # I
]
ATOM_TYPE_VOCAB = len(ATOM_TYPES) + 1  # +1 for "other"
_ATOM_Z_TO_IDX = {z: i for i, z in enumerate(ATOM_TYPES)}

# ── 混成軌道 ──────────────────────────────────────────────────────────────────
# インデックス 0〜4: known, 5: "other"
try:
    from rdkit.Chem import rdchem as _rc
    HYBRIDIZATION_TYPES = [
        _rc.HybridizationType.SP,
        _rc.HybridizationType.SP2,
        _rc.HybridizationType.SP3,
        _rc.HybridizationType.SP3D,
        _rc.HybridizationType.SP3D2,
    ]
    _HYB_TO_IDX = {h: i for i, h in enumerate(HYBRIDIZATION_TYPES)}
except ImportError:
    HYBRIDIZATION_TYPES = []
    _HYB_TO_IDX = {}

HYBRIDIZATION_VOCAB = len(HYBRIDIZATION_TYPES) + 1

# ── 結合タイプ ────────────────────────────────────────────────────────────────
# 0: SINGLE, 1: DOUBLE, 2: TRIPLE, 3: AROMATIC, 4: OTHER/UNKNOWN
BOND_TYPE_VOCAB = 5

# ── 特徴量次元数（参照用定数） ─────────────────────────────────────────────────
ATOM_CONT_DIM = 5   # [aromatic, in_ring, formal_charge, total_Hs, mass/100]
BOND_CONT_DIM = 2   # [is_conjugated, is_in_ring]
RWPE_DIM = 16       # ランダムウォーク PE のステップ数
LAPPE_DIM = 8       # Laplacian PE の eigenvector 数


# ── 原子・結合ごとの特徴量抽出 ─────────────────────────────────────────────────

def atom_type_idx(atom: "Atom") -> int:
    """原子番号 → embedding インデックス。"""
    return _ATOM_Z_TO_IDX.get(atom.GetAtomicNum(), len(ATOM_TYPES))


def hybridization_idx(atom: "Atom") -> int:
    """混成軌道 → embedding インデックス。"""
    return _HYB_TO_IDX.get(atom.GetHybridization(), len(HYBRIDIZATION_TYPES))


def atom_cont_features(atom: "Atom") -> List[float]:
    """原子の連続値特徴量（次元数 = ATOM_CONT_DIM）。"""
    return [
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs()),
        atom.GetMass() / 100.0,
    ]


def bond_type_idx(bond: "Bond") -> int:
    """結合タイプ → embedding インデックス。"""
    from rdkit.Chem import rdchem
    bt = bond.GetBondType()
    if bt == rdchem.BondType.SINGLE:   return 0
    if bt == rdchem.BondType.DOUBLE:   return 1
    if bt == rdchem.BondType.TRIPLE:   return 2
    if bt == rdchem.BondType.AROMATIC: return 3
    return 4


def bond_cont_features(bond: "Bond") -> List[float]:
    """結合の連続値特徴量（次元数 = BOND_CONT_DIM）。"""
    return [
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


# ── ポジショナルエンコーディング ───────────────────────────────────────────────

def compute_rwpe(edge_index: np.ndarray, num_nodes: int, k: int = RWPE_DIM) -> np.ndarray:
    """
    Random Walk Positional Encoding (Dwivedi et al., 2022).

    pe[i, t] = (T^t)[i, i]  ここで T = D^{-1} A は遷移行列。
    各ノードの「自己帰還確率」を k ステップ分計算する。

    Parameters
    ----------
    edge_index : (2, E) np.int64 — 有向エッジ（両方向含む）
    num_nodes  : N
    k          : ランダムウォークのステップ数

    Returns
    -------
    pe : (N, k) np.float32
    """
    import scipy.sparse as sp

    if edge_index.shape[1] == 0:
        return np.zeros((num_nodes, k), dtype=np.float32)

    row, col = edge_index
    data = np.ones(len(row), dtype=np.float32)
    A = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv = np.where(deg > 0, 1.0 / deg, 0.0)
    T = sp.diags(deg_inv) @ A   # 遷移行列: D^{-1} A

    pe = np.zeros((num_nodes, k), dtype=np.float32)
    T_power = T.copy().astype(np.float32)
    for i in range(k):
        pe[:, i] = np.array(T_power.diagonal(), dtype=np.float32)
        T_power = T_power @ T

    return pe


def compute_lappe(
    edge_index: np.ndarray,
    num_nodes: int,
    k: int = LAPPE_DIM,
) -> np.ndarray:
    """
    Laplacian Positional Encoding (Dwivedi & Bresson, 2020).

    正規化グラフラプラシアンの固有ベクトル（第 2〜k+1 番目）を返す。
    固有値 0 に対応する定数ベクトルはスキップする。

    Returns
    -------
    pe : (N, k) np.float32  (固有ベクトルが k 本未満の場合はゼロパディング)
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    pe = np.zeros((num_nodes, k), dtype=np.float32)

    if num_nodes < k + 2 or edge_index.shape[1] == 0:
        return pe

    row, col = edge_index
    data = np.ones(len(row), dtype=np.float32)
    A = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    # 対称化（無向グラフ前提）
    A = (A + A.T) / 2

    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    L_sym = sp.eye(num_nodes) - D_inv_sqrt @ A @ D_inv_sqrt

    try:
        # k+1 本求めて第 1 番（定数ベクトル）をスキップ
        n_eig = min(k + 1, num_nodes - 1)
        eigvals, eigvecs = spla.eigsh(L_sym, k=n_eig, which='SM', tol=1e-3)
        # 固有値でソート（小さい順）、第 1 番をスキップ
        order = np.argsort(eigvals)
        vecs = eigvecs[:, order[1:k+1]]  # (N, ≤k)
        n_actual = vecs.shape[1]
        pe[:, :n_actual] = vecs.astype(np.float32)
    except Exception:
        pass

    return pe


# ── mol → データ dict 変換 ─────────────────────────────────────────────────────

def mol_to_data(
    mol: "Mol",
    sid: str = '',
    center: bool = True,
    use_rwpe: bool = True,
    use_lappe: bool = False,
) -> Dict[str, Any]:
    """
    RDKit Mol → 学習用データ dict。

    mol は 3D Conformer を持っている必要がある。

    Returns
    -------
    {
        atom_type_idx  : np.int32  (N,)
        hyb_idx        : np.int32  (N,)
        atom_cont      : np.float32 (N, ATOM_CONT_DIM)
        bond_type_idx  : np.int32  (2E,)  両方向
        bond_cont      : np.float32 (2E, BOND_CONT_DIM)
        edge_index     : np.int64  (2, 2E)
        n_atoms        : int
        rwpe           : np.float32 (N, RWPE_DIM)   use_rwpe=True の場合
        lappe          : np.float32 (N, LAPPE_DIM)  use_lappe=True の場合
        pos            : np.float32 (N, 3)  重心ゼロ (center=True)
        atomic_nums    : np.int32  (N,)
        sid            : str
    }
    """
    n = mol.GetNumAtoms()
    if n == 0:
        raise ValueError('原子数が 0')
    if mol.GetNumConformers() == 0:
        raise ValueError('3D Conformer が存在しない')

    conf = mol.GetConformer()

    # ── カテゴリカル特徴量 ──
    at_idx = np.array([atom_type_idx(a) for a in mol.GetAtoms()], dtype=np.int32)
    hy_idx = np.array([hybridization_idx(a) for a in mol.GetAtoms()], dtype=np.int32)
    ac = np.array([atom_cont_features(a) for a in mol.GetAtoms()], dtype=np.float32)

    # ── 3D 座標 ──
    pos = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(n)],
        dtype=np.float32,
    )
    if center:
        pos = pos - pos.mean(axis=0)

    # ── 結合特徴量（双方向） ──
    bt_idx: List[int] = []
    bc: List[List[float]] = []
    edges: List[List[int]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        btype = bond_type_idx(bond)
        bcont = bond_cont_features(bond)
        bt_idx += [btype, btype]
        bc += [bcont, bcont]
        edges += [[i, j], [j, i]]

    if edges:
        edge_index = np.array(edges, dtype=np.int64).T     # (2, 2E)
        bt_idx_arr = np.array(bt_idx, dtype=np.int32)
        bc_arr = np.array(bc, dtype=np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        bt_idx_arr = np.empty(0, dtype=np.int32)
        bc_arr = np.empty((0, BOND_CONT_DIM), dtype=np.float32)

    # ── ポジショナルエンコーディング ──
    data: Dict = {
        'atom_type_idx': at_idx,
        'hyb_idx': hy_idx,
        'atom_cont': ac,
        'bond_type_idx': bt_idx_arr,
        'bond_cont': bc_arr,
        'edge_index': edge_index,
        'n_atoms': n,
        'pos': pos,
        'atomic_nums': np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32),
        'sid': sid,
    }

    if use_rwpe:
        data['rwpe'] = compute_rwpe(edge_index, n)
    if use_lappe:
        data['lappe'] = compute_lappe(edge_index, n)

    return data


def smiles_to_data(
    smiles: str,
    add_h: bool = True,
    use_rwpe: bool = True,
    use_lappe: bool = False,
) -> Dict[str, Any]:
    """
    SMILES → 推論用グラフデータ（3D 座標なし）。

    Returns
    -------
    mol_to_data と同一キーを持つ dict（pos キーは存在しない）
    """
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'SMILES パース失敗: {smiles}')
    if add_h:
        mol = Chem.AddHs(mol)

    n = mol.GetNumAtoms()
    at_idx = np.array([atom_type_idx(a) for a in mol.GetAtoms()], dtype=np.int32)
    hy_idx = np.array([hybridization_idx(a) for a in mol.GetAtoms()], dtype=np.int32)
    ac = np.array([atom_cont_features(a) for a in mol.GetAtoms()], dtype=np.float32)

    bt_idx: List[int] = []
    bc: List[List[float]] = []
    edges: List[List[int]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt_idx += [bond_type_idx(bond)] * 2
        bc += [bond_cont_features(bond)] * 2
        edges += [[i, j], [j, i]]

    if edges:
        edge_index = np.array(edges, dtype=np.int64).T
        bt_idx_arr = np.array(bt_idx, dtype=np.int32)
        bc_arr = np.array(bc, dtype=np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        bt_idx_arr = np.empty(0, dtype=np.int32)
        bc_arr = np.empty((0, BOND_CONT_DIM), dtype=np.float32)

    data: Dict = {
        'atom_type_idx': at_idx,
        'hyb_idx': hy_idx,
        'atom_cont': ac,
        'bond_type_idx': bt_idx_arr,
        'bond_cont': bc_arr,
        'edge_index': edge_index,
        'atomic_nums': np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int32),
        'n_atoms': n,
        'sid': smiles,
    }

    if use_rwpe:
        data['rwpe'] = compute_rwpe(edge_index, n)
    if use_lappe:
        data['lappe'] = compute_lappe(edge_index, n)

    return data
