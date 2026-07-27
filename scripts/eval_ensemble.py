"""
同一 SMILES（= uuid）の「生成コンフォーマ集合 vs 真の参照集合」を
**分布ベース**で評価するスクリプト。

polyGen（arXiv:2504.17656）は RMSD ではなく内部座標（二面角など）の分布で
生成品質を評価する。本スクリプトはその思想を PolyOmics の真の配座アンサンブル
（1 uuid = 同一繰返し単位 × 多数の実 MD 配座）で厳密化する。

  参照集合 R : そのuuidの全実配座（最大 --max_ref をランダムサブサンプル）
  生成集合 G : モデルが生成した配座集合（--mode で生成器を切替）

生成器モード（--mode）:
  recon（既定） : 各実配座を encode→μ→decode（再構築を生成集合とする）
  prior         : z~N(0,I) を各原子潜在にサンプル→decode（--n_gen 本）
  dit           : --dit_checkpoint 指定時、flow.sample() で生成（sample.py 準拠）

指標（uuid ごとに代表値を算出 → 全体・サイズ帯別に集約）:
  1. 妥当性ゲート : clash率 / 結合長サニティ / RDKit sanitize（G に適用、fail は分布から除外）
  2. torsion分布JS（主指標）: 共有 quartet の二面角を周期ヒストグラム化して JS
  3. torsion 円環Wasserstein : 各 torsion の周期 Wasserstein-1
  4. TFD : RDKit TorsionFingerprints で G×R の TFD 中央値
  5. bond/angle 分布JS（補助）
  6. COV/MAT（δ Å, 重原子, Kabsch）: COV-R/P, MAT-R/P

使い方:
  "C:/Users/shanu/anaconda3/envs/polygen/python.exe" scripts/eval_ensemble.py \
      --checkpoint runs/polyomics_pilot/vae_best.pt \
      --val_lmdb   data/polyomics_PG_val.lmdb \
      --mode recon --max_smiles 20 --max_ref 100 --out ens_recon.json
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import lmdb
import numpy as np
import torch

# Windows の cp932 コンソールでも Unicode を出力できるようにする
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

from rdkit import Chem, RDLogger
from rdkit.Chem import TorsionFingerprints

from poly3d.data.dataset import ConformerDataset, collate_fn
from poly3d.model.geo_losses import _angle_between, _dihedral

# evaluate_vae のモデルロードを再利用（VAE チェックポイント → cond_encoder, vae, margs）
from evaluate_vae import _load_models

RDLogger.DisableLog('rdApp.*')   # sanitize の警告ログを抑制
PI = math.pi
_PT = Chem.GetPeriodicTable()

# 結合タイプ index → RDKit BondType（features.py の BOND_TYPE_VOCAB に一致）
#   0:SINGLE 1:DOUBLE 2:TRIPLE 3:AROMATIC 4:OTHER→SINGLE でフォールバック
_BT_MAP = {
    0: Chem.BondType.SINGLE,
    1: Chem.BondType.DOUBLE,
    2: Chem.BondType.TRIPLE,
    3: Chem.BondType.AROMATIC,
    4: Chem.BondType.SINGLE,
}


# ══════════════════════════════════════════════════════════════════════════════
# uuid グルーピング（参照アンサンブルの構築）
# ══════════════════════════════════════════════════════════════════════════════

def build_uuid_index(lmdb_path: str, max_smiles: int) -> 'OrderedDict[str, list]':
    """lmdb を走査して uuid → レコード idx リストの辞書を作る。

    sid = "<uuid>:<mol_id>:<resnum>" 形式。uuid = sid.split(':')[0] でグループ化。
    同一 uuid のレコードは lmdb 内で連続に並ぶ（build_dataset がファイル単位で
    逐次処理するため）。この連続性を利用し、max_smiles>0 のときは新しい uuid が
    上限を超えた時点で走査を打ち切る（全 DB 走査を回避）。連続でない場合でも
    グループが分割されるだけで破綻はしない（縮退動作、report 参照）。

    key は "%09d" 文字列（dataset.py の __getitem__ と同一規約）。
    """
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False, max_readers=256)
    groups: 'OrderedDict[str, list]' = OrderedDict()
    with env.begin() as txn:
        n = txn.get(b'__len__')
        total = int(n.decode('ascii')) if n else txn.stat()['entries']
        for idx in range(total):
            key = f'{idx:09d}'.encode('ascii')
            val = txn.get(key)
            if val is None:
                continue
            d = pickle.loads(val)
            uuid = d.get('sid', '').split(':')[0]
            if uuid == '':
                continue
            if uuid not in groups:
                if max_smiles > 0 and len(groups) >= max_smiles:
                    break   # 上限到達 & 新規 uuid 出現 → 連続性前提で打ち切り
                groups[uuid] = []
            groups[uuid].append(idx)
    env.close()
    return groups


# ══════════════════════════════════════════════════════════════════════════════
# 内部座標の抽出（共有トポロジー上で R / G をまとめて計算）
# ══════════════════════════════════════════════════════════════════════════════

def _stack_pos(pos_list: list) -> torch.Tensor:
    """(n,3) テンソルのリストを (K, n, 3) にスタック（fp32）。"""
    return torch.stack([p.float() for p in pos_list], dim=0)


def torsion_angles(pos_stack: torch.Tensor, quartets: torch.Tensor) -> torch.Tensor:
    """全配座の二面角を計算。(K,n,3), (Q,4) → (K, Q) ラジアン [-π, π]。"""
    if quartets is None or quartets.size(0) == 0:
        return pos_stack.new_zeros((pos_stack.size(0), 0))
    i, j, k, l = quartets[:, 0], quartets[:, 1], quartets[:, 2], quartets[:, 3]
    # _dihedral は (...,3) をブロードキャスト → (K,Q,3) 入力で (K,Q) 出力
    return _dihedral(pos_stack[:, i], pos_stack[:, j],
                     pos_stack[:, k], pos_stack[:, l])


def angle_values(pos_stack: torch.Tensor, triplets: torch.Tensor) -> torch.Tensor:
    """全配座の結合角を計算。(K,n,3), (T,3) → (K, T) ラジアン [0, π]。"""
    if triplets is None or triplets.size(0) == 0:
        return pos_stack.new_zeros((pos_stack.size(0), 0))
    i, j, k = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    v1 = pos_stack[:, i] - pos_stack[:, j]
    v2 = pos_stack[:, k] - pos_stack[:, j]
    return _angle_between(v1, v2)


def bond_lengths(pos_stack: torch.Tensor, bonds: torch.Tensor) -> torch.Tensor:
    """全配座の結合長を計算。(K,n,3), (2,B) → (K, B) Å。"""
    if bonds.size(1) == 0:
        return pos_stack.new_zeros((pos_stack.size(0), 0))
    bi, bj = bonds[0], bonds[1]
    return (pos_stack[:, bi] - pos_stack[:, bj]).norm(dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# 分布距離: JS divergence（周期対応）と 円環 Wasserstein-1
# ══════════════════════════════════════════════════════════════════════════════

def _hist_prob(x: np.ndarray, bins: int, lo: float, hi: float) -> np.ndarray:
    """[lo, hi] 上の正規化ヒストグラム（確率）。周期量は端点を範囲に含める。"""
    h, _ = np.histogram(x, bins=bins, range=(lo, hi))
    s = h.sum()
    if s == 0:
        return np.full(bins, 1.0 / bins)
    return h / s


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence（底 2、値域 [0, 1]）。"""
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return float(0.5 * kl_pm + 0.5 * kl_qm)


def _circular_w1(a_ref: np.ndarray, a_gen: np.ndarray,
                 bins: int = 36, lo: float = -PI, hi: float = PI) -> float:
    """円環（周期）Wasserstein-1 距離 [ラジアン]。

    周期 L=hi-lo の円上の 1D 最適輸送。ヒストグラム p, q の prefix-sum 差
    D_i = Σ_{k<=i}(p_k - q_k) に対し、円環上の EMD は定数シフト c を選んで
        EMD = min_c Σ_i |D_i - c| * binwidth
    で与えられ、最小化子 c は D_i の中央値（Werman/Rabin の円環 EMD 閉形式）。
    scipy.stats.wasserstein_distance を素朴に使うと周期の切れ目（±π 境界）で
    距離を過大評価するため、この円環版を用いる。
    """
    p = _hist_prob(a_ref, bins, lo, hi)
    q = _hist_prob(a_gen, bins, lo, hi)
    diff = np.cumsum(p - q)
    c = np.median(diff)
    binw = (hi - lo) / bins
    return float(np.sum(np.abs(diff - c)) * binw)


def per_element_js(ref_mat: torch.Tensor, gen_mat: torch.Tensor,
                   bins: int, lo: float, hi: float) -> float:
    """各内部座標（列）ごとに R/G のヒスト JS を計算し平均。列が 0 本なら NaN。"""
    if ref_mat.size(1) == 0 or ref_mat.size(0) == 0 or gen_mat.size(0) == 0:
        return float('nan')
    R = ref_mat.cpu().numpy()
    G = gen_mat.cpu().numpy()
    vals = []
    for c in range(R.shape[1]):
        p = _hist_prob(R[:, c], bins, lo, hi)
        q = _hist_prob(G[:, c], bins, lo, hi)
        vals.append(_js_divergence(p, q))
    return float(np.mean(vals)) if vals else float('nan')


def per_torsion_circular_w1(ref_mat: torch.Tensor, gen_mat: torch.Tensor,
                            bins: int = 36) -> float:
    """各 torsion の円環 W1 を計算し平均。torsion が 0 本なら NaN。"""
    if ref_mat.size(1) == 0 or ref_mat.size(0) == 0 or gen_mat.size(0) == 0:
        return float('nan')
    R = ref_mat.cpu().numpy()
    G = gen_mat.cpu().numpy()
    vals = [_circular_w1(R[:, c], G[:, c], bins=bins) for c in range(R.shape[1])]
    return float(np.mean(vals)) if vals else float('nan')


# ══════════════════════════════════════════════════════════════════════════════
# COV / MAT（重原子・Kabsch、全 R×G ペアをバッチ SVD で一括計算）
# ══════════════════════════════════════════════════════════════════════════════

def cov_mat_metrics(R_heavy: torch.Tensor, G_heavy: torch.Tensor,
                    delta: float) -> dict:
    """COV/MAT を計算する。

    R_heavy : (nr, m, 3)  参照集合の重原子座標
    G_heavy : (ng, m, 3)  生成集合の重原子座標（妥当性 pass のみ）
    delta   : COV の閾値 [Å]

    全 (r, g) ペアの Kabsch RMSD を (nr, ng) 行列としてバッチ SVD で一括計算する
    （evaluate_vae._kabsch_rmsd_single と同一の回転整合 RMSD）。原子順序は同一
    トポロジーの identity マッピング。

    COV-R = |{r: min_g RMSD < δ}| / |R|,   MAT-R = mean_r min_g RMSD
    COV-P = |{g: min_r RMSD < δ}| / |G|,   MAT-P = mean_g min_r RMSD
    """
    nr, m, _ = R_heavy.shape
    ng = G_heavy.shape[0]
    if nr == 0 or ng == 0 or m < 2:
        return {'cov_r': float('nan'), 'cov_p': float('nan'),
                'mat_r': float('nan'), 'mat_p': float('nan')}

    dev = R_heavy.device
    Rc = R_heavy - R_heavy.mean(dim=1, keepdim=True)   # (nr, m, 3)
    Gc = G_heavy - G_heavy.mean(dim=1, keepdim=True)   # (ng, m, 3)

    # 相関行列 H[r,g] = Σ_m Rc[r,m] ⊗ Gc[g,m]  → (nr, ng, 3, 3)
    H = torch.einsum('rmi,gmj->rgij', Rc, Gc)
    H = H + 1e-6 * torch.eye(3, device=dev)
    U, _S, Vh = torch.linalg.svd(H)                    # (nr,ng,3,3)
    d = torch.sign(torch.det(Vh.mT @ U.mT))            # (nr,ng)
    D = torch.eye(3, device=dev).expand(nr, ng, 3, 3).clone()
    D[..., 2, 2] = d
    Rot = Vh.mT @ D @ U.mT                              # (nr,ng,3,3)  Gc ≈ Rot·Rc

    # Rc を各ペアの最適回転で回してから Gc と比較
    Rc_rot = torch.einsum('rgij,rmj->rgmi', Rot, Rc)   # (nr,ng,m,3)
    rmsd = (Rc_rot - Gc.unsqueeze(0)).pow(2).sum(dim=-1).mean(dim=-1).sqrt()  # (nr,ng)

    min_over_g = rmsd.min(dim=1).values   # (nr,)  各 r に最も近い g
    min_over_r = rmsd.min(dim=0).values   # (ng,)  各 g に最も近い r
    return {
        'cov_r': (min_over_g < delta).float().mean().item(),
        'cov_p': (min_over_r < delta).float().mean().item(),
        'mat_r': min_over_g.mean().item(),
        'mat_p': min_over_r.mean().item(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 妥当性ゲート（clash / 結合長サニティ / RDKit sanitize）
# ══════════════════════════════════════════════════════════════════════════════

def build_rdmol(atomic_nums: np.ndarray, edge_index: np.ndarray,
                bond_type_idx: np.ndarray, atom_cont: np.ndarray):
    """トポロジーから RDKit Mol を再構成し sanitize する（幾何非依存）。

    トポロジーは uuid 内で固定なので sanitize 可否は 1 回判定すれば足りる。
    失敗時は None（TFD/sanitize ゲートで「使えない分子」として扱う）。
    """
    rw = Chem.RWMol()
    for z, ac in zip(atomic_nums, atom_cont):
        a = Chem.Atom(int(z))
        a.SetNoImplicit(True)                     # 水素は明示的に含まれる
        a.SetFormalCharge(int(round(float(ac[2]))))  # atom_cont[2] = 形式電荷
        if float(ac[0]) > 0.5:                    # atom_cont[0] = 芳香族フラグ
            a.SetIsAromatic(True)
        rw.AddAtom(a)

    seen = set()
    for e in range(edge_index.shape[1]):
        i, j = int(edge_index[0, e]), int(edge_index[1, e])
        key = (i, j) if i < j else (j, i)
        if key in seen or i == j:
            continue
        seen.add(key)
        bt = _BT_MAP.get(int(bond_type_idx[e]), Chem.BondType.SINGLE)
        rw.AddBond(i, j, bt)
        if bt == Chem.BondType.AROMATIC:
            rw.GetBondBetweenAtoms(i, j).SetIsAromatic(True)

    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def atom_match_to_ref(mol_ref, atomic_nums: np.ndarray, edge_index: np.ndarray,
                      bond_type_idx: np.ndarray, atom_cont: np.ndarray):
    """conformer のトポロジーから mol を作り、mol_ref への原子対応 match を返す。

    ★PolyOmics は同一 uuid でも配座ごとに原子ラベリングが異なる（対称な原子が
    入れ替わる）。全指標は uuid 内で共有の topo（=mol_ref）の quartet/triplet/
    bond/dist_mat/heavy_idx を使うため、各配座の座標を mol_ref の並びへ揃えない
    と torsion/COV/TFD/clash が全て破綻する（conf0 の結合を別配座の座標に当てる
    と 1.5Å→5.6Å に飛ぶことを実測確認）。RDKit 部分構造マッチで対応を取る。

    戻り値: match（list, len == mol_ref の原子数, match[k]=この conformer 内で
    mol_ref の原子 k に対応する index）。`pos[match]` で mol_ref の並びに揃う。
    対称分子では自己同型が複数あるが任意の 1 つを取る（対称原子は化学的に等価
    なので分布指標には影響しない）。mol 構築失敗・マッチ不成立時は None。
    """
    mj = build_rdmol(atomic_nums, edge_index, bond_type_idx, atom_cont)
    if mj is None:
        return None
    match = mj.GetSubstructMatch(mol_ref)
    if len(match) != mol_ref.GetNumAtoms():
        return None
    return list(match)


def validity_of_conformer(pos: np.ndarray, rvdw: np.ndarray,
                          nonbond_mask: np.ndarray, bi: np.ndarray, bj: np.ndarray,
                          sanitize_ok: bool,
                          clash_factor: float = 0.6,
                          bond_lo: float = 0.7, bond_hi: float = 2.6) -> bool:
    """1 配座の妥当性を判定する。

    - clash : 非結合原子ペアで dist < (vdW_i + vdW_j) * clash_factor があれば fail
    - 結合長サニティ : 結合原子ペア距離が [bond_lo, bond_hi] Å を外れれば fail
    - sanitize : トポロジー再構成が sanitize 可能か（幾何非依存、uuid 単位で共通）

    nonbond_mask : (n, n) bool  clash 判定対象の「真に非結合」ペア = グラフ距離
        3 ホップ以上（1-2 結合・1-3 幾何隣接を除外）。直接結合のみ除外すると
        1-3 幾何隣接（geminal）が誤って clash と判定され、実 MD 参照配座すら
        大量に fail するため、標準的な steric-clash 定義に合わせ 1-2/1-3 を除外する。
    """
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))          # (n, n)

    # clash: 真に非結合なペア（グラフ距離 >= 3）のみを対象
    thr = (rvdw[:, None] + rvdw[None, :]) * clash_factor
    if np.any(nonbond_mask & (dist < thr)):
        return False

    # 結合長サニティ
    if bi.size > 0:
        bd = dist[bi, bj]
        if np.any(bd < bond_lo) or np.any(bd > bond_hi):
            return False

    return bool(sanitize_ok)


def tfd_median(mol, R_heavy_free_pos: list, G_pos: list,
               max_r_tfd: int, rng: np.random.Generator) -> float:
    """G の各配座 vs R の代表配座で TFD を計算し中央値を返す。

    mol : トポロジー再構成済み RDKit Mol（sanitize 済み）
    R_heavy_free_pos / G_pos : 全原子座標 (n,3) numpy のリスト
    max_r_tfd : TFD 計算に使う参照配座数の上限（計算量抑制）

    RDKit TorsionFingerprints.GetTFDBetweenConformers で全 G×R ペアの TFD を得る。
    使えない分子（torsion 定義不可・conformer 追加失敗）は NaN。
    """
    if mol is None or len(G_pos) == 0 or len(R_heavy_free_pos) == 0:
        return float('nan')
    n = mol.GetNumAtoms()
    R_sel = R_heavy_free_pos
    if len(R_sel) > max_r_tfd:
        pick = rng.choice(len(R_sel), size=max_r_tfd, replace=False)
        R_sel = [R_sel[k] for k in pick]

    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    r_ids, g_ids = [], []
    try:
        for p in R_sel:
            conf = Chem.Conformer(n)
            for a in range(n):
                conf.SetAtomPosition(a, (float(p[a, 0]), float(p[a, 1]), float(p[a, 2])))
            r_ids.append(m.AddConformer(conf, assignId=True))
        for p in G_pos:
            conf = Chem.Conformer(n)
            for a in range(n):
                conf.SetAtomPosition(a, (float(p[a, 0]), float(p[a, 1]), float(p[a, 2])))
            g_ids.append(m.AddConformer(conf, assignId=True))
        tfds = TorsionFingerprints.GetTFDBetweenConformers(m, confIds1=g_ids, confIds2=r_ids)
    except Exception:
        return float('nan')
    if not tfds:
        return float('nan')
    return float(np.median(np.asarray(tfds, dtype=np.float64)))


# ══════════════════════════════════════════════════════════════════════════════
# 生成器（recon / prior / dit）
# ══════════════════════════════════════════════════════════════════════════════

def _chunks(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


@torch.no_grad()
def generate_positions(gen_datas: list, mode: str, cond_encoder, vae, flow,
                       device, margs, batch_size: int, n_steps: int) -> list:
    """gen_datas（トポロジー Data のリスト）から生成配座座標を返す。

    recon : 各 Data 自身の実座標 pos を encode→μ→decode
    prior : z~N(0,I) を各原子潜在にサンプル→decode
    dit   : flow.sample() で z0 を得て decode（sample.py 準拠）

    戻り値: 各配座の (n, 3) fp32 cpu テンソルのリスト（gen_datas と同順）。
    collate_fn は入力 Data の topology 属性を破壊的に消費するため、gen_datas は
    使い捨て（呼び出し側で毎回 fresh に取得する）。
    """
    latent = vae.latent_dim
    out = []
    for chunk in _chunks(gen_datas, batch_size):
        batch = collate_fn(list(chunk)).to(device)
        _, e_cond, cond = cond_encoder(
            batch.atom_type_idx, batch.hyb_idx, batch.atom_cont,
            batch.bond_type_idx, batch.bond_cont, batch.edge_index,
            rwpe=getattr(batch, 'rwpe', None),
            lappe=getattr(batch, 'lappe', None), batch=batch.batch,
        )
        dm = getattr(batch, 'dist_mat', None)
        scaf = getattr(batch, 'init_scaffold', None)
        n_total = batch.num_nodes

        if mode == 'recon':
            mu, _ = vae.encoder(cond, batch.pos, batch.edge_index, e_cond,
                                batch.batch, dist_mat=dm)
            z = mu
        elif mode == 'prior':
            z = torch.randn(n_total, latent, device=device)
        elif mode == 'dit':
            # dist_mat（block 対角・off-diag=far）はブロック内のみ参照されるため
            # collate の dist_mat をそのまま flow.sample に供給できる（sample.py と等価）。
            z = flow.sample(n_atoms=n_total, cond=cond, batch=batch.batch,
                            n_steps=n_steps, edge_index=batch.edge_index,
                            device=device, dist_mat=dm)
        else:
            raise ValueError(f'unknown mode: {mode}')

        pos = vae.decode(z, cond, batch.edge_index, e_cond, batch.batch,
                         dist_mat=dm, init_scaffold=scaf).float()
        ptr = batch.ptr
        for m in range(ptr.numel() - 1):
            out.append(pos[int(ptr[m]):int(ptr[m + 1])].cpu())
    return out


# ══════════════════════════════════════════════════════════════════════════════
# uuid 単位の評価
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_uuid(uuid: str, indices: list, ds: ConformerDataset,
                  cond_encoder, vae, flow, device, margs, args,
                  rng: np.random.Generator) -> dict | None:
    """1 uuid の参照/生成集合を作り全指標の代表値を算出する。"""
    # ── 参照サブサンプル（seed 固定で決定的）──
    idxs = list(indices)
    if len(idxs) > args.max_ref:
        pick = rng.choice(len(idxs), size=args.max_ref, replace=False)
        idxs = [idxs[k] for k in sorted(pick.tolist())]

    R_datas = [ds[i] for i in idxs]
    R_datas = [d for d in R_datas if d is not None]   # max_atoms フィルタ
    if len(R_datas) < 2:
        return None

    # ── 参照フレーム（=基準ラベリング）の選定 ──
    # ★同一 uuid でも配座ごとに原子ラベリングが異なるため、共有トポロジーを直接
    #   使うと torsion/COV/clash が破綻する。最初に sanitize 可能な配座を topo に
    #   選び、以降すべての配座（参照・生成）をこの topo の並びへリマップする。
    rdmol = None
    topo = R_datas[0]
    for d in R_datas:
        m = build_rdmol(d.atomic_nums.cpu().numpy(), d.edge_index.cpu().numpy(),
                        d.bond_type_idx.cpu().numpy(), d.atom_cont.cpu().numpy())
        if m is not None:
            rdmol = m
            topo = d
            break
    sanitize_ok = rdmol is not None

    n_atoms = int(topo.num_nodes)
    edge_index_np = topo.edge_index.cpu().numpy()
    atomic_nums = topo.atomic_nums.cpu().numpy()
    quartets = topo.quartets if hasattr(topo, 'quartets') else None
    triplets = topo.triplets if hasattr(topo, 'triplets') else None

    # 共有の結合ペア（無向・重複除去）
    ei = topo.edge_index
    und = ei[0] < ei[1]
    bonds = torch.stack([ei[0][und], ei[1][und]], dim=0)   # (2, B)
    bi_np = bonds[0].cpu().numpy()
    bj_np = bonds[1].cpu().numpy()
    # clash 判定対象 = グラフ距離 3 ホップ以上の「真に非結合」ペア。
    # dist_mat は 0..4 にクランプ済み（0=自己, 1=結合, 2=1-3, 3, 4=far）。
    if hasattr(topo, 'dist_mat'):
        hop = topo.dist_mat.cpu().numpy().astype(np.int16)
        nonbond_mask = hop >= 3
    else:   # dist_mat が無い場合のフォールバック（直接結合のみ除外）
        nonbond_mask = np.ones((n_atoms, n_atoms), dtype=bool)
        nonbond_mask[edge_index_np[0], edge_index_np[1]] = False
        nonbond_mask[edge_index_np[1], edge_index_np[0]] = False
        np.fill_diagonal(nonbond_mask, False)

    heavy = atomic_nums > 1
    heavy_idx = np.nonzero(heavy)[0]
    n_heavy = int(heavy.sum())

    rvdw = np.array([_PT.GetRvdw(int(z)) for z in atomic_nums], dtype=np.float64)

    def _match(d):
        """d を topo フレームへ揃える match（list）。rdmol 無しは None（整列不能）。"""
        if rdmol is None:
            return None
        return atom_match_to_ref(rdmol, d.atomic_nums.cpu().numpy(),
                                 d.edge_index.cpu().numpy(),
                                 d.bond_type_idx.cpu().numpy(),
                                 d.atom_cont.cpu().numpy())

    # ── 参照座標を topo フレームへリマップ（マッチ不成立は除外）──
    R_pos = []
    for d in R_datas:
        mt = _match(d)
        if mt is None:
            if rdmol is None:
                R_pos.append(d.pos.float())    # 整列不能（degenerate: 従来動作）
            continue
        R_pos.append(d.pos.float()[mt])
    if len(R_pos) < 2:
        return None
    R_stack = _stack_pos(R_pos)                      # (nr, n, 3)

    # ── 生成 ──
    if args.mode == 'recon':
        gen_datas = [ds[i] for i in idxs]            # collate に消費される fresh Data
        gen_datas = [d for d in gen_datas if d is not None]
    else:
        n_gen = args.n_gen if args.n_gen > 0 else len(R_datas)
        # トポロジーのみ必要（pos は使わない）。group から巡回して fresh 取得。
        gen_datas = []
        k = 0
        while len(gen_datas) < n_gen:
            d = ds[indices[k % len(indices)]]
            k += 1
            if d is not None:
                gen_datas.append(d)
            if k > 4 * n_gen:   # 念のための無限ループガード
                break

    # 生成前に各 gen_data の topo フレームへの match を確保（collate が topology を
    # 破壊消費するため、生成呼び出しの前に読む）。
    gen_matches = [_match(d) for d in gen_datas]

    G_pos_raw = generate_positions(gen_datas, args.mode, cond_encoder, vae, flow,
                                   device, margs, args.batch_size, args.n_steps)

    # 生成座標を topo フレームへリマップ（マッチ不成立は除外）
    G_pos_all = []
    for p, mt in zip(G_pos_raw, gen_matches):
        if mt is None:
            if rdmol is None:
                G_pos_all.append(p)
            continue
        G_pos_all.append(p[mt])

    # ── 妥当性ゲート（G に適用、fail は分布/COV-MAT から除外）──
    G_valid = []
    for p in G_pos_all:
        pn = p.cpu().numpy()
        if validity_of_conformer(pn, rvdw, nonbond_mask, bi_np, bj_np, sanitize_ok):
            G_valid.append(p)
    n_gen_total = len(G_pos_all)
    n_fail = n_gen_total - len(G_valid)
    pass_rate = (len(G_valid) / n_gen_total) if n_gen_total > 0 else float('nan')

    res = {
        'uuid': uuid,
        'n_heavy': n_heavy,
        'n_atoms': n_atoms,
        'n_ref': len(R_pos),
        'n_gen': n_gen_total,
        'n_gen_valid': len(G_valid),
        'n_fail': n_fail,
        'validity_pass_rate': pass_rate,
        'sanitize_ok': sanitize_ok,
        'torsion_js': float('nan'),
        'torsion_w1': float('nan'),
        'tfd_median': float('nan'),
        'bond_js': float('nan'),
        'angle_js': float('nan'),
        'cov_r': float('nan'), 'cov_p': float('nan'),
        'mat_r': float('nan'), 'mat_p': float('nan'),
    }
    if len(G_valid) == 0:
        return res

    G_stack = _stack_pos(G_valid)                    # (ng, n, 3)

    # ── torsion 分布 JS（主指標）+ 円環 W1 ──
    R_tors = torsion_angles(R_stack, quartets)
    G_tors = torsion_angles(G_stack, quartets)
    res['torsion_js'] = per_element_js(R_tors, G_tors, bins=36, lo=-PI, hi=PI)
    res['torsion_w1'] = per_torsion_circular_w1(R_tors, G_tors, bins=36)

    # ── bond / angle 分布 JS（補助）──
    R_bond = bond_lengths(R_stack, bonds)
    G_bond = bond_lengths(G_stack, bonds)
    res['bond_js'] = per_element_js(R_bond, G_bond, bins=200, lo=0.9, hi=2.0)

    R_ang = angle_values(R_stack, triplets)
    G_ang = angle_values(G_stack, triplets)
    res['angle_js'] = per_element_js(R_ang, G_ang, bins=60, lo=0.0, hi=PI)

    # ── COV / MAT（重原子・Kabsch）──
    if n_heavy >= 2:
        hidx = torch.from_numpy(heavy_idx).long()
        R_heavy = R_stack[:, hidx, :].to(device)
        G_heavy = G_stack[:, hidx, :].to(device)
        res.update(cov_mat_metrics(R_heavy, G_heavy, args.delta))

    # ── TFD ──
    R_pos_np = [p.cpu().numpy() for p in R_pos]
    G_pos_np = [p.cpu().numpy() for p in G_valid]
    res['tfd_median'] = tfd_median(rdmol, R_pos_np, G_pos_np,
                                   max_r_tfd=min(20, len(R_pos_np)), rng=rng)
    return res


# ══════════════════════════════════════════════════════════════════════════════
# 集約・出力
# ══════════════════════════════════════════════════════════════════════════════

_METRIC_KEYS = ['torsion_js', 'torsion_w1', 'tfd_median', 'bond_js', 'angle_js',
                'cov_r', 'cov_p', 'mat_r', 'mat_p', 'validity_pass_rate']

_SIZE_BINS = [0, 30, 50, 70, 10_000]


def _agg(values: list) -> dict:
    """nan を除外して median / mean / p90 を計算。"""
    t = torch.tensor([v for v in values if v == v], dtype=torch.float64)
    if t.numel() == 0:
        return {'median': float('nan'), 'mean': float('nan'),
                'p90': float('nan'), 'n': 0}
    return {
        'median': torch.median(t).item(),
        'mean': t.mean().item(),
        'p90': torch.quantile(t, 0.90).item(),
        'n': int(t.numel()),
    }


def aggregate(per_uuid: list) -> dict:
    """uuid 代表値のリストを全体 / サイズ帯別に集約する。"""
    overall = {k: _agg([r[k] for r in per_uuid]) for k in _METRIC_KEYS}

    by_size = {}
    for i in range(len(_SIZE_BINS) - 1):
        lo, hi = _SIZE_BINS[i], _SIZE_BINS[i + 1]
        sub = [r for r in per_uuid if lo <= r['n_heavy'] < hi]
        if not sub:
            continue
        label = f'[{lo},{hi if hi < 10_000 else "+"})'
        by_size[label] = {'n_uuid': len(sub),
                          **{k: _agg([r[k] for r in sub]) for k in _METRIC_KEYS}}
    return {'overall': overall, 'by_size': by_size}


def _print_report(result: dict):
    agg = result['aggregate']
    print(f'\n{"=" * 74}')
    print(f'  アンサンブル分布評価  |  mode={result["mode"]}  |  {result["n_uuid"]} uuid')
    print(f'{"=" * 74}')
    print(f'  {"指標":<22} {"median":>10} {"mean":>10} {"p90":>10} {"n":>5}')
    print(f'  {"-" * 66}')
    labels = {
        'torsion_js': 'torsion-JS (主)', 'torsion_w1': 'torsion-W1(円環)',
        'tfd_median': 'TFD', 'bond_js': 'bond-JS', 'angle_js': 'angle-JS',
        'cov_r': 'COV-R', 'cov_p': 'COV-P', 'mat_r': 'MAT-R', 'mat_p': 'MAT-P',
        'validity_pass_rate': '妥当性pass率',
    }
    for k in _METRIC_KEYS:
        a = agg['overall'][k]
        print(f'  {labels[k]:<22} {a["median"]:>10.4f} {a["mean"]:>10.4f} '
              f'{a["p90"]:>10.4f} {a["n"]:>5}')

    print(f'\n  ── サイズ帯別（重原子数）中央値 ──')
    hdr = f'  {"帯":>10} {"n":>4}'
    for k in ('torsion_js', 'tfd_median', 'cov_r', 'cov_p', 'mat_r', 'mat_p'):
        hdr += f' {k:>9}'
    print(hdr)
    print('  ' + '-' * 70)
    for label, st in agg['by_size'].items():
        line = f'  {label:>10} {st["n_uuid"]:>4}'
        for k in ('torsion_js', 'tfd_median', 'cov_r', 'cov_p', 'mat_r', 'mat_p'):
            line += f' {st[k]["median"]:>9.4f}'
        print(line)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='同一uuidの生成/参照アンサンブルの分布評価')
    p.add_argument('--checkpoint', type=str, required=True, help='VAE チェックポイント')
    p.add_argument('--val_lmdb', type=str, required=True)
    p.add_argument('--mode', type=str, default='recon',
                   choices=['recon', 'prior', 'dit'])
    p.add_argument('--dit_checkpoint', type=str, default=None,
                   help='--mode dit のとき必須（DiT チェックポイント）')
    p.add_argument('--max_smiles', type=int, default=0, help='評価する uuid 数上限（0=全）')
    p.add_argument('--max_ref', type=int, default=100, help='参照配座のサブサンプル上限')
    p.add_argument('--n_gen', type=int, default=0, help='prior/dit の生成本数（0=参照数と同数）')
    p.add_argument('--delta', type=float, default=1.0, help='COV の閾値 [Å]')
    p.add_argument('--max_atoms', type=int, default=288)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=0)   # 直接インデックスアクセスのため 0
    p.add_argument('--n_steps', type=int, default=100, help='dit の ODE ステップ数')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', type=str, default=None, help='結果 JSON の保存先')
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if args.device == 'auto' else torch.device(args.device))

    if args.mode == 'dit' and not args.dit_checkpoint:
        print('--mode dit には --dit_checkpoint が必要です', file=sys.stderr)
        sys.exit(1)

    # ── モデルロード ──
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cond_encoder, vae, margs = _load_models(ckpt, device)

    flow = None
    if args.mode == 'dit':
        from poly3d.model.builder import build_dit
        from poly3d.model.flow_matching import FlowMatching
        dck = torch.load(args.dit_checkpoint, map_location=device, weights_only=False)
        dargs = argparse.Namespace(**dck['args'])
        dit = build_dit(dargs).to(device)
        dit.load_state_dict(dck['flow'])
        flow = FlowMatching(dit, t_max=dargs.t_max)
        flow.eval()

    # ── 参照アンサンブルへのグルーピング ──
    print(f'checkpoint : {args.checkpoint}')
    print(f'val_lmdb   : {args.val_lmdb}')
    print(f'device     : {device}   mode : {args.mode}')
    print('uuid グルーピング中...')
    groups = build_uuid_index(args.val_lmdb, args.max_smiles)
    print(f'対象 uuid 数: {len(groups)}')

    # dataset（直接インデックスアクセス。mds_init/topology は margs に追従）
    ds = ConformerDataset(
        args.val_lmdb, max_atoms=args.max_atoms, precompute_topology=True,
        mds_init=getattr(margs, 'mds_init', False),
        topology_cache_size=8192,
    )

    per_uuid = []
    for ui, (uuid, indices) in enumerate(groups.items()):
        r = evaluate_uuid(uuid, indices, ds, cond_encoder, vae, flow,
                          device, margs, args, rng)
        if r is None:
            continue
        per_uuid.append(r)
        print(f'  [{ui + 1}/{len(groups)}] {uuid[:12]}… '
              f'nheavy={r["n_heavy"]:>3} nref={r["n_ref"]:>3} '
              f'ngen={r["n_gen"]:>3}(valid {r["n_gen_valid"]:>3}) '
              f'torJS={r["torsion_js"]:.3f} COV-R={r["cov_r"]:.2f} '
              f'MAT-R={r["mat_r"]:.2f} pass={r["validity_pass_rate"]:.2f}')

    ds.close()

    if not per_uuid:
        print('評価可能な uuid がありませんでした')
        return

    result = {
        'checkpoint': str(args.checkpoint),
        'val_lmdb': str(args.val_lmdb),
        'mode': args.mode,
        'delta': args.delta,
        'max_ref': args.max_ref,
        'n_uuid': len(per_uuid),
        'aggregate': aggregate(per_uuid),
        'per_uuid': per_uuid,
    }
    _print_report(result)

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                  encoding='utf-8')
        print(f'結果を保存: {args.out}')


if __name__ == '__main__':
    main()
