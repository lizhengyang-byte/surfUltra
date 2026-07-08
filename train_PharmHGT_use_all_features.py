"""
Reproduce_PharmHGT_LogCMC.py — 完整复现 PharmHGT 论文方法用于表面活性剂 LogCMC 预测
(PyTorch Geometric 版本，适用于 Windows 环境)

依赖安装:
  pip install torch torch_geometric rdkit optuna scikit-learn numpy pandas

数据来源:
  GitHub: https://github.com/Graph-transformers-GCN-GAT/GCN-GAT-PharmaHGT
  Zenodo: https://doi.org/10.5281/zenodo.16129095

  训练集: data/surfpro_imputed.csv
  测试集: data/surfpro_test.csv
  需包含 "SMILES" 和 "pCMC" 列

论文公式编号:
  Eq.1  → Multi-head Attention
  Eq.2  → Multi-View Message Passing (MVMP)
  Eq.3-4 → Hierarchical Readout Attention
  Eq.5  → MLP Regression Head (LogCMC)
"""
import os, sys, re, copy, json, warnings, random, math
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys, BRICS

import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
RDLogger.logger().setLevel(RDLogger.ERROR)

# ─────────────────────────────────────────────
# 0. 全局设置 & 工具函数
# ─────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
# 1. 分子图构建 — RDKit → PyG HeteroData
# ─────────────────────────────────────────────

# ----- 1a. 原子特征（55 维，对应论文 atom_dim=55） -----
ATOM_FEATURES = {
    "atomic_num": list(range(1, 36)),
    "degree":     list(range(0, 8)),
    "formal_charge": list(range(-3, 4)),
    "hybridization": [
        Chem.HybridizationType.SP,
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3,
    ],
    "num_h":      list(range(0, 6)),
    "chirality":  [Chem.ChiralType.CHI_UNSPECIFIED, Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                   Chem.ChiralType.CHI_TETRAHEDRAL_CCW],
}


def one_hot(val, choices) -> List[int]:
    encoding = [0] * (len(choices) + 1)
    try:
        idx = choices.index(val)
        encoding[idx] = 1
    except ValueError:
        encoding[-1] = 1
    return encoding


def get_atom_features(atom: Chem.Atom) -> np.ndarray:
    """55 维原子特征向量。"""
    feats = []
    feats += one_hot(atom.GetAtomicNum(), ATOM_FEATURES["atomic_num"])      # 36
    feats += one_hot(atom.GetDegree(), ATOM_FEATURES["degree"])             # 9
    feats += one_hot(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])  # 8
    feats += one_hot(atom.GetHybridization(), ATOM_FEATURES["hybridization"]) # 4
    feats += one_hot(atom.GetTotalNumHs(), ATOM_FEATURES["num_h"])          # 7
    feats += one_hot(atom.GetChiralTag(), ATOM_FEATURES["chirality"])       # 3+1=4
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)                       # 1
    feats.append(1.0 if atom.IsInRing() else 0.0)                            # 1
    feats.append(1.0 if atom.IsInRingSize(3) else 0.0)
    feats.append(1.0 if atom.IsInRingSize(4) else 0.0)
    feats.append(1.0 if atom.IsInRingSize(5) else 0.0)
    feats.append(1.0 if atom.IsInRingSize(6) else 0.0)
    feats.append(1.0 if atom.IsInRingSize(7) else 0.0)                       # +5 = total ~76
    # 归一化质量
    mass = float(atom.GetMass())
    feats.append(min(mass / 200.0, 1.0))
    # 范德华半径
    vdw_radii = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 16: 1.80, 17: 1.75, 35: 1.85}
    vdw = vdw_radii.get(atom.GetAtomicNum(), 1.50)
    feats.append(min(vdw / 2.0, 1.0))

    # 裁剪或填充至 55 维
    if len(feats) < 55:
        feats += [0.0] * (55 - len(feats))
    elif len(feats) > 55:
        feats = feats[:55]
    return np.array(feats, dtype=np.float32)


# ----- 1b. 键特征（14 维） -----
BOND_TYPES = [
    Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE, Chem.BondType.AROMATIC,
]


def get_bond_features(bond: Chem.Bond) -> np.ndarray:
    """14 维键特征向量。"""
    btype = bond.GetBondType()
    feats = one_hot(btype, BOND_TYPES)                                     # 5
    feats.append(1.0 if bond.GetIsConjugated() else 0.0)                   # 1
    feats.append(1.0 if bond.IsInRing() else 0.0)                          # 1
    stereo_chem = bond.GetStereo()
    feats.append(1.0 if stereo_chem != Chem.BondStereo.STEREONONE else 0.0)  # 1
    stereo_map = {
        Chem.BondStereo.STEREONONE: [1, 0, 0, 0],
        Chem.BondStereo.STEREOANY:  [0, 1, 0, 0],
        Chem.BondStereo.STEREOZ:    [0, 0, 1, 0],
        Chem.BondStereo.STEREOE:    [0, 0, 0, 1],
    }
    feats += stereo_map.get(stereo_chem, [0, 0, 0, 1])       # 4
    # 额外：是否在环中 (第二个指示符)
    feats.append(1.0 if bond.IsInRing() else 0.0)                          # +1 = 13
    # 确保 14 维
    if len(feats) < 14:
        feats += [0.0] * (14 - len(feats))
    return np.array(feats[:14], dtype=np.float32)


# ----- 1c. MACCS 药效团特征（194 维）-----
def get_maccs_features(mol: Chem.Mol) -> np.ndarray:
    maccs = MACCSkeys.GenMACCSKeys(mol)
    bits = [int(b) for b in maccs.ToBitString()]
    return np.array(bits, dtype=np.float32)


# ----- 1d. BRICS 反应特征（34 维）-----
def get_brics_features(mol: Chem.Mol) -> np.ndarray:
    try:
        bonds = list(BRICS.FindBRICSBonds(mol))
    except Exception:
        bonds = []
    bits = [0] * 34
    for (a1, a2), _ in bonds:
        idx = (a1 + a2) % 34
        bits[idx] = 1
    return np.array(bits, dtype=np.float32)


# ----- 1e. 表面活性剂检测 (Section 2.1.2) -----

HEAD_ANIONIC_SMARTS = [
    "[O-]S(=O)(=O)",    # 磺酸根 -SO3-
    "[O-]C(=O)",        # 羧酸根 -COO-
    "[O-]S(=O)(=O)O",   # 硫酸酯基 -OSO3-
    "[O-]P(=O)(O)O",    # 磷酸根 -OPO3-
]
HEAD_CATIONIC_SMARTS = [
    "[N+X3](C)(C)C",    # 季铵 -NR3+
    "[N+X2]=C",         # 吡啶 / 亚胺
    "[N+](C)(C)C",      # 取代铵
]
HEAD_NONIONIC_SMARTS = [
    "[OH]",             # 羟基
    "[O]C",             # 醚键
    "C(=O)N",           # 酰胺
]
MIN_TAIL_LENGTH = 4


def _match_smarts(mol: Chem.Mol, smarts_list: List[str]) -> set:
    matched = set()
    for sma in smarts_list:
        pat = Chem.MolFromSmarts(sma)
        if pat is None:
            continue
        for m in mol.GetSubstructMatches(pat):
            matched.update(m)
    return matched


def _dfs_longest_carbon_chain(mol: Chem.Mol) -> List[int]:
    """DFS 找最长连续碳链（≥4 碳）。"""
    adj = defaultdict(list)
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if mol.GetAtomWithIdx(a1).GetAtomicNum() == 6 and mol.GetAtomWithIdx(a2).GetAtomicNum() == 6:
            adj[a1].append(a2)
            adj[a2].append(a1)
    best_path = []
    visited_global = set()
    for start in adj:
        if start in visited_global:
            continue
        stack = [(start, [start], {start})]
        while stack:
            node, path, vis = stack.pop()
            if len(path) > len(best_path):
                best_path = path[:]
            for nb in adj[node]:
                if nb not in vis:
                    stack.append((nb, path + [nb], vis | {nb}))
        visited_global.update(best_path)
    return best_path if len(best_path) >= MIN_TAIL_LENGTH else []


def detect_surfactant(mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray, bool]:
    """检测表面活性剂头基/尾基，返回 atom-level mask。"""
    N = mol.GetNumAtoms()
    head_mask = np.zeros(N, dtype=bool)
    tail_mask = np.zeros(N, dtype=bool)

    anionic = _match_smarts(mol, HEAD_ANIONIC_SMARTS)
    cationic = _match_smarts(mol, HEAD_CATIONIC_SMARTS)
    nonionic = _match_smarts(mol, HEAD_NONIONIC_SMARTS)

    if anionic:
        head_mask[list(anionic)] = True
    elif cationic:
        head_mask[list(cationic)] = True
    elif nonionic:
        head_mask[list(nonionic)] = True
    amphoteric = anionic & cationic
    if amphoteric:
        head_mask[list(amphoteric)] = True

    tail_indices = _dfs_longest_carbon_chain(mol)
    if tail_indices:
        tail_mask[tail_indices] = True

    overlap = head_mask & tail_mask
    tail_mask[overlap] = False
    is_surf = head_mask.any() or tail_mask.any()
    return head_mask, tail_mask, is_surf


# ----- 1f. PyG HeteroData 构建 -----

def mol_to_heterodata(smiles: str) -> Optional[Dict]:
    """SMILES → PyG HeteroData + 特征字典。

    Returns dict with keys:
        'data':       HeteroData 对象 (三视图: atom, pharm, brics)
        'atom_feat':  (N, 55) tensor
        'bond_feat':  (E, 14) tensor
        'pharm_feat': (P, 194) tensor
        'brics_feat': (R, 34) tensor
        'head_mask':  (N,) bool tensor
        'tail_mask':  (N,) bool tensor
        'n_surfactant': bool
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    num_atoms = mol.GetNumAtoms()

    # -- 原子节点特征 --
    atom_feats = [get_atom_features(a) for a in mol.GetAtoms()]
    atom_feat = torch.tensor(np.stack(atom_feats, axis=0), dtype=torch.float32)

    # -- 键（边）--
    src, dst, bond_feats_list = [], [], []
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src.append(a1); dst.append(a2)
        src.append(a2); dst.append(a1)  # 无向图
        bf = get_bond_features(bond)
        bond_feats_list.append(bf)
        bond_feats_list.append(bf)
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    bond_feat = torch.tensor(np.stack(bond_feats_list, axis=0), dtype=torch.float32) if bond_feats_list else torch.zeros((0, 14), dtype=torch.float32)

    # -- MACCS 药效团节点 (Gβ) --
    maccs_bits = get_maccs_features(mol)
    pharm_active = np.where(maccs_bits > 0)[0]
    if len(pharm_active) == 0:
        pharm_active = np.array([0])
    n_pharm = len(pharm_active)
    pharm_feat = torch.zeros(n_pharm, 194, dtype=torch.float32)
    for i, idx in enumerate(pharm_active):
        pharm_feat[i, idx] = 1.0

    # -- BRICS 反应节点 (Gγ) --
    brics_bits = get_brics_features(mol)
    brics_active = np.where(brics_bits > 0)[0]
    if len(brics_active) == 0:
        brics_active = np.array([0])
    n_brics = len(brics_active)
    brics_feat = torch.zeros(n_brics, 34, dtype=torch.float32)
    for i, idx in enumerate(brics_active):
        brics_feat[i, idx] = 1.0

    # -- 构建 PyG HeteroData --
    data = HeteroData()
    data['atom'].x = atom_feat
    data['pharm'].x = pharm_feat
    data['brics'].x = brics_feat

    # 边: atom <-> atom
    if edge_index.size(1) > 0:
        data['atom', 'bond', 'atom'].edge_index = edge_index
        data['atom', 'bond', 'atom'].edge_attr = bond_feat

    # 边: atom <-> pharm (全连接二分图)
    atom_ids = torch.arange(num_atoms, dtype=torch.long)
    pharm_ids = torch.arange(n_pharm, dtype=torch.long)
    ap_src = atom_ids.repeat_interleave(n_pharm)
    ap_dst = pharm_ids.repeat(num_atoms)
    data['atom', 'to_pharm', 'pharm'].edge_index = torch.stack([ap_src, ap_dst], dim=0)
    data['pharm', 'from_pharm', 'atom'].edge_index = torch.stack([ap_dst, ap_src], dim=0)

    # 边: atom <-> brics
    brics_ids = torch.arange(n_brics, dtype=torch.long)
    ab_src = atom_ids.repeat_interleave(n_brics)
    ab_dst = brics_ids.repeat(num_atoms)
    data['atom', 'to_brics', 'brics'].edge_index = torch.stack([ab_src, ab_dst], dim=0)
    data['brics', 'from_brics', 'atom'].edge_index = torch.stack([ab_dst, ab_src], dim=0)

    # -- 表面活性剂检测 --
    head_mask, tail_mask, is_surf = detect_surfactant(mol)

    return {
        "data": data,
        "atom_feat": atom_feat,
        "bond_feat": bond_feat,
        "pharm_feat": pharm_feat,
        "brics_feat": brics_feat,
        "head_mask": torch.tensor(head_mask, dtype=torch.bool),
        "tail_mask": torch.tensor(tail_mask, dtype=torch.bool),
        "n_surfactant": is_surf,
        "num_atoms": num_atoms,
    }


# ─────────────────────────────────────────────
# 2. PharmHGT 模型组件 (Section 2.1)
# ─────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """Eq.1: Multi-head Attention (per view).

    Attention(Q, K, V) = sum_p Ω_p · softmax(Q_p K_p^T / √d_k) · V_p
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.q_lin = nn.Linear(hidden_dim, hidden_dim)
        self.k_lin = nn.Linear(hidden_dim, hidden_dim)
        self.v_lin = nn.Linear(hidden_dim, hidden_dim)
        self.out_lin = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        N, D = x.shape
        Q = self.q_lin(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        K = self.k_lin(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.v_lin(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(0, 1).contiguous().view(N, D)
        return self.out_lin(out)


class PharmGNNLayer(nn.Module):
    """单层 PharmHGT: 视图内自注意力 + 跨视图融合 MVMP (Eq.2)。"""
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_view, x_cross=None):
        # 视图内自注意力
        attn_out = self.self_attn(x_view)
        x = self.norm1(x_view + self.dropout(attn_out))
        # 跨视图融合 MVMP
        if x_cross is not None:
            cross_out = self.cross_attn(x, x_cross)
            x = self.norm2(x + self.dropout(cross_out))
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


class PharmGCNLayer(nn.Module):
    """基于 GCN 的消息传递层 (PyG)，用于异构图中视图内传播。"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.conv = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        return F.relu(self.conv(x, edge_index))


class PharmHGTModel(nn.Module):
    """PharmHGT 异构图表征模型 (Section 2.1)。

    三视图: Gα (atom), Gβ (pharm/MACCS), Gγ (brics/BRICS)
    + 表面活性剂检测增强 (Section 2.1.2)
    """
    def __init__(
        self,
        atom_dim: int = 55,
        pharm_dim: int = 194,
        brics_dim: int = 34,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        use_surfactant: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_surfactant = use_surfactant

        # ---- 视图嵌入 ----
        self.atom_embed = nn.Linear(atom_dim, hidden_dim)
        self.pharm_embed = nn.Linear(pharm_dim, hidden_dim)
        self.brics_embed = nn.Linear(brics_dim, hidden_dim)

        # ---- 表面活性剂检测嵌入 (Section 2.1.2) ----
        self.head_embed = nn.Embedding(2, hidden_dim // 4)
        self.tail_embed = nn.Embedding(2, hidden_dim // 4)
        self.surf_fusion = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)

        # ---- GCN 传播层 (跨异构边) ----
        self.gcn_atom = nn.ModuleList([
            PharmGCNLayer(hidden_dim) for _ in range(num_layers)
        ])
        self.gcn_pharm = nn.ModuleList([
            PharmGCNLayer(hidden_dim) for _ in range(num_layers)
        ])
        self.gcn_brics = nn.ModuleList([
            PharmGCNLayer(hidden_dim) for _ in range(num_layers)
        ])

        # ---- 跨视图融合边 (atom↔pharm, atom↔brics) ----
        self.cross_pharm = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.cross_brics = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # ---- 注意力 Transformer 层 (Eq.1) ----
        self.atom_transformer = nn.ModuleList([
            PharmGNNLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.pharm_transformer = nn.ModuleList([
            PharmGNNLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.brics_transformer = nn.ModuleList([
            PharmGNNLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])

        # ---- 层次 Readout 注意力 (Eq.3-4) ----
        self.readout_attn = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)

        # ---- 回归头 MLP (Eq.5) ----
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout // 2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data_dict: Dict):
        """data_dict: mol_to_heterodata 的输出。"""
        dev = next(self.parameters()).device
        data = data_dict["data"].to(dev)
        atom_x = data['atom'].x
        pharm_x = data['pharm'].x
        brics_x = data['brics'].x

        # 边索引
        bond_ei = data['atom', 'bond', 'atom'].edge_index if ('atom', 'bond', 'atom') in data.edge_types else None
        pharm_ei = data['atom', 'to_pharm', 'pharm'].edge_index if ('atom', 'to_pharm', 'pharm') in data.edge_types else None
        brics_ei = data['atom', 'to_brics', 'brics'].edge_index if ('atom', 'to_brics', 'brics') in data.edge_types else None

        # ---- 嵌入 ----
        h_atom = self.atom_embed(atom_x)
        h_pharm = self.pharm_embed(pharm_x)
        h_brics = self.brics_embed(brics_x)

        # ---- 表面活性剂增强 (Section 2.1.2) ----
        if self.use_surfactant and data_dict.get("n_surfactant", False):
            head_mask = data_dict["head_mask"].to(dev)
            tail_mask = data_dict["tail_mask"].to(dev)
            if head_mask.size(0) == h_atom.size(0):
                head_emb = self.head_embed(head_mask.long())
                tail_emb = self.tail_embed(tail_mask.long())
                surf_feat = torch.cat([head_emb, tail_emb], dim=-1)
                h_atom = self.surf_fusion(torch.cat([h_atom, surf_feat], dim=-1))

        # ---- 逐层消息传递 (MVMP, Eq.2) ----
        for layer in range(len(self.atom_transformer)):
            # --- GCN 消息传递 (视图内) ---
            if bond_ei is not None and bond_ei.size(1) > 0:
                h_atom = self.gcn_atom[layer](h_atom, bond_ei)
            if pharm_ei is not None and pharm_ei.size(1) > 0:
                pharm_msg = self.cross_pharm[layer](h_atom).mean(dim=0, keepdim=True)
                h_pharm = h_pharm + pharm_msg.expand(h_pharm.size(0), -1)
            if brics_ei is not None and brics_ei.size(1) > 0:
                brics_msg = self.cross_brics[layer](h_atom).mean(dim=0, keepdim=True)
                h_brics = h_brics + brics_msg.expand(h_brics.size(0), -1)

            # --- Transformer (Eq.1 + 跨视图融合) ---
            # 跨视图上下文: pharm/brics 特征平均后广播
            if h_pharm.size(0) > 0:
                pharm_ctx = h_pharm.mean(dim=0, keepdim=True).expand(h_atom.size(0), -1)
            else:
                pharm_ctx = torch.zeros_like(h_atom)
            if h_brics.size(0) > 0:
                brics_ctx = h_brics.mean(dim=0, keepdim=True).expand(h_atom.size(0), -1)
            else:
                brics_ctx = torch.zeros_like(h_atom)

            h_atom = self.atom_transformer[layer](h_atom, pharm_ctx + brics_ctx)
            h_pharm = self.pharm_transformer[layer](h_pharm,
                                                     h_atom[:h_pharm.size(0)] if h_pharm.size(0) <= h_atom.size(0) else h_pharm)
            h_brics = self.brics_transformer[layer](h_brics,
                                                     h_atom[:h_brics.size(0)] if h_brics.size(0) <= h_atom.size(0) else h_brics)

        # ---- 层次 Readout 注意力 (Eq.3-4) ----
        def attentive_readout(h, query):
            # query: (1, 1, D), h: (N, D)
            ctx, _ = self.readout_attn(query.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0))
            return ctx.squeeze(0).squeeze(0)  # (D,)

        atom_q = h_atom.mean(dim=0, keepdim=True)
        pharm_q = h_pharm.mean(dim=0, keepdim=True)
        brics_q = h_brics.mean(dim=0, keepdim=True)

        atom_rep = attentive_readout(h_atom, atom_q)
        pharm_rep = attentive_readout(h_pharm, pharm_q)
        brics_rep = attentive_readout(h_brics, brics_q)

        graph_rep = torch.cat([atom_rep, pharm_rep, brics_rep], dim=-1)  # (3D,)

        # ---- MLP 回归头 (Eq.5) ----
        return self.reg_head(graph_rep).squeeze(-1)


# ─────────────────────────────────────────────
# 3. 数据加载 & 划分
# ─────────────────────────────────────────────

def load_data_imputed(
    train_csv: str = "data/surfpro_imputed.csv",
    test_csv: str = "data/surfpro_test.csv",
    target_col: str = "pCMC",
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    """加载 imputed 训练集和测试集（与 002.py 格式一致）。"""
    df_train = pd.read_csv(train_csv).dropna(subset=[target_col])
    df_test  = pd.read_csv(test_csv).dropna(subset=[target_col])
    train_smiles = df_train["SMILES"].tolist()
    train_y = df_train[target_col].values.astype(np.float32)
    test_smiles = df_test["SMILES"].tolist()
    test_y = df_test[target_col].values.astype(np.float32)
    print(f"训练集: {len(train_smiles)} 条 (来自 {train_csv})")
    print(f"测试集: {len(test_smiles)} 条 (来自 {test_csv})")
    return train_smiles, train_y, test_smiles, test_y


def split_train_val(smiles_list: List[str], y: np.ndarray,
                    val_ratio: float = 0.1, random_state: int = 42):
    """从训练集中划分验证集。"""
    train_smi, val_smi, train_y, val_y = train_test_split(
        smiles_list, y, test_size=val_ratio, random_state=random_state
    )
    print(f"划分: 训练 {len(train_smi)}, 验证 {len(val_smi)}")
    return train_smi, val_smi, train_y, val_y


# ─────────────────────────────────────────────
# 4. 训练 & 评估
# ─────────────────────────────────────────────

def train_epoch(model, optimizer, data_list, labels, batch_size=32):
    """训练一个 epoch（逐图处理，支持异构批量的简单实现）。"""
    model.train()
    total_loss = 0.0
    n = len(data_list)
    indices = list(range(n))
    random.shuffle(indices)

    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_y = torch.tensor([labels[i] for i in batch_idx], dtype=torch.float32, device=DEVICE)

        optimizer.zero_grad()
        loss = 0.0
        for j, idx_i in enumerate(batch_idx):
            pred = model(data_list[idx_i])
            loss += F.mse_loss(pred, batch_y[j])
        loss = loss / len(batch_idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * len(batch_idx)

    return total_loss / n


@torch.no_grad()
def evaluate(model, data_list, labels):
    """评估模型，返回 MSE, MAE, R²。"""
    model.eval()
    preds, targets = [], []
    for data_i, y_i in zip(data_list, labels):
        pred = model(data_i.to(DEVICE)).cpu().item()
        preds.append(pred)
        targets.append(y_i)
    preds, targets = np.array(preds), np.array(targets)
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    return mse, mae, r2


# ─────────────────────────────────────────────
# 5. Optuna 超参数搜索 (Section 3.4)
# ─────────────────────────────────────────────

PHARMHGT_PARAMS = {
    "hidden_dim": {"type": "int", "low": 128, "high": 512, "step": 64},
    "num_layers": {"type": "int", "low": 2, "high": 6},
    "dropout":    {"type": "float", "low": 0.1, "high": 0.5},
    "batch_size": {"type": "int", "low": 16, "high": 128, "step": 16},
    "lr_init":    {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
    "lr_max":     {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
    "lr_final":   {"type": "float", "low": 1e-6, "high": 1e-4, "log": True},
    "seed":       {"type": "categorical", "choices": [42, 104, 2024, 4592]},
    "num_heads":  {"type": "int", "low": 2, "high": 8, "step": 2},
}


def sample_pharmhgt_params(trial: optuna.Trial) -> Dict[str, Any]:
    params = {}
    for name, spec in PHARMHGT_PARAMS.items():
        if spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"],
                                             step=spec.get("step", 1))
        elif spec["type"] == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"],
                                               log=spec.get("log", False))
        elif spec["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
    return params


def objective(trial, train_data, val_data, train_y, val_y):
    """Optuna objective: 最小化验证 MSE。"""
    params = sample_pharmhgt_params(trial)
    set_seed(params["seed"])

    model = PharmHGTModel(
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        num_heads=params["num_heads"],
        use_surfactant=True,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr_init"])

    patience = 30
    best_val_mse = float("inf")
    wait = 0

    for epoch in range(500):
        train_epoch(model, optimizer, train_data, train_y, params["batch_size"])
        val_mse, val_mae, val_r2 = evaluate(model, val_data, val_y)

        # OneCycle warmup (前 5 epoch 线性升 lr)
        if epoch < 5:
            lr = params["lr_init"] + (params["lr_max"] - params["lr_init"]) * (epoch + 1) / 5
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        elif epoch == 5:
            for pg in optimizer.param_groups:
                pg["lr"] = params["lr_max"]

        trial.report(val_mse, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    return best_val_mse


# ─────────────────────────────────────────────
# 6. 数据预处理缓存
# ─────────────────────────────────────────────

def preprocess_all(smiles_list: List[str]) -> List[Dict]:
    """批量将 SMILES 转为 HeteroData 字典。"""
    graphs = []
    for smi in tqdm(smiles_list, desc="Preprocessing SMILES"):
        g = mol_to_heterodata(smi)
        if g is not None:
            graphs.append(g)
    print(f"成功处理 {len(graphs)}/{len(smiles_list)} 分子")
    return graphs


# ─────────────────────────────────────────────
# 7. 主函数
# ─────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PharmHGT for LogCMC prediction (PyG)")
    parser.add_argument("--train_csv", type=str, default="data/surfpro_imputed.csv",
                        help="训练集 CSV")
    parser.add_argument("--test_csv", type=str, default="data/surfpro_test.csv",
                        help="测试集 CSV")
    parser.add_argument("--target", type=str, default="pCMC",
                        help="目标列名 (default: pCMC)")
    parser.add_argument("--trials", type=int, default=30,
                        help="Optuna trials")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--output", type=str, default="pharmhgt_best.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_surfactant", action="store_true",
                        help="禁用表面活性剂检测模块")
    parser.add_argument("--quick_test", action="store_true",
                        help="快速测试模式（小数据量验证）")
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"设备: {DEVICE}")
    print(f"PyTorch {torch.__version__}, PyG 2.x, RDKit 202x")

    # ---- 加载数据 ----
    train_smi, train_y, test_smi, test_y = load_data_imputed(
        args.train_csv, args.test_csv, args.target
    )
    if args.quick_test:
        train_smi = train_smi[:50]
        train_y = train_y[:50]
        test_smi = test_smi[:20]
        test_y = test_y[:20]
        print("快速测试模式: 50 训练 + 20 测试")

    # ---- 划分验证集 ----
    train_smi, val_smi, train_y, val_y = split_train_val(
        train_smi, train_y, val_ratio=0.1, random_state=args.seed
    )

    # ---- 预处理 ----
    print("\n预处理训练集...")
    train_data = preprocess_all(train_smi)
    print("预处理验证集...")
    val_data = preprocess_all(val_smi)
    print("预处理测试集...")
    test_data = preprocess_all(test_smi)

    train_y = np.array([train_y[i] for i in range(len(train_data))])
    val_y = np.array([val_y[i] for i in range(len(val_data))])
    test_y = np.array([test_y[i] for i in range(len(test_data))])

    if len(train_data) == 0:
        print("错误: 没有成功处理的分子！")
        return

    # ---- Optuna 超参数搜索 ----
    print("\n" + "=" * 60)
    print(f"Optuna 超参数搜索 ({args.trials} trials) ...")
    print("=" * 60)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    study.optimize(
        lambda trial: objective(trial, train_data, val_data, train_y, val_y),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    print(f"\n最佳 Trial: {study.best_trial.number}")
    print(f"最佳验证 MSE: {study.best_value:.6f}")
    print(f"最佳超参数: {study.best_params}")

    # ---- 最终训练 ----
    print("\n" + "=" * 60)
    print("用最佳参数训练最终模型 ...")
    print("=" * 60)

    best_params = study.best_params
    set_seed(best_params.get("seed", args.seed))

    final_model = PharmHGTModel(
        hidden_dim=best_params["hidden_dim"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
        num_heads=best_params.get("num_heads", 4),
        use_surfactant=not args.no_surfactant,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params["lr_init"])

    best_mse = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, args.epochs + 1):
        train_mse = train_epoch(final_model, optimizer, train_data, train_y,
                                best_params["batch_size"])
        val_mse, val_mae, val_r2 = evaluate(final_model, val_data, val_y)

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: Train MSE={train_mse:.4f}, "
                  f"Val MSE={val_mse:.4f}, Val MAE={val_mae:.4f}, Val R²={val_r2:.4f}")

        if val_mse < best_mse:
            best_mse = val_mse
            best_state = copy.deepcopy(final_model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        final_model.load_state_dict(best_state)

    # ---- 最终测试 ----
    print("\n" + "=" * 60)
    print("最终测试集评估")
    print("=" * 60)
    test_mse, test_mae, test_r2 = evaluate(final_model, test_data, test_y)
    print(f"  Test MSE : {test_mse:.6f}")
    print(f"  Test MAE : {test_mae:.6f}")
    print(f"  Test R²  : {test_r2:.4f}")

    torch.save({
        "model_state": final_model.state_dict(),
        "best_params": best_params,
        "metrics": {"test_mse": test_mse, "test_mae": test_mae, "test_r2": test_r2},
    }, args.output)
    print(f"\n模型已保存到 {args.output}")


if __name__ == "__main__":
    main()
