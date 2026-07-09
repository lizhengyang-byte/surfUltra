"""
pharmhgt_logcmc.py — PharmHGT: Pharmacophoric-constrained Heterogeneous Graph Transformer
for LogCMC (pCMC) prediction of surfactant molecules.

Based on: "Harnessing Graph Learning for Surfactant Chemistry: PharmHGT, GCN, and GAT in LogCMC Prediction"

Key references (paper sections):
  - 2.1.1 Model Architecture
  - 2.1.2 Surfactant-Specific Adaptations
  - 2.1.3 Implementation Details
  - 3.4 Hyperparameter Optimization

Architecture summary:
  - Atom-level view Gα: nodes=atoms (55-dim), edges=bonds (14-dim)
  - Pharmacophore view Gβ: MACCS keys (194-dim) + BRICS (34-dim)
  - Junction view Gγ: combined heterograph
  - Multi-view attention (Eq.1), MVMP (Eq.2), Hierarchical Readout (Eq.3-4), MLP (Eq.5)

Dependencies:
  pip install torch torch-geometric rdkit optuna scikit-learn pandas

Usage:
  python pharmhgt_logcmc.py

Data expected at:
  ./data/surfpro_imputed.csv  (training, columns: SMILES, types, pCMC, ...)
  ./data/surfpro_test.csv     (test, columns: SMILES, type, pCMC, ...)
"""

import os, sys, math, random, warnings
from copy import deepcopy
from collections import defaultdict

# Suppress RDKit warnings immediately
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

import numpy as np
import pandas as pd

# RDKit
from rdkit import Chem
from rdkit.Chem import (
    rdMolDescriptors, Descriptors, AllChem, MACCSkeys, BRICS, Crippen, rdchem
)
from rdkit.Chem.rdchem import BondType as BT, HybridizationType

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# PyTorch Geometric
import torch_geometric as pyg
from torch_geometric.data import Data, HeteroData, Batch
from torch_geometric.nn import (
    GCNConv, Linear, MessagePassing,
    global_mean_pool, global_add_pool,
)
from torch_geometric.utils import scatter
from torch_geometric.loader import DataLoader as PyGDataLoader

# Optuna
import optuna
from optuna.pruners import MedianPruner

# scikit-learn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ===========================================================================
# Device & Constants
# ===========================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Base dimensions from paper Section 2.1.3
ATOM_FEAT_DIM = 55
BOND_FEAT_DIM = 14
PHARM_FEAT_DIM = 194   # MACCS keys
REACT_FEAT_DIM = 34    # BRICS bond types

# ===========================================================================
# 1. Feature Extraction — Atom-level (55-dim) & Bond-level (14-dim)
# ===========================================================================

_ATOM_TYPES = [1, 3, 5, 6, 7, 8, 9, 11, 14, 15, 16, 17, 19, 35, 53, 79]
_ATOM_TYPE_TO_IDX = {at: i for i, at in enumerate(_ATOM_TYPES)}
_NUM_ATOM_TYPES = len(_ATOM_TYPES)  # 16

_HYBRIDIZATION_TYPES = [
    HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3,
    HybridizationType.SP3D, HybridizationType.SP3D2
]
_HYB_TO_IDX = {h: i for i, h in enumerate(_HYBRIDIZATION_TYPES)}


def get_atom_features(atom: Chem.Atom) -> np.ndarray:
    """55-dim atom feature vector.

    [0:16]    Atom type one-hot (16 elements)
    [16:22]   Degree one-hot (0-5+)
    [22]      Formal charge (normalized [-2,2]→[-1,1])
    [23:28]   Implicit Hs one-hot (0-4+)
    [28:33]   Hybridization one-hot
    [33]      Is aromatic
    [34]      In ring
    [35]      Mass / 100
    [36]      Chiral center
    [37]      Radical electrons / 2
    [38:42]   Explicit valence one-hot (1-5+)
    [42:46]   Ring size one-hot (3,4,5,6)
    [46:50]   Gasteiger charge bins
    [50]      Ring >= 7
    [51]      N or O
    [52]      H donor
    [53]      H acceptor
    [54]      Heavy neighbors / 4
    """
    feat = np.zeros(55, dtype=np.float32)
    mol = atom.GetOwningMol()

    atomic_num = atom.GetAtomicNum()
    type_idx = _ATOM_TYPE_TO_IDX.get(atomic_num, len(_ATOM_TYPES) - 1)
    feat[type_idx] = 1.0

    deg = min(atom.GetDegree(), 5)
    feat[16 + deg] = 1.0

    fc = np.clip(atom.GetFormalCharge(), -2, 2) / 2.0
    feat[22] = fc

    imp_h = min(atom.GetTotalNumHs(includeNeighbors=False), 4)
    feat[23 + imp_h] = 1.0

    hyb = atom.GetHybridization()
    feat[28 + _HYB_TO_IDX.get(hyb, 0)] = 1.0

    feat[33] = 1.0 if atom.GetIsAromatic() else 0.0
    feat[34] = 1.0 if atom.IsInRing() else 0.0
    feat[35] = atom.GetMass() / 100.0

    if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
        feat[36] = 1.0

    feat[37] = min(atom.GetNumRadicalElectrons(), 2) / 2.0

    val = atom.GetExplicitValence()
    if 1 <= val <= 4:
        feat[37 + val] = 1.0
    elif val >= 5:
        feat[41] = 1.0

    if atom.IsInRing():
        for ring in mol.GetRingInfo().AtomRings():
            if atom.GetIdx() in ring:
                sz = len(ring)
                if 3 <= sz <= 6:
                    feat[41 + sz - 2] = 1.0
                if sz >= 7:
                    feat[50] = 1.0
                break

    try:
        gc = float(atom.GetDoubleProp('_GasteigerCharge'))
        if np.isfinite(gc):
            gc = np.clip(gc, -1.0, 1.0)
            feat[46 + min(int((gc + 1.0) / 0.5), 3)] = 1.0
    except (KeyError, ValueError):
        pass

    feat[51] = 1.0 if atomic_num in (7, 8) else 0.0
    if atomic_num in (7, 8) and atom.GetTotalNumHs() > 0:
        feat[52] = 1.0
    feat[53] = 1.0 if atomic_num in (7, 8) else 0.0
    feat[54] = atom.GetDegree() / 4.0

    return feat


_BOND_TYPE_MAP = {
    BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3,
}
_BOND_STEREO_MAP = {
    Chem.rdchem.BondStereo.STEREONONE: 0,
    Chem.rdchem.BondStereo.STEREOANY: 1,
    Chem.rdchem.BondStereo.STEREOZ: 2,
    Chem.rdchem.BondStereo.STEREOE: 3,
    Chem.rdchem.BondStereo.STEREOCIS: 4,
    Chem.rdchem.BondStereo.STEREOTRANS: 5,
}


def get_bond_features(bond: Chem.Bond) -> np.ndarray:
    """14-dim bond feature vector.

    [0:4]   Bond type
    [4]     Conjugated
    [5]     In ring
    [6:12]  Stereo
    [12]    Aromatic
    [13]    In ring
    """
    feat = np.zeros(14, dtype=np.float32)
    feat[_BOND_TYPE_MAP.get(bond.GetBondType(), 0)] = 1.0
    feat[4] = 1.0 if bond.GetIsConjugated() else 0.0
    feat[5] = 1.0 if bond.IsInRing() else 0.0
    feat[6 + _BOND_STEREO_MAP.get(bond.GetStereo(), 0)] = 1.0
    feat[12] = 1.0 if bond.GetBondType() == BT.AROMATIC else 0.0
    feat[13] = 1.0 if bond.IsInRing() else 0.0
    return feat


# ===========================================================================
# 2. Pharmacophore & Reaction Features (Gβ)
# ===========================================================================

def get_pharmacophore_features(mol: Chem.Mol) -> np.ndarray:
    """194-dim: MACCS keys padded to 194."""
    feat = np.zeros(PHARM_FEAT_DIM, dtype=np.float32)
    try:
        maccs = MACCSkeys.GenMACCSKeys(mol)
        bits = np.array(list(maccs.ToBitString()), dtype=np.float32)
        n = min(len(bits), PHARM_FEAT_DIM)
        feat[:n] = bits[:n]
    except Exception:
        pass
    return feat


def get_reaction_features(mol: Chem.Mol) -> np.ndarray:
    """34-dim: BRICS fragment type histogram.

    Falls back gracefully if BRICS decomposition fails or hangs.
    """
    feat = np.zeros(REACT_FEAT_DIM, dtype=np.float32)
    try:
        # Quick pre-check: molecule must have rotatable bonds for BRICS
        n_rot = Descriptors.NumRotatableBonds(mol)
        if n_rot < 1:
            return feat
        # BRICS decomposition
        frags_gen = BRICS.BRICSDecompose(mol, returnMols=False)
        frags = []
        for i, f in enumerate(frags_gen):
            if i >= 128:  # safety limit
                break
            frags.append(f)
        if frags:
            for f in frags:
                feat[abs(hash(f)) % REACT_FEAT_DIM] += 1.0
            feat = feat / max(len(frags), 1)
    except Exception:
        pass
    return feat


# ===========================================================================
# 3. Surfactant Detection (Section 2.1.2)
# ===========================================================================

SURF_TYPE_ANIONIC = 'anionic'
SURF_TYPE_CATIONIC = 'cationic'
SURF_TYPE_NONIONIC = 'nonionic'
SURF_TYPE_ZWITTERIONIC = 'zwitterionic'
SURF_TYPES = [SURF_TYPE_ANIONIC, SURF_TYPE_CATIONIC, SURF_TYPE_NONIONIC, SURF_TYPE_ZWITTERIONIC]
SURF_TYPE_TO_IDX = {t: i for i, t in enumerate(SURF_TYPES)}


def detect_surfactant(smiles: str):
    """Detect head group, tail (≥4 carbon chain), and surfactant type.

    Returns:
        atom_mask_head: ndarray (N_atoms,) bool
        atom_mask_tail: ndarray (N_atoms,) bool
        surfactant_type: str
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(len(smiles), dtype=bool), np.zeros(len(smiles), dtype=bool), SURF_TYPE_NONIONIC

    n_atoms = mol.GetNumAtoms()
    head_mask = np.zeros(n_atoms, dtype=bool)
    tail_mask = np.zeros(n_atoms, dtype=bool)

    # Counterions to exclude
    counterion_patts = {
        'Na': Chem.MolFromSmarts('[Na+]'), 'Li': Chem.MolFromSmarts('[Li+]'),
        'K': Chem.MolFromSmarts('[K+]'), 'Cl': Chem.MolFromSmarts('[Cl-]'),
        'Br': Chem.MolFromSmarts('[Br-]'), 'I': Chem.MolFromSmarts('[I-]'),
    }
    counterion_atoms = set()
    for patt in counterion_patts.values():
        if patt:
            for m in mol.GetSubstructMatches(patt):
                counterion_atoms.update(m)

    # ---- Detectors ----
    anionic_patts = [
        ('sulfonate', 'S(=O)(=O)[O-]'), ('sulfate', 'OS(=O)(=O)[O-]'),
        ('carboxylate', 'C(=O)[O-]'), ('phosphate', 'OP(=O)([O-])[O-]'),
    ]
    cationic_patts = [
        ('quat_ammonium', '[N+](C)(C)C'), ('ammonium', '[NH3+]'),
        ('pyridinium', '[n+]1ccccc1'), ('imidazolium', '[n+]1cncc1'),
    ]
    nonionic_patts = [
        ('hydroxyl', '[OH]'), ('ether', 'COC'), ('polyoxyethylene', 'CCOCCO'),
        ('amide', 'NC(=O)'), ('ester', 'C(=O)OC'),
    ]

    # Classify
    has_anionic = mol.HasSubstructMatch(Chem.MolFromSmarts('[O-]')) or \
                  mol.HasSubstructMatch(Chem.MolFromSmarts('[S-]'))
    has_cationic = mol.HasSubstructMatch(Chem.MolFromSmarts('[N+]')) or \
                   mol.HasSubstructMatch(Chem.MolFromSmarts('[n+]'))

    if has_anionic and has_cationic:
        surf_type = SURF_TYPE_ZWITTERIONIC
    elif has_anionic:
        surf_type = SURF_TYPE_ANIONIC
    elif has_cationic:
        surf_type = SURF_TYPE_CATIONIC
    else:
        surf_type = SURF_TYPE_NONIONIC

    # Head mask
    detectors = {'anionic': anionic_patts, 'cationic': cationic_patts,
                 'nonionic': nonionic_patts}
    if surf_type == SURF_TYPE_ZWITTERIONIC:
        all_patts = anionic_patts + cationic_patts
    else:
        all_patts = detectors.get(surf_type, nonionic_patts)

    for name, sma in all_patts:
        patt = Chem.MolFromSmarts(sma)
        if patt:
            for m in mol.GetSubstructMatches(patt):
                for idx in m:
                    if idx not in counterion_atoms:
                        head_mask[idx] = True

    # ---- Tail: DFS for longest carbon chain >= 4 ----
    adj = defaultdict(list)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i].append(j)
        adj[j].append(i)

    def dfs_longest(start, visited, chain):
        best = chain[:]
        for nb in adj[start]:
            if nb in visited:
                continue
            if mol.GetAtomWithIdx(nb).GetAtomicNum() == 6 and nb not in counterion_atoms:
                visited.add(nb)
                result = dfs_longest(nb, visited, chain + [nb])
                if len(result) > len(best):
                    best = result
                visited.remove(nb)
        return best

    carbons = [a.GetIdx() for a in mol.GetAtoms()
               if a.GetAtomicNum() == 6 and a.GetIdx() not in counterion_atoms]
    best_tail = []
    for c in carbons:
        visited = {c}
        chain = dfs_longest(c, visited, [c])
        if len(chain) > len(best_tail):
            best_tail = chain

    if len(best_tail) >= 4:
        for idx in best_tail:
            tail_mask[idx] = True
    else:
        for a in mol.GetAtoms():
            idx = a.GetIdx()
            if a.GetAtomicNum() == 6 and not head_mask[idx] and idx not in counterion_atoms:
                tail_mask[idx] = True

    for idx in counterion_atoms:
        head_mask[idx] = False
        tail_mask[idx] = False

    return head_mask, tail_mask, surf_type


# ===========================================================================
# 4. PyG HeteroData Construction — Single Molecule
# ===========================================================================

def build_molecule_data(smiles: str):
    """Build PyG HeteroData for one molecule.

    Node types: 'atom' (55-dim), 'pharmacophore' (194-dim), 'reaction' (34-dim)
    Edge types: ('atom','bond','atom') with 14-dim features,
                plus cross-edges connecting pharm/reaction → atoms.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    n_atoms = mol.GetNumAtoms()

    # Atom features
    atom_feats = np.array([get_atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)

    # Bond features (bidirectional)
    edge_index = []
    bond_feats = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = get_bond_features(bond)
        edge_index.append([u, v]); edge_index.append([v, u])
        bond_feats.append(bf); bond_feats.append(bf.copy())

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    bond_feats = torch.tensor(np.array(bond_feats), dtype=torch.float32)

    # Pharmacophore & reaction features
    pharm_feats = torch.tensor(get_pharmacophore_features(mol).reshape(1, -1), dtype=torch.float32)
    react_feats = torch.tensor(get_reaction_features(mol).reshape(1, -1), dtype=torch.float32)

    # HeteroData — convert all to torch tensors
    data = HeteroData()
    data['atom'].x = torch.tensor(atom_feats, dtype=torch.float32)
    data['pharmacophore'].x = pharm_feats
    data['reaction'].x = react_feats

    data['atom', 'bond', 'atom'].edge_index = edge_index
    data['atom', 'bond', 'atom'].edge_attr = bond_feats

    # Cross edges
    data['pharmacophore', 'to_atom', 'atom'].edge_index = torch.tensor(
        [[0]*n_atoms, list(range(n_atoms))], dtype=torch.long)
    data['atom', 'to_pharmacophore', 'pharmacophore'].edge_index = torch.tensor(
        [list(range(n_atoms)), [0]*n_atoms], dtype=torch.long)
    data['reaction', 'to_atom', 'atom'].edge_index = torch.tensor(
        [[0]*n_atoms, list(range(n_atoms))], dtype=torch.long)
    data['atom', 'to_reaction', 'reaction'].edge_index = torch.tensor(
        [list(range(n_atoms)), [0]*n_atoms], dtype=torch.long)
    data['pharmacophore', 'to_reaction', 'reaction'].edge_index = torch.tensor(
        [[0], [0]], dtype=torch.long)
    data['reaction', 'to_pharmacophore', 'pharmacophore'].edge_index = torch.tensor(
        [[0], [0]], dtype=torch.long)

    # Surfactant info
    head_mask, tail_mask, surf_type = detect_surfactant(smiles)
    data['atom'].head_mask = torch.tensor(head_mask, dtype=torch.bool)
    data['atom'].tail_mask = torch.tensor(tail_mask, dtype=torch.bool)
    data.surf_type = surf_type
    data.surf_type_idx = SURF_TYPE_TO_IDX.get(surf_type, 2)

    return data


# ===========================================================================
# 5. Dataset
# ===========================================================================

class SurfactantDataset(Dataset):
    """Dataset: SMILES → PyG HeteroData, with caching and progress tracking."""

    def __init__(self, smiles_list, logcmc_list, cache=True, name=''):
        self.smiles_list = smiles_list
        self.logcmc_list = logcmc_list
        self.cache = cache
        self.name = name
        self._cache = {}

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        if idx in self._cache:
            data = self._cache[idx]
        else:
            data = build_molecule_data(self.smiles_list[idx])
            if self.cache:
                self._cache[idx] = data
        data.y = torch.tensor([self.logcmc_list[idx]], dtype=torch.float32)
        return data

    def build_all(self):
        """Pre-build all graphs with a progress bar."""
        n = len(self.smiles_list)
        print(f"  Building {n} graphs for {self.name}...")
        for i in range(n):
            if i % 200 == 0 and i > 0:
                print(f"    ... {i}/{n} ({100*i//n}%)")
            _ = self[i]  # triggers caching


# ===========================================================================
# 6. PharmHGT Model (Sections 2.1.1–2.1.3)
# ===========================================================================

class MultiViewCrossAttention(nn.Module):
    """Multi-View Cross Attention (Eq.1).

    Attention(Q, K, V) = Σ_p Ω_p · softmax(Q_p K_pᵀ / √d_k) · V_p
    """

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # View weights Ω_p (α=atom, β=pharm, γ=react)
        self.view_weights = nn.Parameter(torch.ones(3) / 3.0)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_atom, x_pharm, x_react):
        """(N_atom, H) ← cross-attend to pharm(1,H) + react(1,H)."""
        N, H = x_atom.shape
        x_src = torch.cat([x_pharm, x_react], dim=0)  # (2, H)

        Q = self.q_proj(x_atom).view(N, self.num_heads, self.head_dim)
        K = self.k_proj(x_src).view(2, self.num_heads, self.head_dim)
        V = self.v_proj(x_src).view(2, self.num_heads, self.head_dim)

        # (h, N, 2) — no inplace ops
        attn = torch.einsum('nhd,shd->hns', Q, K) * self.scale

        view_w = F.softmax(self.view_weights, dim=0)
        attn_w = torch.empty_like(attn)
        attn_w[:, :, 0] = attn[:, :, 0] * view_w[1]  # β-pharm
        attn_w[:, :, 1] = attn[:, :, 1] * view_w[2]  # γ-reaction

        attn_w = F.softmax(attn_w, dim=-1)
        attn_w = self.dropout(attn_w)

        out = torch.einsum('hns,shd->nhd', attn_w, V).reshape(N, H)
        out = self.out_proj(out)
        return out


class SimpleGNNLayer(nn.Module):
    """Simple GNN layer with explicit message passing — no inplace ops.

    Message:   MLP( cat(x_src, x_dst, e) )
    Aggregate: mean over incoming neighbors
    Update:    LayerNorm( x + MLP(cat(x, agg)) )
    """

    def __init__(self, in_dim, out_dim, edge_dim, dropout=0.1):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + edge_dim, out_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        # Message: MLP(x_src || x_dst || e)
        msg = self.msg_mlp(torch.cat([x[row], x[col], edge_attr], dim=-1))
        # Aggregate: mean over incoming (col) neighbors
        agg = scatter(msg, col, dim=0, dim_size=x.size(0), reduce='mean')
        # Update: residual + MLP
        upd = self.upd_mlp(torch.cat([x, agg], dim=-1))
        return x + upd


class PharmHGTModel(nn.Module):
    """PharmHGT (Sections 2.1.1-2.1.3).

    1. Atom-level GNN (Gα)
    2. Pharmacophore encoding (Gβ)
    3. Surfactant attention (head/tail masks)
    4. MVMP — Multi-view message passing (Eq.2)
    5. Hierarchical readout (Eq.3,4)
    6. Output MLP → LogCMC (Eq.5)
    """

    def __init__(self, atom_dim=55, bond_dim=14, pharm_dim=194, react_dim=34,
                 hidden_dim=256, num_layers=4, dropout=0.2, num_heads=8):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Feature projectors
        self.atom_embed = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.bond_embed = nn.Sequential(
            nn.Linear(bond_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.pharm_embed = nn.Sequential(
            nn.Linear(pharm_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.react_embed = nn.Sequential(
            nn.Linear(react_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout))

        # Surfactant attention (Sec 2.1.2)
        self.head_proj = nn.Linear(hidden_dim, hidden_dim)
        self.tail_proj = nn.Linear(hidden_dim, hidden_dim)

        # Multi-view GNN layers (MVMP, Eq.2)
        self.atom_conv = nn.ModuleList()
        self.pharm_mlp = nn.ModuleList()
        self.cross_attn = nn.ModuleList()
        self.norm = nn.ModuleList()

        for _ in range(num_layers):
            self.atom_conv.append(
                SimpleGNNLayer(hidden_dim, hidden_dim, hidden_dim, dropout))
            self.pharm_mlp.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim*2), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim*2, hidden_dim)))
            self.cross_attn.append(
                MultiViewCrossAttention(hidden_dim, num_heads, dropout))
            self.norm.append(nn.ModuleDict({
                'atom': nn.LayerNorm(hidden_dim),
                'pharm': nn.LayerNorm(hidden_dim),
                'react': nn.LayerNorm(hidden_dim)}))

        # Hierarchical readout (Eq.3,4)
        self.fusion1 = nn.Linear(hidden_dim * 2, hidden_dim)   # Zγ + Zβ → Zγβ
        self.fusion2 = nn.Linear(hidden_dim * 2, hidden_dim)   # Zγβ + Zα → Z_fused
        self.readout_attn = nn.Linear(hidden_dim, 1)           # attention over atoms

        # Output MLP (Eq.5)
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, hidden_dim//4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim//4, 1))

        self.dropout = nn.Dropout(dropout)

    def _apply_surfactant_attention(self, h_atom, head_mask, tail_mask, batch):
        """Apply head/tail mask-guided attention (Section 2.1.2).

        Computes per-graph head/tail prototypes and updates atom embeddings.
        Uses non-inplace scatter_add to avoid autograd version errors.
        """
        n_atoms = h_atom.size(0)
        device = h_atom.device
        # Start with zeros and accumulate
        delta = torch.zeros_like(h_atom)

        for g in batch.unique():
            mask = (batch == g)
            hg = h_atom[mask]
            hm = head_mask[mask]
            tm = tail_mask[mask]

            head_proto = hg[hm].mean(dim=0, keepdim=True).tanh() if hm.any() else \
                         torch.zeros(1, h_atom.size(1), device=device)
            tail_proto = hg[tm].mean(dim=0, keepdim=True).tanh() if tm.any() else \
                         torch.zeros(1, h_atom.size(1), device=device)

            head_bias = self.head_proj(head_proto)  # (1, H)
            tail_bias = self.tail_proj(tail_proto)  # (1, H)

            hm_f = hm.float().unsqueeze(1)
            tm_f = tm.float().unsqueeze(1)
            # Compute delta for this graph's atoms (non-inplace)
            delta_g = head_bias * hm_f + tail_bias * tm_f  # (n_g, H)
            # Scatter delta back to atom positions (non-inplace via index_add)
            g_indices = mask.nonzero(as_tuple=True)[0]
            delta = delta.index_add(0, g_indices, delta_g)

        return h_atom + delta

    def forward(self, data):
        """Forward pass.

        Args:
            data: batched HeteroData
        Returns:
            logcmc_pred: (B,)
        """
        x_atom = data['atom'].x
        x_pharm = data['pharmacophore'].x
        x_react = data['reaction'].x

        edge_index = data['atom', 'bond', 'atom'].edge_index
        edge_attr = data['atom', 'bond', 'atom'].edge_attr

        head_mask = data['atom'].head_mask
        tail_mask = data['atom'].tail_mask
        batch = data['atom'].batch if hasattr(data['atom'], 'batch') else \
                torch.zeros(x_atom.size(0), dtype=torch.long, device=x_atom.device)

        n_graphs = x_pharm.size(0)

        # Embed
        h_atom = self.atom_embed(x_atom)          # (total_atoms, H)
        h_pharm = self.pharm_embed(x_pharm)       # (B, H)
        h_react = self.react_embed(x_react)       # (B, H)
        h_bond = self.bond_embed(edge_attr)       # (total_bonds, H)

        # Surfactant attention (Sec 2.1.2)
        h_atom = self._apply_surfactant_attention(h_atom, head_mask, tail_mask, batch)

        # Get per-graph atom counts for cross-attention
        atom_counts = []
        for g in range(n_graphs):
            atom_counts.append((batch == g).sum().item())

        # Multi-view message passing (MVMP, Eq.2)
        for i in range(self.num_layers):
            # Atom-level GNN (SimpleGNNLayer already has residual + activation)
            h_atom_in = h_atom
            h_atom = self.norm[i]['atom'](self.atom_conv[i](h_atom, edge_index, h_bond))

            # Pharmacophore update
            h_pharm_in = h_pharm
            h_pharm = self.pharm_mlp[i](h_pharm)
            h_pharm = self.norm[i]['pharm'](h_pharm + h_pharm_in)

            # Cross-view attention (Eq.1)
            h_atom_out_list = []
            start = 0
            for g in range(n_graphs):
                n_a = atom_counts[g]
                h_a = h_atom[start:start + n_a]
                h_p = h_pharm[g:g + 1]
                h_r = h_react[g:g + 1]
                h_a_out = self.cross_attn[i](h_a, h_p, h_r)
                h_atom_out_list.append(h_a_out)
                start += n_a
            h_atom = torch.cat(h_atom_out_list, dim=0)

        # Hierarchical readout (Eq.3,4)
        preds = []
        start = 0
        for g in range(n_graphs):
            n_a = atom_counts[g]
            z_alpha = h_atom[start:start + n_a]         # (n_a, H)
            z_beta = h_pharm[g:g + 1]                    # (1, H)
            z_gamma = h_react[g:g + 1]                   # (1, H)

            # Eq.3: Zγ + Zβ → Zγβ
            z_gamma_beta = self.fusion1(
                torch.cat([z_gamma, z_beta], dim=-1)).tanh()

            # Eq.4: Attention-pooled Zα fused with Zγβ
            attn_scores = self.readout_attn(z_alpha)     # (n_a, 1)
            attn_w = F.softmax(attn_scores, dim=0)       # (n_a, 1)
            z_alpha_pooled = (z_alpha * attn_w).sum(dim=0, keepdim=True)

            z_fused = self.fusion2(
                torch.cat([z_gamma_beta, z_alpha_pooled], dim=-1)).relu()

            # Eq.5: MLP → LogCMC
            preds.append(self.output_mlp(z_fused).squeeze(-1))
            start += n_a

        return torch.cat(preds)


# ===========================================================================
# 7. Training & Evaluation
# ===========================================================================

def train_epoch(model, loader, optimizer, device):
    """Train one epoch. Return average MSE."""
    model.train()
    total_loss = 0.0
    n = 0
    for batch_idx, data in enumerate(loader):
        if batch_idx % 10 == 0:
            print(f"    batch {batch_idx}/{len(loader)}", end='\r')
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = F.mse_loss(pred, data.y.squeeze(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        n += data.num_graphs
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate. Return (metrics_dict, y_pred, y_true)."""
    model.eval()
    preds, targets = [], []
    for data in loader:
        data = data.to(device)
        preds.append(model(data).cpu().numpy())
        targets.append(data.y.squeeze(-1).cpu().numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'mse': mse, 'mae': mae, 'r2': r2, 'rmse': np.sqrt(mse)}, y_pred, y_true


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================================================================
# 8. Optuna (Section 3.4)
# ===========================================================================

def objective(trial, train_dataset, val_dataset, device):
    """Optuna objective — minimize validation MSE.

    Search space (paper Sec 3.4):
      hidden_dim 128-512, depth 2-6, dropout 0.1-0.5,
      batch_size 16-128, lr 1e-5~1e-3, heads {4,8}
    """
    hp = {
        'hidden_dim': trial.suggest_int('hidden_dim', 128, 512, step=32),
        'num_layers': trial.suggest_int('num_layers', 2, 6),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'num_heads': trial.suggest_categorical('num_heads', [4, 8]),
    }

    model = PharmHGTModel(
        atom_dim=ATOM_FEAT_DIM, bond_dim=BOND_FEAT_DIM,
        pharm_dim=PHARM_FEAT_DIM, react_dim=REACT_FEAT_DIM,
        hidden_dim=hp['hidden_dim'], num_layers=hp['num_layers'],
        dropout=hp['dropout'], num_heads=hp['num_heads'],
    ).to(device).float()

    opt = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=1e-5)
    tr_loader = PyGDataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True, num_workers=0)
    val_loader = PyGDataLoader(val_dataset, batch_size=hp['batch_size'], shuffle=False, num_workers=0)

    best = float('inf')
    patience, counter = 15, 0

    for epoch in range(1, 101):
        train_epoch(model, tr_loader, opt, device)
        if epoch % 5 == 0:
            metrics, _, _ = evaluate(model, val_loader, device)
            trial.report(metrics['mse'], epoch)
            if metrics['mse'] < best:
                best = metrics['mse']; counter = 0
            else:
                counter += 1
            if counter >= patience and epoch >= 30:
                break
            if trial.should_prune():
                raise optuna.TrialPruned()
    return best


def run_optuna(train_dataset, val_dataset, n_trials=30):
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=5)
    study = optuna.create_study(
        study_name='pharmhgt_optuna', direction='minimize',
        sampler=sampler, pruner=pruner)
    study.optimize(
        lambda t: objective(t, train_dataset, val_dataset, DEVICE),
        n_trials=n_trials, show_progress_bar=True)
    print(f"\n=== Best trial ===")
    print(f"  Val MSE: {study.best_value:.6f}")
    print(f"  Params: {study.best_params}")
    return study.best_params


# ===========================================================================
# 9. Main
# ===========================================================================

def main():
    DATA_TRAIN = './data/surfpro_imputed.csv'
    DATA_TEST = './data/surfpro_test.csv'
    TARGET_COL = 'pCMC'
    SMILES_COL = 'SMILES'
    VAL_FRAC = 0.125
    SEED = 42
    USE_OPTUNA = False       # set True to enable Optuna search
    N_OPTUNA_TRIALS = 30
    BATCH_SIZE = 64
    EPOCHS = 200
    LR = 5e-4
    PATIENCE = 30

    set_seed(SEED)
    print(f"Device: {DEVICE}")

    # ---- Load ----
    df_train = pd.read_csv(DATA_TRAIN).dropna(subset=[TARGET_COL])
    df_test = pd.read_csv(DATA_TEST).dropna(subset=[TARGET_COL])
    print(f"Train: {len(df_train)}, Test: {len(df_test)}")

    y_full = df_train[TARGET_COL].values.astype(np.float32)
    y_test = df_test[TARGET_COL].values.astype(np.float32)
    smi_train = df_train[SMILES_COL].tolist()
    smi_test = df_test[SMILES_COL].tolist()

    tidx, vidx = train_test_split(np.arange(len(smi_train)), test_size=VAL_FRAC, random_state=SEED)
    smi_tr = [smi_train[i] for i in tidx]; y_tr = y_full[tidx]
    smi_val = [smi_train[i] for i in vidx]; y_val = y_full[vidx]
    print(f"Train: {len(smi_tr)}, Val: {len(smi_val)}, Test: {len(smi_test)}")

    # ---- Build datasets ----
    print("Building datasets...")
    ds_tr = SurfactantDataset(smi_tr, y_tr, cache=True, name='train')
    ds_val = SurfactantDataset(smi_val, y_val, cache=True, name='val')
    ds_test = SurfactantDataset(smi_test, y_test, cache=True, name='test')
    ds_tr.build_all()
    ds_val.build_all()
    ds_test.build_all()

    # ---- Optuna or default params ----
    if USE_OPTUNA:
        best_params = run_optuna(ds_tr, ds_val, n_trials=N_OPTUNA_TRIALS)
    else:
        best_params = {'hidden_dim': 256, 'num_layers': 4, 'dropout': 0.2,
                       'batch_size': 64, 'lr': 5e-4, 'num_heads': 8}
        print(f"Using default params: {best_params}")

    # ---- Final training ----
    print("\nTraining final model...")
    smi_final = smi_tr + smi_val
    y_final = np.concatenate([y_tr, y_val])
    ds_final = SurfactantDataset(smi_final, y_final, cache=True, name='final')
    ds_final.build_all()
    loader_final = PyGDataLoader(ds_final, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    loader_val = PyGDataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    loader_test = PyGDataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = PharmHGTModel(
        atom_dim=ATOM_FEAT_DIM, bond_dim=BOND_FEAT_DIM,
        pharm_dim=PHARM_FEAT_DIM, react_dim=REACT_FEAT_DIM,
        hidden_dim=best_params['hidden_dim'], num_layers=best_params['num_layers'],
        dropout=best_params['dropout'], num_heads=best_params['num_heads'],
    ).to(DEVICE).float()

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    best_state = None
    best_val = float('inf')
    counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, loader_final, opt, DEVICE)
        sched.step(train_loss)

        if epoch % 5 == 0:
            metrics, _, _ = evaluate(model, loader_val, DEVICE)
            if metrics['mse'] < best_val:
                best_val = metrics['mse']
                best_state = deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                      f"Val MSE: {metrics['mse']:.4f} | Val R²: {metrics['r2']:.4f}")
            if counter >= PATIENCE:
                print(f"  Early stopping @ epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)

    # ---- Test ----
    print(f"\n{'='*60}")
    print("Test Evaluation")
    print(f"{'='*60}")
    test_metrics, yp, yt = evaluate(model, loader_test, DEVICE)
    print(f"  Test MSE:  {test_metrics['mse']:.4f}")
    print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"  Test MAE:  {test_metrics['mae']:.4f}")
    print(f"  Test R²:   {test_metrics['r2']:.4f}")

    # ---- Save ----
    torch.save({
        'state_dict': model.state_dict(),
        'params': best_params,
        'metrics': test_metrics,
    }, 'pharmhgt_best_model.pth')
    print(f"\nSaved to pharmhgt_best_model.pth")

    print(f"\n{'='*60}")
    print("SUMMARY — PharmHGT LogCMC Prediction")
    print(f"{'='*60}")
    print(f"  Train: {len(smi_tr)} + Val: {len(smi_val)} = {len(smi_final)}")
    print(f"  Test:  {len(smi_test)}")
    print(f"  Params: {best_params}")
    print(f"  Test MSE:  {test_metrics['mse']:.4f}")
    print(f"  Test MAE:  {test_metrics['mae']:.4f}")
    print(f"  Test R²:   {test_metrics['r2']:.4f}")
    print(f"\n  Ref: paper R² = 0.943 (Data1) / 0.915 (Data2) on different data.")
    print(f"  MD validation: optional — not implemented.")


if __name__ == '__main__':
    main()
