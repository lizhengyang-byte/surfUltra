"""
smiles_to_features_pharmhgt.py — Shared PharmHGT-style 522-dim Featurization
=============================================================================

All three tree-model training scripts (CatBoost, LightGBM, XGBoost) import
featurization functions from this module. Features are cached as .npy files
under data/features/pharmhgt/ to avoid redundant recomputation.

Usage (training scripts):
    from smiles_to_features_pharmhgt import load_or_compute_features

    X_train, y_train, X_test, y_test = load_or_compute_features()
    # ... train model ...

Usage (single molecule, e.g. predictor):
    from smiles_to_features_pharmhgt import smiles_to_features_pharmhgt

    vec = smiles_to_features_pharmhgt("CCO")
    print(vec.shape)  # (522,)
"""

import hashlib, json, os, warnings, zlib
from collections import defaultdict

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
from rdkit.Chem import (
    rdMolDescriptors, Descriptors, AllChem, MACCSkeys, BRICS, Crippen, rdchem
)
from rdkit.Chem.rdchem import BondType as BT, HybridizationType

warnings.filterwarnings('ignore')

# ===========================================================================
# Constants
# ===========================================================================
ATOM_FEAT_DIM = 55
BOND_FEAT_DIM = 14
PHARM_FEAT_DIM = 194   # MACCS keys
REACT_FEAT_DIM = 34    # BRICS bond types
FEATURE_DIM = 522

CACHE_DIR = os.path.join('data', 'features', 'pharmhgt')

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

    val = atom.GetValence(Chem.rdchem.ValenceType.EXPLICIT)
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
# 2. Pharmacophore & Reaction Features
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
        n_rot = Descriptors.NumRotatableBonds(mol)
        if n_rot < 1:
            return feat
        frags_gen = BRICS.BRICSDecompose(mol, returnMols=False)
        frags = []
        for i, f in enumerate(frags_gen):
            if i >= 128:  # safety limit
                break
            frags.append(f)
        if frags:
            for f in frags:
                feat[zlib.crc32(f.encode()) % REACT_FEAT_DIM] += 1.0
            feat = feat / max(len(frags), 1)
    except Exception:
        pass
    return feat


# ===========================================================================
# 3. Surfactant Detection
# ===========================================================================

SURF_TYPE_ANIONIC = 'anionic'
SURF_TYPE_CATIONIC = 'cationic'
SURF_TYPE_NONIONIC = 'nonionic'
SURF_TYPE_ZWITTERIONIC = 'zwitterionic'
SURF_TYPES = [SURF_TYPE_ANIONIC, SURF_TYPE_CATIONIC, SURF_TYPE_NONIONIC, SURF_TYPE_ZWITTERIONIC]
SURF_TYPE_TO_IDX = {t: i for i, t in enumerate(SURF_TYPES)}


def detect_surfactant(smiles: str):
    """Detect head group, tail (= 4 carbon chain), and surfactant type.

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
# 4. Feature Vector Construction — per molecule
# ===========================================================================

def build_feature_vector(smiles: str) -> np.ndarray:
    """Construct a fixed-length feature vector for one molecule.

    Combines:
      - Aggregated atom features (mean/std/min/max of 55-dim) -> 220
      - Aggregated bond features (mean/std/min/max of 14-dim) -> 56
      - Pharmacophore MACCS keys                           -> 194
      - BRICS reaction features                            -> 34
      - Surfactant type one-hot                            -> 4
      - Head/tail atom ratios                              -> 2
      - Basic molecular descriptors                        -> 12
      ----------------------------------------------------------
      Total: 522 features

    Returns None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    n_atoms = mol.GetNumAtoms()

    # ---- 1. Atom-level aggregated features (55-dim -> 220) ----
    atom_feats = np.array([get_atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)
    if n_atoms > 0:
        atom_agg = np.concatenate([
            atom_feats.mean(axis=0),   # 55
            atom_feats.std(axis=0),    # 55
            atom_feats.min(axis=0),    # 55
            atom_feats.max(axis=0),    # 55
        ])  # 220
    else:
        atom_agg = np.zeros(ATOM_FEAT_DIM * 4, dtype=np.float32)

    # ---- 2. Bond-level aggregated features (14-dim -> 56) ----
    bond_feats_list = [get_bond_features(b) for b in mol.GetBonds()]
    if bond_feats_list:
        bond_feats = np.array(bond_feats_list, dtype=np.float32)
        bond_agg = np.concatenate([
            bond_feats.mean(axis=0),
            bond_feats.std(axis=0),
            bond_feats.min(axis=0),
            bond_feats.max(axis=0),
        ])  # 56
    else:
        bond_agg = np.zeros(BOND_FEAT_DIM * 4, dtype=np.float32)

    # ---- 3. Pharmacophore features (194) ----
    pharm_feats = get_pharmacophore_features(mol)

    # ---- 4. Reaction / BRICS features (34) ----
    react_feats = get_reaction_features(mol)

    # ---- 5. Surfactant features (4 + 2) ----
    head_mask, tail_mask, surf_type = detect_surfactant(smiles)
    surf_onehot = np.zeros(4, dtype=np.float32)
    surf_onehot[SURF_TYPE_TO_IDX.get(surf_type, 2)] = 1.0
    n_head = head_mask.sum() / max(n_atoms, 1)
    n_tail = tail_mask.sum() / max(n_atoms, 1)

    # ---- 6. Basic molecular descriptors (12) ----
    mol_wt = Descriptors.MolWt(mol) / 500.0
    logp = Descriptors.MolLogP(mol) / 10.0
    tpsa = Descriptors.TPSA(mol) / 200.0
    n_rot = Descriptors.NumRotatableBonds(mol) / max(n_atoms, 1)
    n_hba = Descriptors.NumHAcceptors(mol) / max(n_atoms, 1)
    n_hbd = Descriptors.NumHDonors(mol) / max(n_atoms, 1)
    n_rings = rdMolDescriptors.CalcNumRings(mol) / 20.0
    n_aro = rdMolDescriptors.CalcNumAromaticRings(mol) / 10.0
    n_ali = rdMolDescriptors.CalcNumAliphaticRings(mol) / 10.0
    frac_sp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    heavy_atoms = mol.GetNumHeavyAtoms() / 100.0
    n_atoms_norm = n_atoms / 200.0

    desc_feats = np.array([
        mol_wt, logp, tpsa, n_rot, n_hba, n_hbd, n_rings, n_aro, n_ali,
        frac_sp3, heavy_atoms, n_atoms_norm,
    ], dtype=np.float32)

    # ---- Concatenate all ----
    feature_vector = np.concatenate([
        atom_agg,       # 220
        bond_agg,       # 56
        pharm_feats,    # 194
        react_feats,    # 34
        surf_onehot,    # 4
        [n_head, n_tail],  # 2
        desc_feats,     # 12
    ])

    return feature_vector  # 522-dim


# ===========================================================================
# 5. Caching Layer
# ===========================================================================

def _smiles_hash(df, smiles_col='SMILES'):
    """Compute MD5 of concatenated SMILES to detect data changes."""
    combined = ''.join(df[smiles_col].dropna().values)
    return hashlib.md5(combined.encode()).hexdigest()


def _featurize_dataframe(df, smiles_col='SMILES', target_col='pCMC', verbose=True):
    """Featurize all SMILES in a DataFrame, returning X, y aligned to valid molecules."""
    from tqdm import tqdm

    n_total = len(df)
    results = []
    iterator = tqdm(df[smiles_col].values, desc="Featurizing") if verbose else df[smiles_col].values
    for smi in iterator:
        results.append(build_feature_vector(smi))

    valid_mask = [r is not None for r in results]
    X = np.array([r for r in results if r is not None], dtype=np.float32)
    y = df[target_col].values[valid_mask]
    n_valid = len(X)
    if verbose:
        print(f"  {n_valid}/{n_total} valid molecules")

    return X, y, valid_mask


def load_or_compute_features(
    train_csv='./data/surfpro_imputed.csv',
    test_csv='./data/surfpro_test.csv',
    target_col='pCMC',
    smiles_col='SMILES',
    cache_dir=None,
    force_recompute=False,
    verbose=True
):
    """Load cached features or compute and cache them.

    Caches X_train.npy, y_train.npy, X_test.npy, y_test.npy under cache_dir.
    Cache is invalidated when SMILES content changes (MD5 hash check).

    Args:
        train_csv: Path to training CSV.
        test_csv: Path to test CSV.
        target_col: Column name for target variable.
        smiles_col: Column name for SMILES.
        cache_dir: Directory for .npy cache (default: data/features/pharmhgt).
        force_recompute: If True, ignore cache and recompute.
        verbose: Print progress messages.

    Returns:
        X_train: ndarray (n_train_valid, 522)
        y_train: ndarray (n_train_valid,)
        X_test: ndarray (n_test_valid, 522)
        y_test: ndarray (n_test_valid,)
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR

    df_train = pd.read_csv(train_csv).dropna(subset=[target_col])
    df_test = pd.read_csv(test_csv).dropna(subset=[target_col])

    train_hash = _smiles_hash(df_train, smiles_col)
    test_hash = _smiles_hash(df_test, smiles_col)
    meta_path = os.path.join(cache_dir, 'metadata.json')

    cache_hit = False
    if not force_recompute and os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get('train_smiles_hash') == train_hash and meta.get('test_smiles_hash') == test_hash:
                cache_hit = True
        except (json.JSONDecodeError, KeyError):
            pass

    if cache_hit:
        if verbose:
            print(f"[Cache HIT] Loading features from {cache_dir}")
        X_train = np.load(os.path.join(cache_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(cache_dir, 'y_train.npy'))
        X_test = np.load(os.path.join(cache_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(cache_dir, 'y_test.npy'))
    else:
        if verbose:
            print(f"[Cache MISS] Computing features...")

        X_train, y_train, _ = _featurize_dataframe(
            df_train, smiles_col, target_col, verbose=verbose)
        X_test, y_test, _ = _featurize_dataframe(
            df_test, smiles_col, target_col, verbose=verbose)

        # NaN/Inf safety
        for name, arr in [('train', X_train), ('test', X_test)]:
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                if verbose:
                    print(f"  [WARNING] NaN/Inf in {name} features → replacing with 0")
                np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        # Save cache
        os.makedirs(cache_dir, exist_ok=True)
        np.save(os.path.join(cache_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(cache_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(cache_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(cache_dir, 'y_test.npy'), y_test)
        with open(meta_path, 'w') as f:
            json.dump({
                'train_smiles_hash': train_hash,
                'test_smiles_hash': test_hash,
                'train_csv': os.path.basename(train_csv),
                'test_csv': os.path.basename(test_csv),
                'target_col': target_col,
                'feature_dim': FEATURE_DIM,
            }, f, indent=2)
        if verbose:
            print(f"  Cached to {cache_dir}/")

    return X_train, y_train, X_test, y_test


# ===========================================================================
# 6. Single-molecule API
# ===========================================================================

def smiles_to_features_pharmhgt(smiles: str) -> np.ndarray:
    """Compute 522-dim feature vector for a single SMILES string.

    Args:
        smiles: SMILES string.

    Returns:
        522-dim feature vector, or None if SMILES is invalid.
    """
    return build_feature_vector(smiles)


# ===========================================================================
# Feature names (for interpretability)
# ===========================================================================

FEATURE_NAMES = []

# Atom stats (220)
for i in range(220):
    group = i // 55
    dim = i % 55
    prefix = ['atom_mean', 'atom_std', 'atom_min', 'atom_max'][group]
    FEATURE_NAMES.append(f'{prefix}_{dim}')

# Bond stats (56)
for i in range(56):
    group = i // 14
    dim = i % 14
    prefix = ['bond_mean', 'bond_std', 'bond_min', 'bond_max'][group]
    FEATURE_NAMES.append(f'{prefix}_{dim}')

# MACCS (194)
for i in range(194):
    FEATURE_NAMES.append(f'maccs_{i}')

# BRICS (34)
for i in range(34):
    FEATURE_NAMES.append(f'brics_{i}')

# Surfactant one-hot (4)
FEATURE_NAMES.extend(['surf_anionic', 'surf_cationic', 'surf_nonionic', 'surf_zwitterionic'])
# Head/tail ratio (2)
FEATURE_NAMES.extend(['head_ratio', 'tail_ratio'])
# Descriptors (12)
FEATURE_NAMES.extend([
    'MolWt', 'LogP', 'TPSA', 'RotBonds', 'HBA', 'HBD',
    'NumRings', 'AroRings', 'AliRings', 'FracSP3', 'HeavyAtoms', 'NAtoms',
])

assert len(FEATURE_NAMES) == FEATURE_DIM, f"FEATURE_NAMES length {len(FEATURE_NAMES)} != {FEATURE_DIM}"


if __name__ == '__main__':
    # Quick smoke test
    X_train, y_train, X_test, y_test = load_or_compute_features()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Feature dim: {X_train.shape[1]} (expect {FEATURE_DIM})")
