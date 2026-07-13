"""
featurizer.py — Unified molecular featurization for pCMC prediction.

Provides two featurization pipelines:
  1. PharmHGT-style 522-dim vector (atom/bond aggregates + MACCS + BRICS + surfactant)
  2. All RDKit descriptors 209-dim vector

Usage:
    from featurizer import build_feature_vector_pharmhgt, smiles_to_features_all
    vec_522 = build_feature_vector_pharmhgt("CCO")
    vec_209 = smiles_to_features_all("CCO")
"""

import hashlib
from collections import defaultdict

import numpy as np

from rdkit import Chem  # pyright: ignore[reportAttributeAccessIssue]
from rdkit.Chem import (  # pyright: ignore[reportAttributeAccessIssue]
    rdMolDescriptors, Descriptors, AllChem, MACCSkeys, BRICS,
)
from rdkit.Chem.rdchem import BondType as BT, HybridizationType  # pyright: ignore[reportAttributeAccessIssue]

# ===========================================================================
# Constants
# ===========================================================================
ATOM_FEAT_DIM = 55
BOND_FEAT_DIM = 14
PHARM_FEAT_DIM = 194
REACT_FEAT_DIM = 34

_ATOM_TYPES = [1, 3, 5, 6, 7, 8, 9, 11, 14, 15, 16, 17, 19, 35, 53, 79]
_ATOM_TYPE_TO_IDX = {at: i for i, at in enumerate(_ATOM_TYPES)}

_HYBRIDIZATION_TYPES = [
    HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3,
    HybridizationType.SP3D, HybridizationType.SP3D2,
]
_HYB_TO_IDX = {h: i for i, h in enumerate(_HYBRIDIZATION_TYPES)}

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

SURF_TYPE_ANIONIC = 'anionic'
SURF_TYPE_CATIONIC = 'cationic'
SURF_TYPE_NONIONIC = 'nonionic'
SURF_TYPE_ZWITTERIONIC = 'zwitterionic'
SURF_TYPES = [SURF_TYPE_ANIONIC, SURF_TYPE_CATIONIC, SURF_TYPE_NONIONIC, SURF_TYPE_ZWITTERIONIC]
SURF_TYPE_TO_IDX = {t: i for i, t in enumerate(SURF_TYPES)}


# ===========================================================================
# 1. Low-level feature extractors (atom, bond, pharmacophore, reaction)
# ===========================================================================

def get_atom_features(atom: Chem.Atom) -> np.ndarray:
    """55-dim atom feature vector."""
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


def get_bond_features(bond: Chem.Bond) -> np.ndarray:
    """14-dim bond feature vector."""
    feat = np.zeros(14, dtype=np.float32)
    feat[_BOND_TYPE_MAP.get(bond.GetBondType(), 0)] = 1.0
    feat[4] = 1.0 if bond.GetIsConjugated() else 0.0
    feat[5] = 1.0 if bond.IsInRing() else 0.0
    feat[6 + _BOND_STEREO_MAP.get(bond.GetStereo(), 0)] = 1.0
    feat[12] = 1.0 if bond.GetBondType() == BT.AROMATIC else 0.0
    feat[13] = 1.0 if bond.IsInRing() else 0.0
    return feat


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
    """34-dim: BRICS fragment type histogram."""
    feat = np.zeros(REACT_FEAT_DIM, dtype=np.float32)
    try:
        n_rot = Descriptors.NumRotatableBonds(mol)
        if n_rot < 1:
            return feat
        frags_gen = BRICS.BRICSDecompose(mol, returnMols=False)
        frags = []
        for i, f in enumerate(frags_gen):
            if i >= 128:
                break
            frags.append(f)
        if frags:
            for f in frags:
                feat[int(hashlib.md5(f.encode()).hexdigest(), 16) % REACT_FEAT_DIM] += 1.0
            feat = feat / max(len(frags), 1)
    except Exception:
        pass
    return feat


# ===========================================================================
# 2. Surfactant Detection
# ===========================================================================

def detect_surfactant(smiles: str):
    """Detect head group, tail (>=4 carbon chain), and surfactant type.

    Returns:
        atom_mask_head: ndarray (N_atoms,) bool
        atom_mask_tail: ndarray (N_atoms,) bool
        surfactant_type: str
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool), SURF_TYPE_NONIONIC

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

    for _, sma in all_patts:
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
# 3. PharmHGT-style 522-dim feature vector (for tree models 1-4)
# ===========================================================================

def build_feature_vector_pharmhgt(smiles: str) -> np.ndarray | None:
    """Construct fixed-length 522-dim feature vector for one molecule.

    Pipeline:
      - Aggregated atom features (mean/std/min/max of 55-dim) -> 220
      - Aggregated bond features (mean/std/min/max of 14-dim) -> 56
      - Pharmacophore MACCS keys                           -> 194
      - BRICS reaction features                            -> 34
      - Surfactant type one-hot                            -> 4
      - Head/tail atom ratios                              -> 2
      - Basic molecular descriptors                        -> 12
      ---------------------------------------------------------
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
            atom_feats.mean(axis=0),
            atom_feats.std(axis=0),
            atom_feats.min(axis=0),
            atom_feats.max(axis=0),
        ])
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
        ])
    else:
        bond_agg = np.zeros(BOND_FEAT_DIM * 4, dtype=np.float32)

    # ---- 3. Pharmacophore features (194) ----
    pharm_feats = get_pharmacophore_features(mol)

    # ---- 4. Reaction / BRICS features (34) ----
    react_feats = get_reaction_features(mol)

    # ---- 5. Surfactant features (4 + 2) ----
    head_mask, tail_mask, surf_type = detect_surfactant(smiles)
    if len(head_mask) != n_atoms:
        head_mask = np.zeros(n_atoms, dtype=bool)
        tail_mask = np.zeros(n_atoms, dtype=bool)
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
# 4. All RDKit descriptors 209-dim (for model 5: CatBoost all features)
# ===========================================================================

def compute_all_descriptors(mol):
    """Compute all RDKit molecular descriptors.

    Returns:
        (feature_vector, descriptor_names)
    """
    descriptors = []
    names = []
    for desc_name, func in Descriptors.descList:
        try:
            val = func(mol)
            if val is not None and np.isfinite(val):
                descriptors.append(val)
            else:
                descriptors.append(0.0)
        except Exception:
            descriptors.append(0.0)
        names.append(desc_name)
    return np.array(descriptors), names


def smiles_to_features_all(smiles: str) -> np.ndarray:
    """Compute all RDKit descriptors from a SMILES string (~209-dim).

    Returns zero vector for invalid SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        dummy, _ = compute_all_descriptors(Chem.MolFromSmiles("C"))
        return np.zeros(len(dummy))
    feature_vector, _ = compute_all_descriptors(mol)
    return feature_vector


def get_all_descriptor_names() -> list:
    """Return all descriptor names in order, matching smiles_to_features_all output."""
    mol = Chem.MolFromSmiles("C")
    _, names = compute_all_descriptors(mol)
    return names
