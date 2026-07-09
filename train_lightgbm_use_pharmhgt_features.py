"""
train_lightgbm_use_pharmhgt_features.py — LightGBM with PharmHGT-style Featurization
=================================================================================

Keeps all feature extraction from pharmhgt_logcmc.py unchanged (atom 55-dim,
bond 14-dim, pharmacophore MACCS 194-dim, BRICS 34-dim, surfactant detection),
then aggregates them into per-molecule feature vectors for LightGBM.

Usage:
  python train_lightgbm_use_pharmhgt_features.py

Data:
  ./data/surfpro_imputed.csv  (training, imputed)
  ./data/surfpro_test.csv     (test)
"""

import os, sys, math, random, warnings
from copy import deepcopy
from collections import defaultdict

import numpy as np
import pandas as pd

# RDKit
from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
from rdkit.Chem import (
    rdMolDescriptors, Descriptors, AllChem, MACCSkeys, BRICS, Crippen, rdchem
)
from rdkit.Chem.rdchem import BondType as BT, HybridizationType

# LightGBM
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optuna
import optuna
from optuna.pruners import MedianPruner

warnings.filterwarnings('ignore')

# ===========================================================================
# Constants (same as pharmhgt_logcmc.py)
# ===========================================================================
ATOM_FEAT_DIM = 55
BOND_FEAT_DIM = 14
PHARM_FEAT_DIM = 194   # MACCS keys
REACT_FEAT_DIM = 34    # BRICS bond types

# ===========================================================================
# 1. Feature Extraction — Atom-level (55-dim) & Bond-level (14-dim)
#   EXACTLY as in pharmhgt_logcmc.py
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
#   EXACTLY as in pharmhgt_logcmc.py
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
                feat[abs(hash(f)) % REACT_FEAT_DIM] += 1.0
            feat = feat / max(len(frags), 1)
    except Exception:
        pass
    return feat


# ===========================================================================
# 3. Surfactant Detection (Section 2.1.2)
#   EXACTLY as in pharmhgt_logcmc.py
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
# 4. Feature Vector Construction — per molecule (LightGBM input)
# ===========================================================================

def build_feature_vector(smiles: str) -> np.ndarray:
    """Construct a fixed-length feature vector for one molecule.

    Combines:
      - Aggregated atom features (mean/std/min/max of 55-dim) → 220
      - Aggregated bond features (mean/std/min/max of 14-dim) → 56
      - Pharmacophore MACCS keys                           → 194
      - BRICS reaction features                            → 34
      - Surfactant type one-hot                            → 4
      - Head/tail atom ratios                              → 2
      - Basic molecular descriptors                        → 12
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

    # ---- 1. Atom-level aggregated features (55-dim → 220) ----
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

    # ---- 2. Bond-level aggregated features (14-dim → 56) ----
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
# 5. Main — Load Data, Featurize, Train LightGBM with Optuna
# ===========================================================================

def main():
    DATA_TRAIN = './data/surfpro_imputed.csv'
    DATA_TEST = './data/surfpro_test.csv'
    TARGET_COL = 'pCMC'
    SMILES_COL = 'SMILES'
    VAL_FRAC = 0.125
    SEED = 42
    N_OPTUNA_TRIALS = 50
    N_FOLDS = 5

    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("LightGBM + PharmHGT-style Featurization for LogCMC (pCMC) Prediction")
    print("=" * 60)

    # ---- Load data ----
    df_train = pd.read_csv(DATA_TRAIN).dropna(subset=[TARGET_COL])
    df_test = pd.read_csv(DATA_TEST).dropna(subset=[TARGET_COL])
    print(f"Train rows: {len(df_train)}, Test rows: {len(df_test)}")

    # ---- Featurize ----
    print("\nFeaturizing training set...")
    train_vecs, train_idx = [], []
    n_total = len(df_train)
    for i, smi in enumerate(df_train[SMILES_COL]):
        if i % 100 == 0 and i > 0:
            print(f"    ... {i}/{n_total} ({100*i//n_total}%)")
        vec = build_feature_vector(smi)
        if vec is not None:
            train_vecs.append(vec)
            train_idx.append(i)

    X_full = np.array(train_vecs, dtype=np.float32)
    y_full = df_train[TARGET_COL].values[train_idx]
    print(f"  Train features: {X_full.shape}")

    print("Featurizing test set...")
    test_vecs, test_idx = [], []
    n_test = len(df_test)
    for i, smi in enumerate(df_test[SMILES_COL]):
        if i % 50 == 0 and i > 0:
            print(f"    ... {i}/{n_test} ({100*i//n_test}%)")
        vec = build_feature_vector(smi)
        if vec is not None:
            test_vecs.append(vec)
            test_idx.append(i)

    X_test = np.array(test_vecs, dtype=np.float32)
    y_test = df_test[TARGET_COL].values[test_idx]
    print(f"  Test features:  {X_test.shape}")

    # Check for NaN/Inf
    if np.any(np.isnan(X_full)) or np.any(np.isinf(X_full)):
        print("  [WARNING] NaN/Inf in train features — replacing with 0")
        X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- Train/Validation split ----
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=VAL_FRAC, random_state=SEED)
    print(f"\nSplit: Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}")

    # ======================================================================
    # Optuna Hyperparameter Optimization (K-Fold CV)
    # ======================================================================
    print("\n" + "=" * 60)
    print(f"Optuna Hyperparameter Tuning ({N_OPTUNA_TRIALS} trials, {N_FOLDS}-Fold CV)")
    print("=" * 60)

    FEATURE_NAME = 'pharmhgt_522'

    def objective(trial):
        params = {
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'num_leaves': trial.suggest_int('num_leaves', 15, 255),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 1e-1, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'cat_smooth': trial.suggest_float('cat_smooth', 0.0, 50.0),
            'cat_l2': trial.suggest_float('cat_l2', 0.0, 50.0),
            'verbose': -1,
        }

        # DART-specific
        if params['boosting_type'] == 'dart':
            params['drop_rate'] = trial.suggest_float('drop_rate', 0.01, 0.3)
            params['max_drop'] = trial.suggest_int('max_drop', 1, 50)
            params['skip_drop'] = trial.suggest_float('skip_drop', 0.01, 0.3)

        cv_scores = []
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        for train_idx_cv, val_idx_cv in kf.split(X_full):
            X_tr_cv = X_full[train_idx_cv]
            y_tr_cv = y_full[train_idx_cv]
            X_val_cv = X_full[val_idx_cv]
            y_val_cv = y_full[val_idx_cv]

            model_cv = lgb.LGBMRegressor(**params, random_state=SEED, n_jobs=-1)
            model_cv.fit(
                X_tr_cv, y_tr_cv,
                eval_set=[(X_val_cv, y_val_cv)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
            )
            y_pred_cv = model_cv.predict(X_val_cv)
            rmse_cv = np.sqrt(mean_squared_error(y_val_cv, y_pred_cv))
            cv_scores.append(rmse_cv)

        return np.mean(cv_scores)

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
    study = optuna.create_study(
        study_name=f'lightgbm_{FEATURE_NAME}',
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

    print(f"\n=== Best Trial ===")
    print(f"  CV RMSE: {study.best_value:.6f}")
    print(f"  Params:  {study.best_params}")

    # ======================================================================
    # Final Training with Best Params
    # ======================================================================
    print("\n" + "=" * 60)
    print("Training Final Model with Best Hyperparameters")
    print("=" * 60)

    best_params = study.best_params
    best_params['verbose'] = -1
    # Remove DART-specific params if final model is GBDT
    if best_params.get('boosting_type') != 'dart':
        for dart_key in ['drop_rate', 'max_drop', 'skip_drop']:
            best_params.pop(dart_key, None)

    final_model = lgb.LGBMRegressor(**best_params, random_state=SEED, n_jobs=-1)
    final_model.fit(
        X_full, y_full,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )

    # ======================================================================
    # Evaluation
    # ======================================================================
    print(f"\n{'='*60}")
    print("Test Evaluation")
    print(f"{'='*60}")

    y_pred = final_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    print(f"  Test MSE:  {test_mse:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test MAE:  {test_mae:.4f}")
    print(f"  Test R²:   {test_r2:.4f}")

    # ---- Feature Importance ----
    print(f"\n{'='*60}")
    print("Top 20 Feature Importances")
    print(f"{'='*60}")
    importances = final_model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]
    feature_names = [
        'atom_mean', 'atom_std', 'atom_min', 'atom_max',
        'bond_mean', 'bond_std', 'bond_min', 'bond_max',
    ] + [f'maccs_{i}' for i in range(194)] + [f'brics_{i}' for i in range(34)] + [
        'surf_anionic', 'surf_cationic', 'surf_nonionic', 'surf_zwitterionic',
        'head_ratio', 'tail_ratio',
        'MolWt', 'LogP', 'TPSA', 'RotBonds', 'HBA', 'HBD',
        'NumRings', 'AroRings', 'AliRings', 'FracSP3', 'HeavyAtoms', 'NAtoms',
    ]
    # The concatenated feature vector has a specific layout — generate names
    # atom agg (220) + bond agg (56) + maccs (194) + brics (34) + surf (4) + head/tail (2) + desc (12)
    names = []
    for i in range(220):
        group = i // 55
        dim = i % 55
        prefix = ['atom_mean', 'atom_std', 'atom_min', 'atom_max'][group]
        names.append(f'{prefix}_{dim}')
    for i in range(56):
        group = i // 14
        dim = i % 14
        prefix = ['bond_mean', 'bond_std', 'bond_min', 'bond_max'][group]
        names.append(f'{prefix}_{dim}')
    for i in range(194):
        names.append(f'maccs_{i}')
    for i in range(34):
        names.append(f'brics_{i}')
    names.extend(['surf_anionic', 'surf_cationic', 'surf_nonionic', 'surf_zwitterionic'])
    names.extend(['head_ratio', 'tail_ratio'])
    names.extend([
        'MolWt', 'LogP', 'TPSA', 'RotBonds', 'HBA', 'HBD',
        'NumRings', 'AroRings', 'AliRings', 'FracSP3', 'HeavyAtoms', 'NAtoms',
    ])

    for rank, idx in enumerate(top_idx):
        print(f"  {rank+1:2d}. {names[idx]:25s}  {importances[idx]:.1f}")

    # ---- Save predictions plot ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('LightGBM + PharmHGT Features — pCMC Prediction', fontsize=14)

        # Pred vs True
        ax = axes[0]
        ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
        lims = [min(y_test.min(), y_pred.min()) - 0.5, max(y_test.max(), y_pred.max()) + 0.5]
        ax.plot(lims, lims, 'r--', alpha=0.8, linewidth=1)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('True pCMC'); ax.set_ylabel('Predicted pCMC')
        ax.set_title(f'Test R² = {test_r2:.4f}')
        ax.axis('square')

        # Residuals
        ax = axes[1]
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax.set_xlabel('Predicted pCMC'); ax.set_ylabel('Residuals')
        ax.set_title(f'MAE = {test_mae:.4f}')

        plt.tight_layout()
        plot_path = 'reports/lightgbm_pharmhgt_pred_vs_true.png'
        os.makedirs('reports', exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {plot_path}")
    except ImportError:
        print("\n(Matplotlib not available — skipping plot)")

    # ---- Save model ----
    import joblib
    model_path = 'lightgbm_pharmhgt_model.pkl'
    joblib.dump(final_model, model_path)
    print(f"Model saved to {model_path}")

    print(f"\n{'='*60}")
    print("SUMMARY — LightGBM + PharmHGT Features")
    print(f"{'='*60}")
    print(f"  Features:  {X_full.shape[1]}-dim (atom_agg + bond_agg + MACCS + BRICS + surfactant + descriptors)")
    print(f"  Train:     {len(X_full)} (split {len(X_train)} train + {len(X_val)} val)")
    print(f"  Test:      {len(X_test)}")
    print(f"  Optuna:    {N_OPTUNA_TRIALS} trials, {N_FOLDS}-fold CV")
    print(f"  Best CV RMSE: {study.best_value:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test MAE:  {test_mae:.4f}")
    print(f"  Test R²:   {test_r2:.4f}")


if __name__ == '__main__':
    main()
