# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SurfPredict** — Predict surfactant (表面活性剂) interfacial properties from SMILES molecular structures.

**Target variables:** pCMC (primary), AW_ST_CMC, Gamma_max, Area_min, Pi_CMC, pC20  
**Data:** ~1335 training / ~140 test molecules (CSV with SMILES + property columns)  
**Featurization:** ~62 selected RDKit descriptors (molecular weight, LogP, TPSA, counts, VSA distributions, etc.)  
**Formal charge handling:** Surfactants contain counterions (Na⁺, Li⁺, K⁺) — RDKit parses them but keep this in mind for graph featurization.

## Architecture

| File | Purpose |
|---|---|
| `smiles_to_features.py` | SMILES → 62-dim descriptor vector via RDKit (covers size, hydrophobicity, polarity, charge, topology, functional groups) |
| `train_xgboost.py` | XGBoost baseline + Optuna (60 trials) |
| `train_LightGBM.py` | LightGBM + Optuna (60 trials) |
| `train_SVR.py` | SVR (RBF/poly/sigmoid) + StandardScaler + Optuna (60 trials) |
| `train_rnn.py` | Keras MLP (Dense/BatchNorm/Dropout) + Optuna (30 trials) |
| `train_mlp.py` | PyTorch MLP + Optuna (30 trials) |
| `train_gnn.py` | PyG AttentiveFP (learns from molecular graph topology, not descriptors) + Optuna (30 trials) |
| `001.py` | Quick scikit-mol Morgan fingerprints + Ridge pipeline |
| `002.py` | Quick descriptive statistics |

**Key difference:** `train_gnn.py` builds molecular graphs on-the-fly (39-dim atom + 11-dim bond features → AttentiveFP), while all other scripts use the fixed 62-dim descriptor vector from `smiles_to_features.py`.

## Commands

```bash
# Run any training script directly (example for MLP)
python train_mlp.py

# Training scripts will:
# 1. Load data/surfpro_train.csv, drop pCMC NaN rows
# 2. Convert SMILES to features
# 3. Train/Val/Test split (70/15/15 or 80/20)
# 4. Train baseline model
# 5. Run Optuna hyperparameter search
# 6. Train final model with best params
# 7. Print RMSE/MAE/R² for each split
# 8. Save prediction plot to reports/

# Quick descriptive stats
python 002.py

# Quick Ridge baseline (Morgan fingerprints)
python 001.py
```

## Key Dependencies

All installed: RDKit 2026.3, PyTorch 2.6+cu124, PyG 2.8, scikit-learn, XGBoost 3.3, LightGBM 4.6, TensorFlow 2.21, Optuna 4.9, pandas 3.0, numpy, matplotlib, seaborn.

## Important Notes

- **No test suite** — the project has no unit tests. Validation is done inline (NaN checks, shape assertions, print statements).
- **All scripts are standalone** — there is no shared module structure. Each `train_*.py` copies the same data-loading preamble.
- **Graph data not precomputed** — `train_gnn.py` converts SMILES → PyG Data objects on-the-fly (no `.pt` cache files).
- **Reports directory** — prediction-vs-truth scatter plots save to `reports/` with filenames like `gnn_pred_vs_true.png`.
- **No .gitignore** — consider adding one.
