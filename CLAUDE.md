# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SurfPredict** — Predict surfactant (表面活性剂) interfacial properties from SMILES molecular structures.

**Target variables:** pCMC (primary — lowest missing rate, approx normal), AW_ST_CMC, Gamma_max, Area_min, Pi_CMC, pC20  
**Data:** 1335 training / 140 test molecules (CSV with SMILES + property columns). pCMC has ~9.8% missingness in train set.  
**Current best model:** CatBoost + 217-dim RDKit descriptors + Optuna — **Test R² = 0.909**  
**Key challenge:** 6 targets with high missingness (up to 57.8%), strong inter-target correlations (pCMC↔pC20 r=0.77, AW_ST_CMC↔Pi_CMC r=-0.99).

## Architecture — Scripts Overview

All scripts are standalone (no shared modules). Each loads data, featurizes, splits, trains, evaluates.

### Feature Engineering Modules

| Module | Dim | Description |
| --- | --- | --- |
| `smiles_to_features.py` | ~62 | Hand-picked RDKit descriptors (MW, LogP, TPSA, charge, topology, VSA) |
| `smiles_to_features_all.py` | ~217 | All RDKit descriptors; exposes `smiles_to_features_all(smi)` and `compute_all_descriptors(mol)` |
| `smiles_to_features_Word2Vec.py` | configurable | Tokenizes SMILES → trains Word2Vec → averages token vectors. Used by Transformer model |

### Training Scripts

| Script | Features | Model | Optuna Trials | Test R² |
| --- | --- | --- | --- | --- |
| `train_catboost_use_all_features.py` | 217-dim RDKit | CatBoost | 50 | **0.909** ★ |
| `train_lightgbm_use_all_features.py` | 217-dim RDKit | LightGBM | 50 | 0.899 |
| `train_lightgbm_advanced.py` | 1415-dim (RDKit+MACCS+ECFP4+Aux) | LightGBM | 50 | 0.889 |
| `train_xgboost_use_all_features.py` | 217→109 (feature selection) | XGBoost | 100 | 0.867 |
| `train_mlp_use_all_features.py` | 217-dim RDKit | PyTorch MLP (3-layer) | 30 | 0.865 |
| `train_lightgbm_use_all_features.py` (manual) | 217-dim RDKit | LightGBM (manual params) | — | 0.859 |
| `train_rnn.py` (Keras) | 62-dim RDKit | Keras MLP (3-layer) | 30 | 0.840 |
| `train_mlp.py` | 62-dim RDKit | PyTorch MLP (3-layer) | 30 | 0.837 |
| `train_rnn_use_all_features.py` | 217-dim RDKit (as time steps) | PyTorch LSTM (3-layer) | 30 | 0.828 |
| `train_transformer_use_Word2Vec.py` | SMILES sequence tokens | Transformer Encoder | 25 | 0.791 |
| `train_SVR.py` | 62-dim RDKit | SVR (RBF) | 60 | 0.784 (Val) |
| `train_multi_linear_regression_use_all_features.py` | 217-dim RDKit | Ridge / PCA+OLS | 60 | 0.630 (Val) |
| `train_xgboost.py` | 62-dim RDKit | XGBoost | 60 | 0.840 (Val) |
| `train_LightGBM.py` | 62-dim RDKit | LightGBM | 60 | 0.404 (CV) |
| `train_gnn.py` | Molecular graph (atom 39 + bond 11) | AttentiveFP (PyG) | 30 | ⏳ not recorded |

### Utility Scripts

| Script | Purpose |
| --- | --- |
| `001.py` | Quick Ridge baseline via scikit-mol Morgan fingerprints |
| `002.py` | Multi-model benchmark (Ridge, RF, GBR, SVR, KNN, XGB, LGB) on Morgan fingerprints |

## Commands

```bash
python train_catboost_use_all_features.py   # best model
python train_lightgbm_use_all_features.py   # #2 model
python train_mlp_use_all_features.py        # best neural net
python train_gnn.py                         # GNN (AttentiveFP)
python 002.py                               # quick benchmark
python 001.py                               # quick Ridge baseline
```

All training scripts follow the same pattern: load CSV → drop pCMC NaN → featurize SMILES → train/val split (80/20) → baseline → Optuna search → final training → print RMSE/MAE/R² → save plot to `reports/`.

## Key Findings

- **CatBoost + 217-dim + Optuna** is the best combination (R²=0.909). LightGBM close behind (0.899).
- **Tree models > neural nets** for this task: CatBoost/LightGBM beat all MLP/LSTM variants.
- **More features ≠ better:** 1415-dim (RDKit+MACCS+ECFP4) slightly underperforms 217-dim RDKit alone (0.889 vs 0.899).
- **Linear models insufficient:** Ridge maxes at R²=0.63 — pCMC has nonlinear relationships with descriptors.
- **SMILES sequence modeling** (Transformer+Word2Vec, R²=0.79) lags behind descriptor-based methods — physicochemical info matters more than token patterns.
- Targets have high missingness: focus on pCMC. Other targets (AW_ST_CMC, etc.) are secondary.

## Important Notes

- **No test suite** — validation is inline (NaN checks, shape assertions, print statements).
- **All scripts are standalone** — each copies the same data-loading boilerplate. No shared module structure.
- **Graph data not precomputed** — `train_gnn.py` converts SMILES → PyG Data objects on-the-fly (no `.pt` cache).
- **Formal charge handling:** Surfactants contain counterions (Na⁺, Li⁺, K⁺) — RDKit parses them fine, but be aware for graph featurization in GNN.
- **GBK encoding on Windows** — avoid non-ASCII characters (², α, Chinese) in print statements to prevent UnicodeEncodeError with `python` command (use `PYTHONIOENCODING=utf-8` as workaround).
- **Ridge solver** — use `solver="sag"` for numerical stability with 217 collinear features; default Cholesky solver produces ill-conditioned matrix warnings.
- **Report** is at `reports/REPORT.md` — comprehensive model comparison with rankings, analysis, and hyperparameter details.

## Dependencies

All installed: RDKit 2026.3, PyTorch 2.6+cu124, PyG 2.8, scikit-learn, XGBoost 3.3, LightGBM 4.6, TensorFlow 2.21, Optuna 4.9, pandas 3.0, numpy, matplotlib, seaborn, gensim (Word2Vec), scikit-mol.
