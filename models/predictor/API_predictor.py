"""
API_predictor.py — Programmatic API for pCMC prediction using trained models.

Usage:
    from models.predictor.API_predictor import predictor

    # Use all available models (ensemble mean)
    df = predictor("input.csv", "output.csv", "all")

    # Use a specific model
    df = predictor("input.csv", "output.csv", "catboost_pharmhgt", device="cuda")

    # Custom SMILES column name
    df = predictor("input.csv", "out.csv", "all", smiles_col="smiles_column")
"""

import os
import warnings

import numpy as np
import pandas as pd
import torch

from models.predictor.featurizer import (
    build_feature_vector_pharmhgt,
    smiles_to_features_all,
)
from models.predictor.model_loader import (
    load_model,
    get_available_models,
    MODEL_FEATURE_MAP,
)
from models.predictor.pharmhgt_model import predict_pharmhgt_batch


def predict_tree_model(model, smiles_list, feature_fn):
    """Predict using a tree-based model (CatBoost / XGBoost / LightGBM).

    Args:
        model: Loaded sklearn-compatible regressor with .predict(X)
        smiles_list: List of SMILES strings
        feature_fn: Function to convert SMILES -> feature vector

    Returns:
        np.ndarray of predictions (NaN for invalid SMILES)
    """
    X_list = []
    valid_mask = []

    for smi in smiles_list:
        vec = feature_fn(smi)
        if vec is not None:
            X_list.append(vec)
            valid_mask.append(True)
        else:
            X_list.append(np.zeros(522 if 'pharmhgt' in feature_fn.__name__ else 209,
                                   dtype=np.float32))
            valid_mask.append(False)

    if not X_list:
        return np.array([])

    X = np.array(X_list, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    preds = model.predict(X).astype(np.float64)
    preds[~np.array(valid_mask)] = np.nan
    return preds


def predictor(input_csv, output_csv, model_name, smiles_col='SMILES', device=None):
    """Predict pCMC from a CSV of SMILES using a trained model.

    Args:
        input_csv: Path to input CSV with a SMILES column
        output_csv: Path to save the output CSV with predictions appended
        model_name: Model name (e.g. 'catboost_pharmhgt', 'xgboost_pharmhgt',
                    'lightgbm_pharmhgt', 'pharmhgt_gnn', 'catboost_all')
        smiles_col: Name of the SMILES column (default: 'SMILES')
        device: Torch device for GNN inference ('cpu', 'cuda', or None for auto)

    Returns:
        pd.DataFrame with predicted_pCMC column appended
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- Load data ----
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in CSV. Columns: {list(df.columns)}")

    smiles_list = df[smiles_col].astype(str).tolist()
    print(f"Loaded {len(smiles_list)} rows from {input_csv}")

    # ---- Predict ----
    if model_name == 'all':
        available = get_available_models()
        if not available:
            raise FileNotFoundError("No model weight files found in weights/ directory.")
        print(f"Using 'all' -> predicting with {len(available)} models: {available}")
        for m_name in available:
            _predict_single(df, smiles_list, m_name, device)
        df['predicted_pCMC'] = df[[f'predicted_pCMC_{m}' for m in available]].mean(axis=1)
    else:
        _predict_single(df, smiles_list, model_name, device)

    # ---- Save ----
    df.to_csv(output_csv, index=False)
    print(f"\n[Done] Results saved to {output_csv}")
    print(f"   Columns added: {[c for c in df.columns if c.startswith('predicted_pCMC')]}")
    return df


def _predict_single(df, smiles_list, model_name, device):
    """Predict with a single model and add the result column to df in-place."""
    print(f"  Loading {model_name} ... ", end='', flush=True)

    loaded = load_model(model_name, device=device)
    feature_type = MODEL_FEATURE_MAP[model_name]
    print(f"[predicting] ", end='', flush=True)

    if model_name == 'pharmhgt_gnn':
        model_obj = loaded[0] if isinstance(loaded, (list, tuple)) else loaded
        try:
            preds = predict_pharmhgt_batch(model_obj, smiles_list, device)
        except Exception as e:
            warnings.warn(f"GNN prediction failed: {e}")
            preds = np.full(len(smiles_list), np.nan, dtype=np.float64)
    else:
        feature_fn = (
            build_feature_vector_pharmhgt if feature_type == 'pharmhgt_522'
            else smiles_to_features_all
        )
        preds = predict_tree_model(loaded, smiles_list, feature_fn)

    col_name = f'predicted_pCMC_{model_name}'
    df[col_name] = preds
    n_valid = int(np.sum(~np.isnan(preds)))
    mean_val = np.nanmean(preds)
    print(f"done -- {n_valid}/{len(preds)} valid, mean={mean_val:.4f}")

    # Also set concise column for single-model case
    df['predicted_pCMC'] = preds
