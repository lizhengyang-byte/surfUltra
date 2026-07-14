"""
API_predictor.py — Programmatic API for pCMC prediction using trained models.

Usage:
    from models.predictor.API_predictor import predictor, predictor_single

    # Use all available models (ensemble mean)
    df = predictor("input.csv", "output.csv", "all")

    # Use a specific model
    df = predictor("input.csv", "output.csv", "catboost_pharmhgt", device="cuda")
    df = predictor("input.csv", "output.csv", "mlp_pharmhgt", device="cuda")

    # Custom SMILES column name
    df = predictor("input.csv", "out.csv", "all", smiles_col="smiles_column")

    # Predict from a single SMILES string (new)
    result = predictor_single("CCO", "catboost_pharmhgt")
    print(result['predicted_pCMC'])

    # Predict from a list of SMILES
    result = predictor_single(["CCO", "CC(=O)O"], "all")
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
from models.predictor.torch_models import predict_torch_model, predict_tree_model


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
    elif model_name in ('mlp_pharmhgt', 'rnn_pharmhgt', 'transformer_pharmhgt'):
        # PyTorch model path (MLP / RNN / Transformer)
        preds = predict_torch_model(loaded, smiles_list, build_feature_vector_pharmhgt, device)
    else:
        # Tree model path (CatBoost / XGBoost / LightGBM / CatBoost all)
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

def predictor_single(smiles_input, model_name, device=None):
    """Predict pCMC from a single SMILES string or a list of SMILES strings.

    Args:
        smiles_input: A single SMILES string (str) or a list of SMILES strings.
                      Pass [] to get an empty result structure.
        model_name: Model name — one of 'catboost_pharmhgt', 'xgboost_pharmhgt',
                    'lightgbm_pharmhgt', 'pharmhgt_gnn', 'catboost_all',
                    'mlp_pharmhgt', 'rnn_pharmhgt', 'transformer_pharmhgt',
                    or 'all' for ensemble mean of all available models.
        device: Torch device for GNN inference ('cpu', 'cuda', or None for auto)

    Returns:
        dict with keys:
            - 'smiles': list of input SMILES strings
            - 'predicted_pCMC': list of prediction values (float or NaN)
            - 'model': model name used
            - If model_name='all': also includes individual model columns
              (e.g. 'predicted_pCMC_catboost_pharmhgt') and 'models_used'
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- Normalize input: str -> list, list -> list ----
    if isinstance(smiles_input, str):
        smiles_list = [smiles_input]
        single_input = True
    elif isinstance(smiles_input, list):
        smiles_list = smiles_input
        single_input = False
    else:
        raise TypeError(f"smiles_input must be str or list, got {type(smiles_input).__name__}")

    # ---- Handle empty list ----
    if len(smiles_list) == 0:
        print("Empty SMILES list — returning empty result.")
        result = {
            'smiles': [],
            'predicted_pCMC': [],
            'model': model_name,
        }
        if model_name == 'all':
            result['models_used'] = []
        return result

    # ---- Build a minimal DataFrame for _predict_single to work with ----
    df = pd.DataFrame({'SMILES': smiles_list})

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

    # ---- Build result dict ----
    pred_cols = [c for c in df.columns if c.startswith('predicted_pCMC')]
    result = {
        'smiles': smiles_list,
        'predicted_pCMC': df['predicted_pCMC'].tolist(),
        'model': model_name,
    }
    for col in pred_cols:
        if col != 'predicted_pCMC':
            result[col] = df[col].tolist()
    if model_name == 'all':
        result['models_used'] = available

    # Print summary
    n_valid = sum(1 for v in result['predicted_pCMC'] if not np.isnan(v))
    print(f"\n[Done] {n_valid}/{len(smiles_list)} valid predictions")
    if single_input:
        val = result['predicted_pCMC'][0]
        if np.isnan(val):
            print(f"   pCMC = NaN (invalid SMILES: '{smiles_list[0]}')")
        else:
            print(f"   pCMC = {val:.4f}")

    return result