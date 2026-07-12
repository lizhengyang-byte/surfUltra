"""
model_loader.py — Load any of the 5 trained pCMC prediction models.

Usage:
    from model_loader import load_model, get_available_models
    model = load_model('catboost_pharmhgt')
    model = load_model('pharmhgt_gnn', device='cuda')
    print(get_available_models())
"""

import os
import joblib
import warnings

from .pharmhgt_model import load_pharmhgt_model

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')

# Model registry: internal_name -> weight_filename
WEIGHT_FILES = {
    'catboost_pharmhgt': 'catboost_pharmhgt_model.pkl',
    'xgboost_pharmhgt': 'xgboost_pharmhgt_model.pkl',
    'lightgbm_pharmhgt': 'lightgbm_pharmhgt_model.pkl',
    'pharmhgt_gnn': 'pharmhgt_best_model.pth',
    'catboost_all': 'catboost_all_features_model.pkl',
}

# Feature type each model expects
MODEL_FEATURE_MAP = {
    'catboost_pharmhgt': 'pharmhgt_522',
    'xgboost_pharmhgt': 'pharmhgt_522',
    'lightgbm_pharmhgt': 'pharmhgt_522',
    'pharmhgt_gnn': 'gnn',
    'catboost_all': 'all_209',
}


def get_available_models():
    """Return list of model names whose weight files exist in weights/."""
    available = []
    for name, fname in WEIGHT_FILES.items():
        if os.path.exists(os.path.join(WEIGHTS_DIR, fname)):
            available.append(name)
    return available


def load_catboost_pharmhgt(weights_dir=WEIGHTS_DIR):
    """Load CatBoost model (522-dim PharmHGT features + Optuna)."""
    path = os.path.join(weights_dir, WEIGHT_FILES['catboost_pharmhgt'])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weight file not found: {path}")
    return joblib.load(path)


def load_xgboost_pharmhgt(weights_dir=WEIGHTS_DIR):
    """Load XGBoost model (522-dim PharmHGT features + Optuna)."""
    path = os.path.join(weights_dir, WEIGHT_FILES['xgboost_pharmhgt'])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weight file not found: {path}")
    return joblib.load(path)


def load_lightgbm_pharmhgt(weights_dir=WEIGHTS_DIR):
    """Load LightGBM model (522-dim PharmHGT features + Optuna)."""
    path = os.path.join(weights_dir, WEIGHT_FILES['lightgbm_pharmhgt'])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weight file not found: {path}")
    return joblib.load(path)


def load_pharmhgt_gnn(weights_dir=WEIGHTS_DIR, device='cpu'):
    """Load PharmHGT GNN model from .pth checkpoint.

    Returns:
        (model, params, metrics) tuple
    """
    path = os.path.join(weights_dir, WEIGHT_FILES['pharmhgt_gnn'])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weight file not found: {path}")
    return load_pharmhgt_model(path, device=device)


def load_catboost_all(weights_dir=WEIGHTS_DIR):
    """Load CatBoost model (209-dim all RDKit descriptors + Optuna)."""
    path = os.path.join(weights_dir, WEIGHT_FILES['catboost_all'])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Weight file not found: {path}\n"
            f"  Run `python models/predictor/retrain_catboost_all.py` to generate it."
        )
    return joblib.load(path)


def load_model(model_name, weights_dir=WEIGHTS_DIR, device='cpu'):
    """Generic loader — dispatches to the correct loader by model_name.

    Args:
        model_name: One of 'catboost_pharmhgt', 'xgboost_pharmhgt',
                    'lightgbm_pharmhgt', 'pharmhgt_gnn', 'catboost_all'
        weights_dir: Directory containing weight files
        device: Torch device for GNN model (ignored for tree models)

    Returns:
        Loaded model (sklearn-compatible regressor or PharmHGTModel tuple)

    Raises:
        ValueError: Unknown model_name
        FileNotFoundError: Weight file missing
    """
    loaders = {
        'catboost_pharmhgt': lambda: load_catboost_pharmhgt(weights_dir),
        'xgboost_pharmhgt': lambda: load_xgboost_pharmhgt(weights_dir),
        'lightgbm_pharmhgt': lambda: load_lightgbm_pharmhgt(weights_dir),
        'pharmhgt_gnn': lambda: load_pharmhgt_gnn(weights_dir, device),
        'catboost_all': lambda: load_catboost_all(weights_dir),
    }
    if model_name not in loaders:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available: {list(loaders.keys())}"
        )
    return loaders[model_name]()
