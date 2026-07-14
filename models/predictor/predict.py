"""
predict.py — CLI entry point for pCMC prediction using all 8 trained models.

Usage:
    python models/predictor/predict.py input.csv output.csv
    python models/predictor/predict.py input.csv out.csv --model catboost_pharmhgt
    python models/predictor/predict.py input.csv out.csv --model all
    python models/predictor/predict.py input.csv out.csv --smiles-col SMILES --device cpu
    python models/predictor/predict.py --list-models
"""

import argparse
import os
import sys
import warnings

# Ensure project root is on sys.path for both direct and -m execution
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

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
    WEIGHT_FILES,
)
from models.predictor.pharmhgt_model import build_molecule_data, predict_pharmhgt_batch
from models.predictor.torch_models import predict_torch_model, predict_tree_model


def main():
    parser = argparse.ArgumentParser(
        description='Predict pCMC from SMILES CSV using trained models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python models/predictor/predict.py data/surfpro_test.csv results.csv\n"
            "  python models/predictor/predict.py data.csv out.csv -m catboost_pharmhgt\n"
            "  python models/predictor/predict.py data.csv out.csv -m all\n"
        ),
    )
    parser.add_argument('input_csv', nargs='?', default=None,
                        help='Input CSV file with SMILES column')
    parser.add_argument('output_csv', nargs='?', default=None,
                        help='Output CSV file (default: input _predicted.csv)')
    parser.add_argument('--model', '-m', action='append', default=None,
                        help=('Model(s) to use. Repeat flag or use "all". '
                              'Choices: catboost_pharmhgt, xgboost_pharmhgt, '
                              'lightgbm_pharmhgt, pharmhgt_gnn, catboost_all, '
                              'mlp_pharmhgt, rnn_pharmhgt, transformer_pharmhgt, all'))
    parser.add_argument('--smiles-col', '-s', default='SMILES',
                        help='SMILES column name (default: SMILES)')
    parser.add_argument('--device', '-d', default=None,
                        help='Torch device: cpu or cuda (default: auto-detect)')
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                        help='Batch size for GNN inference (default: 64)')
    parser.add_argument('--list-models', action='store_true',
                        help='List available models and exit')
    args = parser.parse_args()

    # ---- List models ----
    if args.list_models:
        available = get_available_models()
        print(f"Weights directory: {os.path.join(os.path.dirname(__file__), 'weights')}")
        print(f"\nAvailable models ({len(available)}/{len(WEIGHT_FILES)}):")
        for m in WEIGHT_FILES:
            status = '[OK]' if m in available else '[MISSING]'
            print(f"  {status:>9}  {m:30s} {WEIGHT_FILES[m]}")
        if 'catboost_all' not in available:
            print(f"\n  Tip: Run `python models/predictor/retrain_catboost_all.py`")
            print(f"       to train and save the missing catboost_all model.")
        return

    if args.input_csv is None:
        parser.print_help()
        sys.exit(1)

    # ---- Device ----
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Device: {device}")

    # ---- Determine models to use ----
    models_to_use = args.model if args.model else ['all']
    if 'all' in models_to_use:
        models_to_use = get_available_models()
    else:
        available = get_available_models()
        for m in list(models_to_use):
            if m not in available:
                print(f"  [Warning] model '{m}' not available (weight file missing). Skipping.")
                models_to_use.remove(m)

    if not models_to_use:
        print("No models available. Use --list-models to check which weight files exist.")
        sys.exit(1)

    # Default output name
    if args.output_csv is None:
        base, ext = os.path.splitext(args.input_csv)
        args.output_csv = f"{base}_predicted.csv"

    # ---- Load data ----
    if not os.path.exists(args.input_csv):
        print(f"Error: input file not found: {args.input_csv}")
        sys.exit(1)

    df = pd.read_csv(args.input_csv)
    if args.smiles_col not in df.columns:
        print(f"Error: column '{args.smiles_col}' not found in CSV. "
              f"Columns: {list(df.columns)}")
        sys.exit(1)

    smiles_list = df[args.smiles_col].astype(str).tolist()
    print(f"Loaded {len(smiles_list)} rows from {args.input_csv}")

    # ---- Predict with each model ----
    for model_name in models_to_use:
        print(f"  Loading {model_name} ... ", end='', flush=True)

        try:
            loaded = load_model(model_name, device=device)
        except FileNotFoundError as e:
            print(f"SKIP -- {e}")
            continue
        except Exception as e:
            print(f"ERROR -- {e}")
            continue

        feature_type = MODEL_FEATURE_MAP[model_name]
        print(f"[predicting] ", end='', flush=True)

        if model_name == 'pharmhgt_gnn':
            # Unpack the (model, params, metrics) tuple from load_pharmhgt_gnn
            if isinstance(loaded, (list, tuple)):
                model_obj = loaded[0]
            else:
                model_obj = loaded

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

    # If only one model, also copy to a concise column
    if len(models_to_use) == 1:
        src_col = f'predicted_pCMC_{models_to_use[0]}'
        df['predicted_pCMC'] = df[src_col]

    # ---- Save ----
    df.to_csv(args.output_csv, index=False)
    print(f"\n[Done] Results saved to {args.output_csv}")
    print(f"   Columns added: {[c for c in df.columns if c.startswith('predicted_pCMC')]}")


if __name__ == '__main__':
    main()
