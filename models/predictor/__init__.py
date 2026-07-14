"""
models/predictor — Unified pCMC prediction pipeline for Surfactant property prediction.

Provides 8 trained models and a CLI for batch prediction from CSV:

  - CatBoost (+ PharmHGT features, Optuna)
  - XGBoost (+ PharmHGT features, Optuna)
  - LightGBM (+ PharmHGT features, Optuna)
  - PharmHGT (Heterogeneous Graph Transformer)
  - CatBoost (all RDKit descriptors, Optuna)
  - MLP (PyTorch, 522-dim PharmHGT features)
  - RNN / LSTM (PyTorch, 522-dim PharmHGT features)
  - Transformer Encoder (PyTorch, 522-dim PharmHGT features)

Usage:
    python -m models.predictor.predict input.csv output.csv
    python models/predictor/predict.py input.csv output.csv --model all
"""

from . import featurizer, model_loader, pharmhgt_model, torch_models
