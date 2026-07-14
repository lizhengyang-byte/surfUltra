"""
torch_models.py — PyTorch model definitions and inference functions for pCMC prediction.

Provides:
  - MLPRegressor, RNNRegressor, TransformerRegressor model classes
  - load_torch_model()  — load a checkpoint dict → reconstructed model
  - predict_torch_model()  — inference for PyTorch models (MLP/RNN/Transformer)
  - predict_tree_model()  — inference for sklearn-compatible tree models (CatBoost/XGBoost/LightGBM)

Shared between:
  - API_predictor.py  (Python programmatic API)
  - predict.py        (CLI entry point)
  - model_loader.py   (model registration & loading)
"""

import math

import numpy as np
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════
# 1. Model classes
# ═══════════════════════════════════════════════════════════════════


class MLPRegressor(nn.Module):
    """Multi-layer perceptron for pCMC regression.

    Architecture: [Linear → BatchNorm → Activation → Dropout] × N → Linear → 1
    """
    def __init__(self, input_dim, hidden_dim, n_layers, dropout, activation='relu'):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RNNRegressor(nn.Module):
    """LSTM-based regressor for pCMC prediction.

    Treats each of the 522 input features as a separate time step
    (seq_len=522, input_size=1).
    """
    def __init__(self, input_dim, hidden_dim, n_layers, dropout, activation='relu'):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)                     # (batch, input_dim) → (batch, input_dim, 1)
        lstm_out, _ = self.lstm(x)              # (batch, seq_len, hidden_dim)
        last_out = lstm_out[:, -1, :]           # (batch, hidden_dim)
        return self.fc(last_out).squeeze(-1)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for TransformerRegressor."""
    pe: torch.Tensor

    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerRegressor(nn.Module):
    """TransformerEncoder-based regressor for pCMC prediction.

    Treats each input feature as a position in a sequence,
    applies positional encoding + TransformerEncoder, mean-pools,
    then projects to a scalar.
    """
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=256, dropout=0.1, activation='relu'):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation=activation, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)                     # (batch, input_dim) → (batch, input_dim, 1)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)          # (batch, seq_len, d_model)
        x = x.mean(dim=1)                        # mean pool → (batch, d_model)
        return self.fc(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════
# 2. Model loader — reconstruct a model from a saved checkpoint dict
# ═══════════════════════════════════════════════════════════════════

# Map model_type string → model class
_TORCH_MODEL_CLASSES = {
    'mlp': MLPRegressor,
    'rnn': RNNRegressor,
    'transformer': TransformerRegressor,
}


def load_torch_model(path, device='cpu'):
    """Load a PyTorch model checkpoint and reconstruct the model.

    The checkpoint dict must contain:
      - 'model_type': one of 'mlp', 'rnn', 'transformer'
      - 'state_dict': model state dictionary
      - architecture parameters matching the relevant class constructor

    For old checkpoints without 'model_type', a heuristic fallback is used:
      - has 'd_model'/'nhead'/'dim_feedforward' → transformer
      - state dict has 'lstm.' keys → rnn
      - otherwise → mlp

    Returns:
        nn.Module in eval mode on the specified device.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    model_type = checkpoint.get('model_type')
    if model_type is None:
        # Fallback: detect model type from checkpoint contents
        if 'd_model' in checkpoint or 'nhead' in checkpoint or 'dim_feedforward' in checkpoint:
            model_type = 'transformer'
        elif any(k.startswith('lstm.') for k in checkpoint.get('state_dict', {})):
            model_type = 'rnn'
        else:
            model_type = 'mlp'

    if model_type not in _TORCH_MODEL_CLASSES:
        raise ValueError(
            f"Unknown model_type '{model_type}' in {path}. "
            f"Expected one of {list(_TORCH_MODEL_CLASSES.keys())}."
        )

    cls = _TORCH_MODEL_CLASSES[model_type]

    # Build constructor kwargs from checkpoint (exclude model_type and state_dict)
    kwargs = {k: v for k, v in checkpoint.items()
              if k not in ('model_type', 'state_dict', 'metrics')}

    model = cls(**kwargs)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════
# 3. Inference functions
# ═══════════════════════════════════════════════════════════════════


@torch.no_grad()
def predict_torch_model(model, smiles_list, feature_fn, device='cpu'):
    """Predict pCMC using a PyTorch model (MLP / RNN / Transformer).

    Args:
        model: A loaded PyTorch nn.Module (in eval mode).
        smiles_list: List of SMILES strings.
        feature_fn: SMILES → feature vector function (usually build_feature_vector_pharmhgt).
        device: Torch device for computation.

    Returns:
        np.ndarray of predictions (NaN for invalid SMILES).
    """
    model.eval()
    X_list = []
    valid_mask = []

    for smi in smiles_list:
        vec = feature_fn(smi)
        if vec is not None:
            X_list.append(vec)
            valid_mask.append(True)
        else:
            X_list.append(np.zeros(522, dtype=np.float32))
            valid_mask.append(False)

    if not X_list:
        return np.array([])

    X = np.array(X_list, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    preds = model(X_tensor).cpu().numpy().astype(np.float64)
    preds[~np.array(valid_mask)] = np.nan
    return preds


def predict_tree_model(model, smiles_list, feature_fn):
    """Predict pCMC using a tree-based model (CatBoost / XGBoost / LightGBM).

    Args:
        model: Loaded sklearn-compatible regressor with .predict(X).
        smiles_list: List of SMILES strings.
        feature_fn: Function to convert SMILES → feature vector.

    Returns:
        np.ndarray of predictions (NaN for invalid SMILES).
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
