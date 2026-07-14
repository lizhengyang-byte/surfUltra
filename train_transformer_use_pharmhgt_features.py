"""
train_transformer_use_pharmhgt_features.py — Transformer Encoder with PharmHGT-style Featurization
===================================================================================================

Uses shared featurization from smiles_to_features_pharmhgt.py (522-dim).
Features are cached under data/features/pharmhgt/ after first computation.
The 522-dim feature vector is treated as 522 time steps × 1 feature
and processed by a Transformer Encoder.

Usage:
  python train_transformer_use_pharmhgt_features.py

Data:
  ./data/surfpro_imputed.csv  (training, imputed)
  ./data/surfpro_test.csv     (test)
"""

import sys, math, random, warnings

import numpy as np
import pandas as pd

# Shared featurization
from smiles_to_features_pharmhgt import load_or_compute_features

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# Detect device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ===========================================================================
# Transformer Model Definition
# ===========================================================================

class PositionalEncoding(nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class TransformerRegressor(nn.Module):
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
        # x: (batch, input_dim) → (batch, input_dim, 1) → (batch, input_dim, d_model)
        x = x.unsqueeze(-1)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)       # (batch, seq_len, d_model)
        x = x.mean(dim=1)                     # mean pool → (batch, d_model)
        return self.fc(x).squeeze(-1)


# ===========================================================================
# Training helpers
# ===========================================================================

def train_epoch(model, loader, optimizer, criterion, device, grad_clip=None):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        total_loss += loss.item() * X_batch.size(0)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    return total_loss / len(loader.dataset), rmse


def make_loader(X, y, batch_size, shuffle=True):
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ===========================================================================
# 5. Main — Load Data, Featurize, Train Transformer
# ===========================================================================

def main():
    DATA_TRAIN = './data/surfpro_imputed.csv'
    DATA_TEST = './data/surfpro_test.csv'
    TARGET_COL = 'pCMC'
    SMILES_COL = 'SMILES'
    VAL_FRAC = 0.125
    SEED = 42

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed(SEED)

    print("=" * 60)
    print("Transformer Encoder + PharmHGT-style Featurization for LogCMC (pCMC) Prediction")
    print("=" * 60)

    # ---- Load / featurize (cached) ----
    X_full, y_full, X_test, y_test = load_or_compute_features(
        train_csv=DATA_TRAIN, test_csv=DATA_TEST,
        target_col=TARGET_COL, smiles_col=SMILES_COL,
    )
    print(f"  Train features: {X_full.shape}")
    print(f"  Test features:  {X_test.shape}")

    # ---- Train/Validation split ----
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=VAL_FRAC, random_state=SEED)
    print(f"\nSplit: Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}")

    input_dim = X_full.shape[1]

    # ======================================================================
    # Fixed Hyperparameters
    # ======================================================================
    print("\n" + "=" * 60)
    print("Using fixed hyperparameters (no Optuna tuning)")
    print("=" * 60)

    d_model = 128
    nhead = 4
    num_layers = 3
    dim_feedforward = 256
    dropout = 0.1
    activation = 'gelu'
    lr = 1e-3
    weight_decay = 1e-5
    batch_size = 16
    n_epochs = 500
    grad_clip = 5.0

    print(f"  d_model:        {d_model}")
    print(f"  nhead:          {nhead}")
    print(f"  num_layers:     {num_layers}")
    print(f"  dim_feedforward:{dim_feedforward}")
    print(f"  dropout:        {dropout}")
    print(f"  activation:     {activation}")
    print(f"  lr:             {lr}")
    print(f"  weight_decay:   {weight_decay}")
    print(f"  batch_size:     {batch_size}")
    print(f"  n_epochs:       {n_epochs}")
    print(f"  grad_clip:      {grad_clip}")

    # ======================================================================
    # Final Training
    # ======================================================================
    print("\n" + "=" * 60)
    print("Training Final Model")
    print("=" * 60)

    final_model = TransformerRegressor(input_dim, d_model, nhead, num_layers, dim_feedforward, dropout, activation).to(DEVICE)
    optimizer = optim.AdamW(final_model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Warmup forward pass to trigger CUDA kernel compilation
    with torch.no_grad():
        dummy = torch.randn(2, input_dim, device=DEVICE)
        _ = final_model(dummy)
    print("  Model warmup OK, starting training...")

    full_loader = make_loader(X_full, y_full, batch_size)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)

    best_rmse = float('inf')
    patience = 30
    trigger = 0
    best_state = None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    for epoch in range(1, n_epochs + 1):
        train_epoch(final_model, full_loader, optimizer, criterion, DEVICE, grad_clip)
        scheduler.step()
        if epoch % 5 == 0 or epoch == n_epochs:
            _, val_rmse = evaluate(final_model, val_loader, criterion, DEVICE)
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_state = {k: v.cpu().clone() for k, v in final_model.state_dict().items()}
                trigger = 0
            else:
                trigger += 1
            if epoch % 1 == 0:
                print(f"  Epoch {epoch:3d} — Val RMSE: {val_rmse:.4f} (best: {best_rmse:.4f})")
            if trigger >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Restore best weights
    if best_state is not None:
        final_model.load_state_dict(best_state)
    final_model.to(DEVICE)

    # ======================================================================
    # Evaluation
    # ======================================================================
    print(f"\n{'='*60}")
    print("Test Evaluation")
    print(f"{'='*60}")

    final_model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        y_pred = final_model(X_test_t).cpu().numpy()

    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    print(f"  Test MSE:  {test_mse:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test MAE:  {test_mae:.4f}")
    print(f"  Test R²:   {test_r2:.4f}")

    # ---- Save model ----
    model_path = 'transformer_pharmhgt_model.pkl'
    # Save model metadata + state dict
    torch.save({
        'input_dim': input_dim,
        'd_model': d_model,
        'nhead': nhead,
        'num_layers': num_layers,
        'dim_feedforward': dim_feedforward,
        'dropout': dropout,
        'activation': activation,
        'state_dict': final_model.state_dict(),
    }, model_path)
    print(f"Model saved to {model_path}")

    print(f"\n{'='*60}")
    print("SUMMARY — Transformer Encoder + PharmHGT Features")
    print(f"{'='*60}")
    print(f"  Features:  {input_dim}-dim (atom_agg + bond_agg + MACCS + BRICS + surfactant + descriptors)")
    print(f"  Train:     {len(X_full)} (split {len(X_train)} train + {len(X_val)} val)")
    print(f"  Test:      {len(X_test)}")
    print(f"  Fixed hyperparams: d_model={d_model}, {num_layers} layers, {nhead} heads, FFN={dim_feedforward}, GELU, lr=1e-3, wd=1e-5, bs={batch_size}, {n_epochs} epoch, grad_clip={grad_clip}, CosineAnnealingLR")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test MAE:  {test_mae:.4f}")
    print(f"  Test R²:   {test_r2:.4f}")


if __name__ == '__main__':
    main()
