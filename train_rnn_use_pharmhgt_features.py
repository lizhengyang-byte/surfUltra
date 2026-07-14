"""
train_rnn_use_pharmhgt_features.py — RNN (LSTM) with PharmHGT-style Featurization
==================================================================================

Uses shared featurization from smiles_to_features_pharmhgt.py (522-dim).
Features are cached under data/features/pharmhgt/ after first computation.
The 522-dim feature vector is treated as 522 time steps × 1 feature.

Usage:
  python train_rnn_use_pharmhgt_features.py

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
# RNN Model Definition (LSTM)
# ===========================================================================

class RNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout, activation='relu'):
        super().__init__()
        # Treat each of the input_dim features as a time step
        # (seq_len=input_dim, input_size=1)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, input_dim) → (batch, input_dim, 1)
        x = x.unsqueeze(-1)
        lstm_out, _ = self.lstm(x)         # (batch, seq_len, hidden_dim)
        last_out = lstm_out[:, -1, :]       # (batch, hidden_dim)
        return self.fc(last_out).squeeze(-1)


# ===========================================================================
# Training helpers
# ===========================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
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
# 5. Main — Load Data, Featurize, Train RNN (LSTM)
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
    print("RNN (LSTM) + PharmHGT-style Featurization for LogCMC (pCMC) Prediction")
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

    n_layers = 3
    hidden_dim = 64
    dropout = 0.2
    activation = 'relu'
    lr = 1e-3
    weight_decay = 1e-5
    batch_size = 32

    print(f"  n_layers:     {n_layers}")
    print(f"  hidden_dim:   {hidden_dim}")
    print(f"  dropout:      {dropout}")
    print(f"  activation:   {activation}")
    print(f"  lr:           {lr}")
    print(f"  weight_decay: {weight_decay}")
    print(f"  batch_size:   {batch_size}")

    # ======================================================================
    # Final Training
    # ======================================================================
    print("\n" + "=" * 60)
    print("Training Final Model")
    print("=" * 60)

    final_model = RNNRegressor(input_dim, hidden_dim, n_layers, dropout, activation).to(DEVICE)
    optimizer = optim.AdamW(final_model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    full_loader = make_loader(X_full, y_full, batch_size)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)

    n_epochs = 800
    best_rmse = float('inf')
    patience = 50
    trigger = 0
    best_state = None

    for epoch in range(1, n_epochs + 1):
        train_epoch(final_model, full_loader, optimizer, criterion, DEVICE)
        if epoch % 5 == 0 or epoch == n_epochs:
            _, val_rmse = evaluate(final_model, val_loader, criterion, DEVICE)
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_state = {k: v.cpu().clone() for k, v in final_model.state_dict().items()}
                trigger = 0
            else:
                trigger += 1
            if epoch % 50 == 0:
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
    model_path = 'models/predictor/weights/rnn_pharmhgt_model.pkl'
    # Save model metadata + state dict
    torch.save({
        'model_type': 'rnn',
        'input_dim': input_dim,
        'n_layers': n_layers,
        'hidden_dim': hidden_dim,
        'dropout': dropout,
        'activation': activation,
        'state_dict': final_model.state_dict(),
    }, model_path)
    print(f"Model saved to {model_path}")

    print(f"\n{'='*60}")
    print("SUMMARY — RNN (LSTM) + PharmHGT Features")
    print(f"{'='*60}")
    print(f"  Features:  {input_dim}-dim (atom_agg + bond_agg + MACCS + BRICS + surfactant + descriptors)")
    print(f"  Train:     {len(X_full)} (split {len(X_train)} train + {len(X_val)} val)")
    print(f"  Test:      {len(X_test)}")
    print(f"  Fixed hyperparams: {n_layers} LSTM layers, {hidden_dim} hidden, lr=1e-3, wd=1e-5, bs=32")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test MAE:  {test_mae:.4f}")
    print(f"  Test R²:   {test_r2:.4f}")


if __name__ == '__main__':
    main()
