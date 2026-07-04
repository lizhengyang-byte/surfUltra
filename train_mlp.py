import pandas as pd
import numpy as np
from smiles_to_features import smiles_to_features

data_train_file_path = 'data/surfpro_train.csv'
data_test_file_path = 'data/surfpro_test.csv'

# 数据清洗
df_train = pd.read_csv(data_train_file_path)
df_train = df_train.dropna(subset=['pCMC'])

# 将SMILES逐条转换为特征
x_train = np.array([smiles_to_features(smi) for smi in df_train["SMILES"]])
y_train = df_train['pCMC'].values

# ==================== GNN 训练 ====================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import optuna
import os
import matplotlib.pyplot as plt

# -------------------- 1. 数据集划分 & 标准化 --------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    x_train, y_train, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_s), torch.FloatTensor(y_train)),
                          batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(torch.FloatTensor(X_val_s),   torch.FloatTensor(y_val)),
                          batch_size=32, shuffle=False)
test_loader  = DataLoader(TensorDataset(torch.FloatTensor(X_test_s),  torch.FloatTensor(y_test)),
                          batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Train: {X_train_s.shape[0]}, Val: {X_val_s.shape[0]}, Test: {X_test_s.shape[0]}")

# -------------------- 2. 定义 MLP 模型 --------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden1=256, hidden2=128, hidden3=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.BatchNorm1d(hidden1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2), nn.BatchNorm1d(hidden2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden2, hidden3), nn.ReLU(),
            nn.Linear(hidden3, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, y_true, y_pred = 0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        total_loss += criterion(pred, yb).item() * xb.size(0)
        y_true.append(yb.cpu()); y_pred.append(pred.cpu())
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    return (total_loss / len(loader.dataset),
            np.sqrt(mean_squared_error(y_true, y_pred)),
            mean_absolute_error(y_true, y_pred),
            r2_score(y_true, y_pred))

# -------------------- 3. 训练函数（供 Optuna 调用） --------------------
def train_model(lr, dropout, wd, h1, h2, h3, bs, max_epochs=500):
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_s), torch.FloatTensor(y_train)),
                        batch_size=bs, shuffle=True)
    val_l   = DataLoader(TensorDataset(torch.FloatTensor(X_val_s),   torch.FloatTensor(y_val)),
                         batch_size=bs, shuffle=False)

    model = MLP(X_train_s.shape[1], h1, h2, h3, dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=15, min_lr=1e-6)
    criterion = nn.MSELoss()

    best_val_loss = float("inf"); best_epoch = 0; trigger = 0
    for epoch in range(max_epochs):
        train_epoch(model, loader, optimizer, criterion)
        val_loss, rmse_v, mae_v, r2_v = evaluate(model, val_l, criterion)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss; best_epoch = epoch; trigger = 0
        else:
            trigger += 1
            if trigger >= 40:
                break
    return best_val_loss, best_epoch

# -------------------- 4. Optuna 超参数搜索 --------------------
print("\n" + "=" * 60)
print("超参数搜索: Optuna (TPE) ...")

N_TRIALS = 30

def objective(trial):
    params = {
        "lr":      trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "wd":      trial.suggest_float("wd", 1e-6, 1e-3, log=True),
        "h1":      trial.suggest_categorical("h1", [128, 256, 512]),
        "h2":      trial.suggest_categorical("h2", [64, 128, 256]),
        "h3":      trial.suggest_categorical("h3", [32, 64]),
        "bs":      trial.suggest_categorical("bs", [16, 32, 64]),
    }
    best_val_loss, _ = train_model(**params)
    return -best_val_loss  # 最大化负损失 ≈ 最小化 MSE

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_params = study.best_params
print(f"\n最佳参数: {best_params}")
print(f"最佳验证 MSE: {-study.best_value:.6f}")

# -------------------- 5. 用最佳参数训练最终模型 --------------------
print("\n" + "=" * 60)
print("用最佳参数训练最终模型 ...")

final_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train_s), torch.FloatTensor(y_train)),
    batch_size=best_params["bs"], shuffle=True,
)
val_l = DataLoader(
    TensorDataset(torch.FloatTensor(X_val_s), torch.FloatTensor(y_val)),
    batch_size=best_params["bs"], shuffle=False,
)

model = MLP(
    X_train_s.shape[1],
    best_params["h1"], best_params["h2"], best_params["h3"],
    best_params["dropout"],
).to(device)
optimizer = optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["wd"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=15, min_lr=1e-6)
criterion = nn.MSELoss()

best_val_loss = float("inf"); trigger = 0
for epoch in range(500):
    train_epoch(model, final_loader, optimizer, criterion)
    val_loss, _, _, _ = evaluate(model, val_l, criterion)
    scheduler.step(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss; trigger = 0
    else:
        trigger += 1
        if trigger >= 50:
            print(f"Early stop @ epoch {epoch + 1}")
            break

# -------------------- 6. 最终评估 --------------------
def print_metrics(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  [{name}] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

with torch.no_grad():
    model.eval()
    for split_name, X_s, y in [("Train", X_train_s, y_train),
                                 ("Val",   X_val_s,   y_val),
                                 ("Test",  X_test_s,  y_test)]:
        pred = model(torch.FloatTensor(X_s).to(device)).cpu().numpy()
        print_metrics(y, pred, split_name)

# -------------------- 7. 保存预测结果 --------------------
os.makedirs("reports", exist_ok=True)
with torch.no_grad():
    model.eval()
    y_pred_test = model(torch.FloatTensor(X_test_s).to(device)).cpu().numpy()

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("True pCMC")
plt.ylabel("Predicted pCMC")
plt.title(f"GNN (MLP) — Test Set\nR² = {r2_score(y_test, y_pred_test):.4f}")
plt.tight_layout()
plt.savefig("reports/gnn_pred_vs_true.png", dpi=150)
plt.close()
print("\n预测结果已保存至 reports/gnn_pred_vs_true.png")