import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from rdkit import Chem
from smiles_to_features_all import compute_all_descriptors
import pandas as pd
import numpy as np
import optuna
import warnings
warnings.filterwarnings("ignore")

# ========== 加载数据 ==========
df_train = pd.read_csv('data/surfpro_train.csv').dropna(subset=['pCMC'])
df_test  = pd.read_csv('data/surfpro_test.csv').dropna(subset=['pCMC'])

y_train_full = df_train['pCMC'].values
y_test_orig  = df_test['pCMC'].values

# ========== 特征提取：全部 RDKit 描述符 ==========
print("计算 RDKit 描述符 ...")

def smiles_to_vector(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    vec, _ = compute_all_descriptors(mol)
    return vec

# 训练集
train_features = []
train_indices = []
for i, smi in enumerate(df_train["SMILES"]):
    vec = smiles_to_vector(smi)
    if vec is not None:
        train_features.append(vec)
        train_indices.append(i)

X_train_all = np.array(train_features, dtype=np.float64)
y_train_full = y_train_full[train_indices]

# 测试集
test_features = []
test_indices = []
for i, smi in enumerate(df_test["SMILES"]):
    vec = smiles_to_vector(smi)
    if vec is not None:
        test_features.append(vec)
        test_indices.append(i)

X_test = np.array(test_features, dtype=np.float64)
y_test_orig = y_test_orig[test_indices]

print(f"训练集: {X_train_all.shape}, 测试集: {X_test.shape}")

# ========== 划分验证集 (80/20) ==========
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train_all, y_train_full, test_size=0.2, random_state=42
)

# ========== 标准化 ==========
scaler = StandardScaler()
X_train_sub_s = scaler.fit_transform(X_train_sub)
X_val_s       = scaler.transform(X_val)
X_test_s      = scaler.transform(X_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("CUDA 不可用！")
print(f"Device: {device}")

# ========== 定义 RNN (LSTM) ==========
class RNNRegressor(nn.Module):
    """将 217 维描述符作为 217 个时间步（每步 1 个特征）输入 LSTM"""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        # x: (batch, 217) → (batch, seq_len=217, input_size=1)
        x = x.unsqueeze(-1)  # (batch, 217, 1)
        lstm_out, (h_n, _) = self.lstm(x)
        # 取最后一层的最后一个时间步输出
        last_out = lstm_out[:, -1, :]  # (batch, hidden_size)
        return self.regressor(last_out).squeeze(1)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
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
        y_true.append(yb.cpu())
        y_pred.append(pred.cpu())
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    return (total_loss / len(loader.dataset),
            np.sqrt(mean_squared_error(y_true, y_pred)),
            mean_absolute_error(y_true, y_pred),
            r2_score(y_true, y_pred))


# ========== Optuna 超参数搜索 ==========
print("\n" + "=" * 60)
print("超参数搜索: Optuna (TPE) ...")

N_TRIALS = 30

def train_model(lr, dropout, wd, hidden_size, num_layers, bs, max_epochs=500):
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_sub_s, dtype=torch.float32, device=device),
                      torch.tensor(y_train_sub, dtype=torch.float32, device=device)),
        batch_size=bs, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val_s, dtype=torch.float32, device=device),
                      torch.tensor(y_val, dtype=torch.float32, device=device)),
        batch_size=bs, shuffle=False,
    )

    model = RNNRegressor(input_size=1, hidden_size=hidden_size,
                          num_layers=num_layers, dropout=dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr * 5,
                                               steps_per_epoch=len(train_loader),
                                               epochs=max_epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_val_r2   = -float("inf")
    trigger = 0

    for epoch in range(max_epochs):
        train_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()
        val_loss, _, _, val_r2 = evaluate(model, val_loader, criterion)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_r2   = val_r2
            trigger = 0
        else:
            trigger += 1
            if trigger >= 80:
                break

    return best_val_loss, best_val_r2


def objective(trial):
    params = {
        "lr":          trial.suggest_float("lr", 1e-3, 5e-3, log=True),
        "dropout":     trial.suggest_float("dropout", 0.1, 0.3),
        "wd":          trial.suggest_float("wd", 1e-5, 1e-3, log=True),
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "num_layers":  trial.suggest_int("num_layers", 1, 3),
        "bs":          trial.suggest_categorical("bs", [32, 64]),
    }
    _, val_r2 = train_model(**params)
    return val_r2


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_params = study.best_params
print(f"\n最佳参数: {best_params}")
print(f"最佳验证 R²: {study.best_value:.4f}")

# ========== 用全部训练集训练最终模型 ==========
print("\n" + "=" * 60)
print("在全部训练集上训练最终模型 ...")

scaler_full = StandardScaler()
X_full_s = scaler_full.fit_transform(X_train_all)
X_test_final_s = scaler_full.transform(X_test)

train_dataset = TensorDataset(
    torch.tensor(X_full_s, dtype=torch.float32, device=device),
    torch.tensor(y_train_full, dtype=torch.float32, device=device)
)
final_loader = DataLoader(train_dataset, batch_size=best_params["bs"], shuffle=True)

model = RNNRegressor(
    input_size=1,
    hidden_size=best_params["hidden_size"],
    num_layers=best_params["num_layers"],
    dropout=best_params["dropout"],
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["wd"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
criterion = nn.MSELoss()

for epoch in range(1000):
    train_loss = train_epoch(model, final_loader, optimizer, criterion)
    scheduler.step()
    if (epoch + 1) % 100 == 0:
        print(f"  Epoch {epoch+1:4d}, Train Loss: {train_loss:.4f}")

# ========== 最终评估 ==========
print(f"\n{'Model':<25} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print("-" * 52)

with torch.no_grad():
    model.eval()
    pred = model(torch.tensor(X_full_s, dtype=torch.float32, device=device)).cpu().numpy()
    train_rmse = np.sqrt(mean_squared_error(y_train_full, pred))
    train_mae = mean_absolute_error(y_train_full, pred)
    train_r2 = r2_score(y_train_full, pred)
    print(f"{'RNN (Train)':<25} {train_rmse:>8.4f} {train_mae:>8.4f} {train_r2:>8.4f}")

    pred_test = model(torch.tensor(X_test_final_s, dtype=torch.float32, device=device)).cpu().numpy()
    test_rmse = np.sqrt(mean_squared_error(y_test_orig, pred_test))
    test_mae = mean_absolute_error(y_test_orig, pred_test)
    test_r2 = r2_score(y_test_orig, pred_test)
    print(f"{'RNN (Test)':<25} {test_rmse:>8.4f} {test_mae:>8.4f} {test_r2:>8.4f}")

print("-" * 52)
print(f"\n✅ 全部 RDKit 描述符 ({X_full_s.shape[1]} 维) + PyTorch LSTM")
print(f"   序列化: 217 个时间步 × 1 特征/步")
print(f"   训练集 R² = {train_r2:.4f}")
print(f"   测试集 R² = {test_r2:.4f}")

# 与原最佳模型对比
print(f"\n📊 与原最佳模型对比:")
print(f"{'Model':<25} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print("-" * 52)
print(f"{'MLP (全描述符)':<25} {'0.4083':>8} {'0.2525':>8} {'0.8650':>8}")
print(f"{'RNN (本模型)':<25} {test_rmse:>8.4f} {test_mae:>8.4f} {test_r2:>8.4f}")