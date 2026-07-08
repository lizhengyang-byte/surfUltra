"""
Transformer 模型 + RDKit 全部描述符 → pCMC 预测

工作流程：
  1. 用 RDKit 计算每个分子的全部描述符（约 217 维）
  2. StandardScaler 标准化
  3. 将 217 维特征视为 217 个时间步（每步 1 个特征），输入 Transformer Encoder
  4. 通过回归头预测 pCMC
  5. 用 Optuna 搜索超参数，训练最终模型

用法：
  python train_transformer_use_all_features.py              # 默认参数训练
  python train_transformer_use_all_features.py --epochs 200  # 指定训练轮数
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from smiles_to_features_all import compute_all_descriptors
import pandas as pd
import numpy as np
import warnings
from typing import Optional
from tqdm import tqdm
import optuna

warnings.filterwarnings("ignore")

# ========== 命令行参数 ==========
parser = argparse.ArgumentParser(description="Transformer + RDKit 全部描述符 分子性质预测")
parser.add_argument("--epochs", type=int, default=200,
                    help="训练轮数（默认: 200）")
parser.add_argument("--dim", type=int, default=128,
                    help="Transformer 嵌入维度（默认: 128）")
parser.add_argument("--seed", type=int, default=42,
                    help="随机种子（默认: 42）")
parser.add_argument("--lr", type=float, default=3e-4,
                    help="学习率（默认: 0.0003）")
parser.add_argument("--dropout", type=float, default=0.15,
                    help="Dropout 比率（默认: 0.15）")
parser.add_argument("--nhead", type=int, default=8,
                    help="Attention head 数（默认: 8）")
parser.add_argument("--num_layers", type=int, default=4,
                    help="Transformer 层数（默认: 4）")
parser.add_argument("--dim_feedforward", type=int, default=512,
                    help="FFN 维度（默认: 512）")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size（默认: 32）")
parser.add_argument("--weight_decay", type=float, default=1e-5,
                    help="权重衰减（默认: 0.00001）")
parser.add_argument("--val_ratio", type=float, default=0.15,
                    help="验证集比例（默认: 0.15）")
parser.add_argument("--patience", type=int, default=30,
                    help="早停耐心值（默认: 30，0=不早停）")
parser.add_argument("--optuna_trials", type=int, default=25,
                    help="Optuna 搜索次数（默认: 25，0=跳过 Optuna）")
parser.add_argument("--optuna_epochs", type=int, default=60,
                    help="Optuna 每 trial 的训练轮数（默认: 60）")
args = parser.parse_args()

# ========== 配置 ==========
RANDOM_SEED = args.seed
TOTAL_EPOCHS = args.epochs
OPTUNA_TRIALS = args.optuna_trials
OPTUNA_EPOCHS = args.optuna_epochs
VAL_RATIO = args.val_ratio
PATIENCE = args.patience

# 默认超参数（可通过命令行覆盖）
best_params = {
    "lr": args.lr,
    "weight_decay": args.weight_decay,
    "dropout": args.dropout,
    "batch_size": args.batch_size,
    "nhead": args.nhead,
    "num_layers": args.num_layers,
    "dim_feedforward": args.dim_feedforward,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("CUDA 不可用！")
print(f"Device: {device}")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ========== 加载数据 ==========
df_train = pd.read_csv('data/surfpro_train.csv').dropna(subset=['pCMC'])
df_test  = pd.read_csv('data/surfpro_test.csv').dropna(subset=['pCMC'])

y_train_full = df_train['pCMC'].values.astype(np.float32)
y_test_orig  = df_test['pCMC'].values.astype(np.float32)

train_smiles = df_train['SMILES'].tolist()
test_smiles  = df_test['SMILES'].tolist()

print(f"训练集: {len(train_smiles)} 条, 测试集: {len(test_smiles)} 条")

# ========== 特征提取：全部 RDKit 描述符 ==========
print("\n" + "=" * 60)
print("步骤 1: 使用 RDKit 计算全部描述符")
print("=" * 60)


def smiles_to_vector(smiles: str) -> Optional[np.ndarray]:
    """将 SMILES 转为 RDKit 全部描述符向量（自动处理 inf/nan 及过大的数值）"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    vec, _ = compute_all_descriptors(mol)
    # 替换 inf / -inf / nan 为 0.0
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    # 裁剪极值，避免 float32 溢出（float32 最大约 3.4e38）
    max_safe = 1e30
    vec = np.clip(vec, -max_safe, max_safe)
    return vec


print("计算训练集描述符...")
train_features = []
train_indices = []
for i, smi in enumerate(tqdm(train_smiles, desc="训练集")):
    vec = smiles_to_vector(smi)
    if vec is not None:
        train_features.append(vec)
        train_indices.append(i)

X_train_all = np.array(train_features, dtype=np.float32)
y_train_full = y_train_full[train_indices]
n_features = X_train_all.shape[1]

print("计算测试集描述符...")
test_features = []
test_indices = []
for i, smi in enumerate(tqdm(test_smiles, desc="测试集")):
    vec = smiles_to_vector(smi)
    if vec is not None:
        test_features.append(vec)
        test_indices.append(i)

X_test_all = np.array(test_features, dtype=np.float32)
y_test_orig = y_test_orig[test_indices]

print(f"训练集特征矩阵: {X_train_all.shape}, 测试集: {X_test_all.shape}")
print(f"描述符维度: {n_features}")

# ========== 划分验证集 ==========
print("\n" + "=" * 60)
print("步骤 2: 划分训练集/验证集 + 标准化")
print("=" * 60)

X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train_all, y_train_full, test_size=VAL_RATIO, random_state=RANDOM_SEED,
)

# 标准化
scaler = StandardScaler()
X_train_sub_s = scaler.fit_transform(X_train_sub)
X_val_s       = scaler.transform(X_val)
X_test_s      = scaler.transform(X_test_all)

print(f"训练子集: {X_train_sub_s.shape}, 验证集: {X_val_s.shape}")
print(f"测试集: {X_test_s.shape}")

# ========== 定义 Transformer 模型 ==========
print("\n" + "=" * 60)
print("步骤 3: 定义 Transformer 模型（描述符序列版）")
print("=" * 60)

D_MODEL = args.dim  # Transformer 嵌入维度


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TransformerRegressor(nn.Module):
    """
    Transformer Encoder + 回归头（用于分子性质预测）

    输入：标准化后的 RDKit 描述符向量 (batch, n_features)
    将每个描述符视为一个时间步，用 Linear(1→d_model) 映射为嵌入向量，
    再加位置编码后通过 Transformer Encoder。

    架构：
      - Linear(1 → d_model) 映射每个标量描述符
      - Positional Encoding
      - Transformer Encoder (N 层)
      - Mean Pooling（对所有时间步平均池化）
      - LayerNorm → Dropout → 回归头
    """

    def __init__(
        self,
        n_features: int,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features

        # 将每个标量描述符映射到 d_model 维度
        self.input_proj = nn.Linear(1, d_model)

        self.pos_encoder = PositionalEncoding(d_model, max_len=n_features)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu',
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features)
        x = x.unsqueeze(-1)  # (batch, n_features, 1)
        x = self.input_proj(x) * np.sqrt(self.d_model)  # (batch, n_features, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch, n_features, d_model)
        x = x.mean(dim=1)  # Mean Pooling → (batch, d_model)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.regressor(x).squeeze(-1)  # (batch,)
        return x


# ========== 训练/评估函数 ==========


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
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
    return (
        total_loss / len(loader.dataset),
        np.sqrt(mean_squared_error(y_true, y_pred)),
        mean_absolute_error(y_true, y_pred),
        r2_score(y_true, y_pred),
    )


# ========== Optuna 超参数搜索 ==========
if OPTUNA_TRIALS > 0:
    print("\n" + "=" * 60)
    print("步骤 4: Optuna 超参数搜索")
    print("=" * 60)

    train_dataset_opt = TensorDataset(
        torch.tensor(X_train_sub_s, dtype=torch.float32),
        torch.tensor(y_train_sub, dtype=torch.float32),
    )
    val_dataset_opt = TensorDataset(
        torch.tensor(X_val_s, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader_opt = DataLoader(train_dataset_opt, batch_size=32, shuffle=True)
    val_loader_opt = DataLoader(val_dataset_opt, batch_size=32, shuffle=False)
    criterion_opt = nn.MSELoss()

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.05, 0.35)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        nhead = trial.suggest_categorical("nhead", [4, 8])
        num_layers = trial.suggest_int("num_layers", 2, 4)
        dim_feedforward = trial.suggest_categorical("dim_feedforward", [256, 512])

        model = TransformerRegressor(
            n_features=n_features,
            d_model=D_MODEL,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=OPTUNA_EPOCHS, eta_min=1e-6,
        )

        best_val_r2 = -float("inf")
        patience_counter = 0
        opt_patience = max(5, OPTUNA_EPOCHS // 4)

        for epoch in range(OPTUNA_EPOCHS):
            train_epoch(model, train_loader_opt, optimizer, criterion_opt)
            scheduler.step()

            if (epoch + 1) % 5 == 0:
                _, _, _, val_r2 = evaluate(model, val_loader_opt, criterion_opt)
                trial.report(val_r2, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= opt_patience:
                    break

        return best_val_r2

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

    print(f"\n┌─ Optuna 搜索完成 ─────────────────────────────")
    print(f"│ 最佳 Trial: #{study.best_trial.number}")
    print(f"│ 最佳验证 R²: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"│   {k}: {v}")
    print(f"└────────────────────────────────────────────────")

    # 将 Optuna 最佳超参数合并到 best_params
    best_params.update(study.best_params)
    best_params.setdefault("batch_size", 32)

# ========== 训练最终模型（全量训练子集 + 早停） ==========
print("\n" + "=" * 60)
if OPTUNA_TRIALS > 0:
    print("步骤 5: 训练最终模型（使用 Optuna 最佳超参数）")
else:
    print("步骤 4: 训练最终模型（使用默认超参数）")
print("=" * 60)

print(f"超参数:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

model = TransformerRegressor(
    n_features=n_features,
    d_model=D_MODEL,
    nhead=best_params["nhead"],
    num_layers=best_params["num_layers"],
    dim_feedforward=best_params["dim_feedforward"],
    dropout=best_params["dropout"],
).to(device)

train_dataset = TensorDataset(
    torch.tensor(X_train_sub_s, dtype=torch.float32),
    torch.tensor(y_train_sub, dtype=torch.float32),
)
val_dataset = TensorDataset(
    torch.tensor(X_val_s, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32),
)
test_dataset = TensorDataset(
    torch.tensor(X_test_s, dtype=torch.float32),
    torch.tensor(y_test_orig, dtype=torch.float32),
)

train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=best_params["batch_size"], shuffle=False)

optimizer = optim.AdamW(
    model.parameters(),
    lr=best_params["lr"],
    weight_decay=best_params["weight_decay"],
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6,
)
criterion = nn.MSELoss()

# ========== 早停 + 最佳模型保存 ==========
best_val_r2 = -float("inf")
best_epoch = 0
patience_counter = 0
best_model_state = None

print(f"\n{'Epoch':>6}  {'Train_Loss':>10}  {'Val_Loss':>9}  {'Val_RMSE':>8}  {'Val_R²':>8}  {'Best_R²':>8}")
print("-" * 70)

for epoch in range(TOTAL_EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    scheduler.step()

    if (epoch + 1) % 5 == 0:
        val_loss, val_rmse, val_mae, val_r2 = evaluate(model, val_loader, criterion)
        marker = ""
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = model.state_dict()
            marker = " ◀"
        else:
            patience_counter += 1

        print(
            f"{epoch+1:>6d}  {train_loss:>10.4f}  {val_loss:>9.4f}  "
            f"{val_rmse:>8.4f}  {val_r2:>8.4f}  {best_val_r2:>8.4f}{marker}"
        )

        if PATIENCE > 0 and patience_counter >= PATIENCE:
            print(f"\n⏹️  早停！验证集 R² 连续 {PATIENCE} 轮未提升，最佳 epoch: {best_epoch}")
            break

# 恢复最佳模型
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"✅ 已恢复最佳模型 (epoch {best_epoch}, Val R²={best_val_r2:.4f})")

# ========== 最终评估（全量训练集 + 测试集，使用同一 scaler）==========
print("\n转换全量训练集特征（使用与训练一致的 scaler）...")
X_full_s = scaler.transform(X_train_all)
X_test_final_s = scaler.transform(X_test_all)

train_eval_dataset = TensorDataset(
    torch.tensor(X_full_s, dtype=torch.float32),
    torch.tensor(y_train_full, dtype=torch.float32),
)
train_eval_loader = DataLoader(train_eval_dataset, batch_size=best_params["batch_size"], shuffle=False)

test_eval_dataset = TensorDataset(
    torch.tensor(X_test_final_s, dtype=torch.float32),
    torch.tensor(y_test_orig, dtype=torch.float32),
)
test_eval_loader = DataLoader(test_eval_dataset, batch_size=best_params["batch_size"], shuffle=False)

print(f"\n{'Model':<35} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print("-" * 62)

with torch.no_grad():
    model.eval()

    # 训练集预测
    train_preds = []
    for xb, _ in train_eval_loader:
        xb = xb.to(device)
        pred = model(xb)
        train_preds.append(pred.cpu().numpy())
    train_preds = np.concatenate(train_preds)

    train_rmse = np.sqrt(mean_squared_error(y_train_full, train_preds))
    train_mae = mean_absolute_error(y_train_full, train_preds)
    train_r2 = r2_score(y_train_full, train_preds)
    print(f"{'Transformer + RDKit (Train)':<35} {train_rmse:>8.4f} {train_mae:>8.4f} {train_r2:>8.4f}")

    # 测试集预测（使用正确的 scaler）
    test_preds = []
    for xb, _ in test_eval_loader:
        xb = xb.to(device)
        pred = model(xb)
        test_preds.append(pred.cpu().numpy())
    test_preds = np.concatenate(test_preds)

    test_rmse = np.sqrt(mean_squared_error(y_test_orig, test_preds))
    test_mae = mean_absolute_error(y_test_orig, test_preds)
    test_r2 = r2_score(y_test_orig, test_preds)
    print(f"{'Transformer + RDKit (Test)':<35} {test_rmse:>8.4f} {test_mae:>8.4f} {test_r2:>8.4f}")

print("-" * 62)

# ========== 模型对比 ==========
print(f"\n📊 模型对比:")
print(f"{'Model':<35} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print("-" * 62)
print(f"{'SVR (原最佳)':<35} {'0.6826':>8} {'0.4436':>8} {'0.6227':>8}")
print(f"{'Transformer + RDKit (Test)':<35} {test_rmse:>8.4f} {test_mae:>8.4f} {test_r2:>8.4f}")

print(f"\n✅ 最终模型参数{'（Optuna 搜索优化）' if OPTUNA_TRIALS > 0 else ''}:")
print(f"   描述符维度: {n_features}")
print(f"   Transformer 嵌入维度 (d_model): {D_MODEL}")
print(f"   Transformer 层数: {best_params['num_layers']}")
print(f"   Attention Head 数: {best_params['nhead']}")
print(f"   FFN 维度: {best_params['dim_feedforward']}")
print(f"   Dropout: {best_params['dropout']:.4f}")
print(f"   Learning Rate: {best_params['lr']:.6f}")
print(f"   Weight Decay: {best_params['weight_decay']:.6f}")
print(f"   Batch Size: {best_params['batch_size']}")
print(f"   验证集比例: {VAL_RATIO}")
print(f"   早停耐心: {PATIENCE}")
print(f"   最佳 epoch: {best_epoch} (Val R²={best_val_r2:.4f})")
print(f"   训练集 R²: {train_r2:.4f}")
print(f"   测试集 R²: {test_r2:.4f}")

print(f"\n💡 提示: 可使用命令行参数调整超参数")
print(f"   例如: python train_transformer_use_all_features.py --optuna_trials 50 --optuna_epochs 80")