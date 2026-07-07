"""
Transformer 模型 + Word2Vec 分子特征化 → pCMC 预测

工作流程：
  1. 用 Word2VecFeaturizer 为训练集 SMILES 训练 Word2Vec 词向量
  2. 建立映射表（token → 整数 ID），构建词汇表
  3. 将 SMILES 分词后映射为 token ID 序列（填充/截断至 max_len）
  4. 使用 nn.TransformerEncoder 处理序列，输出分子级特征向量
  5. 通过回归头预测 pCMC
  6. 用默认超参数训练最终模型

用法：
  python train_transformer_use_Word2Vec.py              # 默认参数训练
  python train_transformer_use_Word2Vec.py --epochs 200  # 指定训练轮数
  python train_transformer_use_Word2Vec.py --dim 128     # 指定嵌入维度
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import warnings
from collections import Counter
from typing import List, Optional
from tqdm import tqdm

from smiles_to_features_Word2Vec import tokenize_smiles, Word2VecFeaturizer

warnings.filterwarnings("ignore")

# ========== 命令行参数 ==========
parser = argparse.ArgumentParser(description="Transformer + Word2Vec 分子性质预测")
parser.add_argument("--epochs", type=int, default=200,
                    help="训练轮数（默认: 200）")
parser.add_argument("--dim", type=int, default=64,
                    help="Word2Vec/Transformer 嵌入维度（默认: 64）")
parser.add_argument("--max_len", type=int, default=128,
                    help="最大序列长度（默认: 128）")
parser.add_argument("--seed", type=int, default=42,
                    help="随机种子（默认: 42）")
parser.add_argument("--lr", type=float, default=1e-3,
                    help="学习率（默认: 0.001）")
parser.add_argument("--dropout", type=float, default=0.2,
                    help="Dropout 比率（默认: 0.2）")
parser.add_argument("--nhead", type=int, default=4,
                    help="Attention head 数（默认: 4）")
parser.add_argument("--num_layers", type=int, default=3,
                    help="Transformer 层数（默认: 3）")
parser.add_argument("--dim_feedforward", type=int, default=256,
                    help="FFN 维度（默认: 256）")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size（默认: 32）")
parser.add_argument("--weight_decay", type=float, default=1e-4,
                    help="权重衰减（默认: 0.0001）")
args = parser.parse_args()

# ========== 配置 ==========
MAX_SEQ_LEN = args.max_len
W2V_EMBEDDING_DIM = args.dim
RANDOM_SEED = args.seed
TOTAL_EPOCHS = args.epochs

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

# ========== 1. 训练 Word2Vec + 构建词汇表 ==========
print("\n" + "=" * 60)
print("步骤 1: 训练 Word2Vec + 构建词汇表")
print("=" * 60)

featurizer = Word2VecFeaturizer(
    smiles_list=train_smiles,
    embedding_dim=W2V_EMBEDDING_DIM,
    window=5,
    min_count=2,
    workers=4,
    sg=0,  # CBOW
)
featurizer.train()

# 构建词汇表（token → id）
# 特殊 token: PAD=0, UNK=1
all_tokens: List[str] = []
for smi in tqdm(train_smiles, desc="收集 token"):
    all_tokens.extend(tokenize_smiles(smi))

token_counts = Counter(all_tokens)
vocab_tokens = [t for t, c in token_counts.items() if c >= 2]
vocab = {t: i + 2 for i, t in enumerate(sorted(vocab_tokens))}  # 0=PAD, 1=UNK
vocab_size = len(vocab) + 2

print(f"词汇表大小: {vocab_size} (含 PAD=0, UNK=1)")
print(f"实际 token 数: {len(vocab_tokens)}")

# 构建 embedding 矩阵：用 Word2Vec 向量初始化
print("构建 Embedding 矩阵...")
embedding_matrix = np.zeros((vocab_size, W2V_EMBEDDING_DIM), dtype=np.float32)
for token, idx in tqdm(vocab.items(), desc="填充 Embedding"):
    if token in featurizer.model.wv:
        embedding_matrix[idx] = featurizer.model.wv[token]

print(f"Embedding 矩阵形状: {embedding_matrix.shape}")

# ========== 2. SMILES → token ID 序列 ==========
print("\n" + "=" * 60)
print("步骤 2: SMILES → token ID 序列")
print("=" * 60)


def smiles_to_ids(smiles: str) -> np.ndarray:
    """将 SMILES 转为固定长度的 token ID 序列（填充/截断至 MAX_SEQ_LEN）"""
    tokens = tokenize_smiles(smiles)
    ids = [vocab.get(t, 1) for t in tokens]  # UNK=1
    if len(ids) > MAX_SEQ_LEN:
        ids = ids[:MAX_SEQ_LEN]
    ids = ids + [0] * (MAX_SEQ_LEN - len(ids))
    return np.array(ids, dtype=np.int64)


print("转换训练集 SMILES → token ID...")
X_train_ids = np.array([smiles_to_ids(smi) for smi in tqdm(train_smiles, desc="训练集")])
print("转换测试集 SMILES → token ID...")
X_test_ids  = np.array([smiles_to_ids(smi) for smi in tqdm(test_smiles, desc="测试集")])

print(f"训练集序列矩阵: {X_train_ids.shape}")
print(f"测试集序列矩阵: {X_test_ids.shape}")

# ========== 3. 定义 Transformer 模型 ==========
print("\n" + "=" * 60)
print("步骤 3: 定义 Transformer 模型")
print("=" * 60)


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

    架构：
      - Token Embedding（冻结 Word2Vec 预训练权重）
      - Positional Encoding
      - Transformer Encoder (N 层)
      - Mean Pooling（对所有 token 做平均池化）
      - LayerNorm → Dropout → 回归头
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        pretrained_embeddings: np.ndarray,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        freeze_embeddings: bool = True,
    ):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(pretrained_embeddings),
            freeze=freeze_embeddings,
            padding_idx=0,
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len=MAX_SEQ_LEN)

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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch, seq_len), mask: (batch, seq_len) True=有效
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = self.pos_encoder(x)

        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask  # True=忽略(PAD)

        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Mean Pooling（只对非 PAD 位置）
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        x = self.norm(x)
        x = self.dropout(x)
        x = self.regressor(x).squeeze(-1)
        return x


# ========== 4. 定义 Dataset 和训练函数 ==========


class SMILESDataset(Dataset):
    def __init__(self, token_ids: np.ndarray, targets: np.ndarray):
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.mask = self.token_ids != 0  # True=有效 token

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.token_ids[idx], self.mask[idx], self.targets[idx]


def train_epoch(model, loader, optimizer, criterion, epoch_bar=None):
    model.train()
    total_loss = 0
    for token_ids, mask, targets in loader:
        token_ids, mask, targets = (
            token_ids.to(device), mask.to(device), targets.to(device),
        )
        optimizer.zero_grad()
        pred = model(token_ids, mask)
        loss = criterion(pred, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item() * token_ids.size(0)
        if epoch_bar is not None:
            epoch_bar.update(token_ids.size(0))
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, y_true, y_pred = 0, [], []
    for token_ids, mask, targets in loader:
        token_ids, mask, targets = (
            token_ids.to(device), mask.to(device), targets.to(device),
        )
        pred = model(token_ids, mask)
        total_loss += criterion(pred, targets).item() * token_ids.size(0)
        y_true.append(targets.cpu())
        y_pred.append(pred.cpu())
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    return (
        total_loss / len(loader.dataset),
        np.sqrt(mean_squared_error(y_true, y_pred)),
        mean_absolute_error(y_true, y_pred),
        r2_score(y_true, y_pred),
    )


# ========== 5. 训练最终模型 ==========
print("\n" + "=" * 60)
print("步骤 4: 在全部训练集上训练 Transformer 模型")
print("=" * 60)

print(f"超参数:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

model = TransformerRegressor(
    vocab_size=vocab_size,
    d_model=W2V_EMBEDDING_DIM,
    pretrained_embeddings=embedding_matrix,
    nhead=best_params["nhead"],
    num_layers=best_params["num_layers"],
    dim_feedforward=best_params["dim_feedforward"],
    dropout=best_params["dropout"],
    freeze_embeddings=True,
).to(device)

train_loader = DataLoader(
    SMILESDataset(X_train_ids, y_train_full),
    batch_size=best_params["batch_size"],
    shuffle=True,
)
test_loader = DataLoader(
    SMILESDataset(X_test_ids, y_test_orig),
    batch_size=best_params["batch_size"],
    shuffle=False,
)

optimizer = optim.AdamW(
    model.parameters(),
    lr=best_params["lr"],
    weight_decay=best_params["weight_decay"],
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6
)
criterion = nn.MSELoss()

# 计算每个 epoch 的 batch 总数，用于进度条的单位
batches_per_epoch = len(train_loader)
samples_per_epoch = len(train_loader.dataset)

epoch_iter = range(TOTAL_EPOCHS)
with tqdm(epoch_iter, desc="训练进度", unit="epoch", ncols=100) as pbar:
    for epoch in pbar:
        # 用子进度条显示每个 epoch 内的 batch 进度
        with tqdm(
            total=samples_per_epoch, desc=f"Epoch {epoch+1:3d}/{TOTAL_EPOCHS}",
            unit="样本", leave=False, ncols=80,
        ) as epoch_bar:
            train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch_bar)

        scheduler.step()

        if (epoch + 1) % 50 == 0:
            val_loss, val_rmse, val_mae, val_r2 = evaluate(model, test_loader, criterion)
            pbar.set_postfix({
                "Loss": f"{train_loss:.4f}",
                "Test_RMSE": f"{val_rmse:.4f}",
                "Test_R²": f"{val_r2:.4f}",
            })
        else:
            pbar.set_postfix({"Loss": f"{train_loss:.4f}"})

# ========== 6. 最终评估 ==========
print(f"\n{'Model':<35} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print("-" * 62)

with torch.no_grad():
    model.eval()

    # 训练集预测
    train_preds = []
    for token_ids, mask, _ in train_loader:
        token_ids, mask = token_ids.to(device), mask.to(device)
        pred = model(token_ids, mask)
        train_preds.append(pred.cpu().numpy())
    train_preds = np.concatenate(train_preds)

    train_rmse = np.sqrt(mean_squared_error(y_train_full, train_preds))
    train_mae = mean_absolute_error(y_train_full, train_preds)
    train_r2 = r2_score(y_train_full, train_preds)
    print(f"{'Transformer + Word2Vec (Train)':<35} {train_rmse:>8.4f} {train_mae:>8.4f} {train_r2:>8.4f}")

    # 测试集预测
    test_preds = []
    for token_ids, mask, _ in test_loader:
        token_ids, mask = token_ids.to(device), mask.to(device)
        pred = model(token_ids, mask)
        test_preds.append(pred.cpu().numpy())
    test_preds = np.concatenate(test_preds)

    test_rmse = np.sqrt(mean_squared_error(y_test_orig, test_preds))
    test_mae = mean_absolute_error(y_test_orig, test_preds)
    test_r2 = r2_score(y_test_orig, test_preds)
    print(f"{'Transformer + Word2Vec (Test)':<35} {test_rmse:>8.4f} {test_mae:>8.4f} {test_r2:>8.4f}")

print("-" * 62)

# ========== 7. 模型对比 ==========
print(f"\n📊 模型对比:")
print(f"{'Model':<35} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print("-" * 62)
print(f"{'SVR (原最佳)':<35} {'0.6826':>8} {'0.4436':>8} {'0.6227':>8}")
print(f"{'Transformer + Word2Vec (Test)':<35} {test_rmse:>8.4f} {test_mae:>8.4f} {test_r2:>8.4f}")

print(f"\n✅ 最终模型参数:")
print(f"   词汇表大小: {vocab_size}")
print(f"   Word2Vec 维度 (d_model): {W2V_EMBEDDING_DIM}")
print(f"   最大序列长度: {MAX_SEQ_LEN}")
print(f"   Transformer 层数: {best_params['num_layers']}")
print(f"   Attention Head 数: {best_params['nhead']}")
print(f"   FFN 维度: {best_params['dim_feedforward']}")
print(f"   Dropout: {best_params['dropout']:.4f}")
print(f"   Learning Rate: {best_params['lr']:.6f}")
print(f"   Batch Size: {best_params['batch_size']}")
print(f"   训练集 R²: {train_r2:.4f}")
print(f"   测试集 R²: {test_r2:.4f}")

print(f"\n💡 提示: 可使用命令行参数调整超参数")
print(f"   例如: python train_transformer_use_Word2Vec.py --epochs 500 --dim 128 --lr 5e-4")