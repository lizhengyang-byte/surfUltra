# models/predictor — pCMC 预测模型管线

本目录提供了统一的 **pCMC（临界胶束浓度）** 预测管线，支持 5 个预训练模型，提供三种使用方式：**Python API**、**CLI**、**Demo 脚本**。

## 目录结构

```
models/
├── predictor/
│   ├── __init__.py               # 包初始化
│   ├── API_predictor.py          # Python 程序化 API（推荐）
│   ├── demo.py                   # 快速 Demo 脚本
│   ├── predict.py                # CLI 入口（批量预测）
│   ├── featurizer.py             # 分子特征化（522 维 / 209 维）
│   ├── model_loader.py           # 5 个预训练模型的统一加载器
│   ├── pharmhgt_model.py         # PharmHGT 异构图 Transformer 模型定义
│   ├── retrain_catboost_all.py   # 用全部 RDKit 描述符重新训练 CatBoost
│   ├── input.csv.csv             # 140 条 SMILES 测试数据（含真实 pCMC 值）
│   └── weights/                  # 训练好的模型权重文件存放目录
│       ├── catboost_pharmhgt_model.pkl
│       ├── xgboost_pharmhgt_model.pkl
│       ├── lightgbm_pharmhgt_model.pkl
│       ├── pharmhgt_best_model.pth
│       └── catboost_all_features_model.pkl
└── test_data/
    └── surfpro_test_predictions.csv  # 示例预测结果
```

## 支持的模型

| 模型名称 | 特征类型 | 特征维度 | 说明 |
|---------|---------|---------|------|
| `catboost_pharmhgt` | PharmHGT 522 维 | 522 | CatBoost + Optuna 调参 |
| `xgboost_pharmhgt` | PharmHGT 522 维 | 522 | XGBoost + Optuna 调参 |
| `lightgbm_pharmhgt` | PharmHGT 522 维 | 522 | LightGBM + Optuna 调参 |
| `pharmhgt_gnn` | GNN（图神经网络） | — | PharmHGT 异构图 Transformer |
| `catboost_all` | 全部 RDKit 描述符 | ~209 | CatBoost + Optuna 调参 |

## 快速开始

```bash
# 方式一：Demo 脚本（推荐快速测试）
python models/predictor/demo.py

# 方式二：CLI 批量预测
python models/predictor/predict.py models/predictor/input.csv.csv results.csv

# 方式三：查看可用模型
python models/predictor/predict.py --list-models
```

---

## 1. Python API 使用（API_predictor.py 推荐）

最高层级的接口，一行代码完成预测。内部自动处理模型加载、特征化、聚合和保存。

### 基本用法

```python
from models.predictor.API_predictor import predictor

# 使用全部 5 个模型做集成预测（取均值）
df = predictor("input.csv", "output.csv", "all")

# 指定单个模型 + GPU
df = predictor("input.csv", "output.csv", "catboost_pharmhgt", device="cuda")

# 自定义 SMILES 列名
df = predictor("input.csv", "out.csv", "all", smiles_col="smiles_column")
```

### API 函数签名

```python
def predictor(input_csv, output_csv, model_name, smiles_col='SMILES', device=None)
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input_csv` | 输入 CSV 文件路径 | 必填 |
| `output_csv` | 输出 CSV 文件路径 | 必填 |
| `model_name` | 模型名称（参见上表，或用 `"all"` 使用全部） | 必填 |
| `smiles_col` | SMILES 列名 | `SMILES` |
| `device` | Torch 设备（`'cpu'` / `'cuda'` / `None` 自动检测） | `None` → 自动 |

### 返回值

返回追加了预测列的 `pd.DataFrame`：

| 模式 | 输出列 |
|------|--------|
| 单个模型 | `predicted_pCMC` + `predicted_pCMC_{model_name}` |
| `model_name='all'` | 每个模型一列 `predicted_pCMC_{name}`，再加 `predicted_pCMC`（均值） |

无效 SMILES 对应的预测值会被设为 `NaN`。

### 底层 API

```python
from models.predictor.model_loader import load_model, get_available_models

# 查看可用模型
print(get_available_models())

# 加载树模型（CatBoost / XGBoost / LightGBM）
tree_model = load_model('catboost_pharmhgt')

# 加载 GNN 模型（可指定设备）
gnn_model, params, metrics = load_model('pharmhgt_gnn', device='cuda')
```

```python
from models.predictor.featurizer import (
    build_feature_vector_pharmhgt,   # 返回 522 维向量
    smiles_to_features_all,          # 返回 ~209 维向量
)

# 单个 SMILES 转特征
vec_522 = build_feature_vector_pharmhgt("CCO")      # 522-dim
vec_209 = smiles_to_features_all("CCO")              # 209-dim
```

```python
from models.predictor.pharmhgt_model import (
    build_molecule_data,           # SMILES → PyG HeteroData
    predict_pharmhgt,              # 单分子预测
    predict_pharmhgt_batch,        # 批量预测
)

# 单分子预测
pred = predict_pharmhgt(gnn_model, "CCO")

# 批量预测
smiles_list = ["CCO", "CCCO", "C(C)(C)O"]
preds = predict_pharmhgt_batch(gnn_model, smiles_list)
```

---

## 2. Demo 脚本（demo.py）

快速验证管线是否正常工作的入口脚本，展示三种典型调用方式：

```bash
python models/predictor/demo.py
```

```python
# demo.py 内部逻辑：
input_path = Path(__file__).parent / "input.csv.csv"

# 1. 全部模型集成预测
predictor(str(input_path), "output_all.csv", "all")

# 2. 单个模型预测
predictor(str(input_path), "output_catboost_pharmhgt.csv",
          "catboost_pharmhgt", device="cpu")

# 3. 自定义 SMILES 列名
predictor(str(input_path), "out_all_default_SMILES_column.csv",
          "all", smiles_col="SMILES")
```

**注意**：`demo.py` 路径采用了 `Path(__file__).parent` 定位文件，因此无论从项目根目录还是直接传路径执行都能正确找到 `input.csv.csv`。

---

## 3. CLI 使用（predict.py）

### 基本用法

```bash
# 使用所有可用模型预测
python models/predictor/predict.py data/surfpro_test.csv results.csv

# 指定单一模型
python models/predictor/predict.py data.csv out.csv --model catboost_pharmhgt

# 指定多个模型
python models/predictor/predict.py data.csv out.csv -m catboost_pharmhgt -m lightgbm_pharmhgt

# 使用 -m all 等价于使用所有可用模型
python models/predictor/predict.py data.csv out.csv -m all

# 查看可用模型列表
python models/predictor/predict.py --list-models
```

### 全部参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `input_csv` | — | 输入 CSV 文件路径 | 必填 |
| `output_csv` | — | 输出 CSV 文件路径 | `input_predictions.csv` |
| `--model` | `-m` | 使用的模型（可重复指定，或 `all`） | `all` |
| `--smiles-col` | `-s` | SMILES 列名 | `SMILES` |
| `--device` | `-d` | Torch 设备（`cpu` / `cuda`） | 自动检测 |
| `--batch-size` | `-b` | GNN 推理批次大小 | `64` |
| `--list-models` | — | 列出可用模型并退出 | — |

### 输出说明

预测结果会写入输出 CSV，每列命名格式为 `predicted_pCMC_{model_name}`。如果只使用一个模型，还会额外生成 `predicted_pCMC` 简写列。

无效 SMILES 对应的预测值会被设为 `NaN`。

---

## 4. 特征化详解（featurizer.py）

### 4.1 PharmHGT 522 维特征

适用于前 4 个模型（`catboost_pharmhgt`、`xgboost_pharmhgt`、`lightgbm_pharmhgt`、`pharmhgt_gnn`）

| 组块 | 维度 | 说明 |
|------|------|------|
| 原子聚合特征 | 220 (55×4) | 每个原子 55 维特征，聚合为 mean/std/min/max |
| 键聚合特征 | 56 (14×4) | 每个键 14 维特征，聚合为 mean/std/min/max |
| 药效团特征 | 194 | MACCS 指纹（填充到 194 维） |
| 反应性特征 | 34 | BRICS 碎片类型直方图 |
| 表面活性剂类型 | 4 | one-hot（阴/阳/非/两性离子） |
| 头/尾比例 | 2 | 亲水头与疏水尾原子占比 |
| 基础分子描述符 | 12 | MW、LogP、TPSA、RotB、HBA、HBD、RingCount 等归一化值 |
| **总计** | **522** | |

### 4.2 全部 RDKit 描述符 209 维特征

适用于 `catboost_all` 模型

使用 `Descriptors.descList` 中的所有 200 多个可用描述符，对无效 SMILES 返回全零向量。

### 4.3 表面活性剂检测

`detect_surfactant()` 可自动识别表面活性剂的：
- **头基**（亲水基团）：磺酸根、硫酸根、羧基、季铵盐、吡啶等
- **尾链**（疏水烷基链）：通过 DFS 搜索最长 ≥4 个碳的碳链
- **类型**：阴离子 / 阳离子 / 非离子 / 两性离子

---

## 5. 模型结构说明

### 树模型（CatBoost / XGBoost / LightGBM）

- **输入**：522 维特征向量（PharmHGT 或全部 RDKit 描述符）
- **特征工程**：使用 Optuna 进行超参数搜索
- **输出**：pCMC 预测值（标量）
- 权重文件格式：`.pkl`（joblib 序列化）

### PharmHGT GNN（异构图 Transformer）

- **输入**：SMILES 经 `build_molecule_data()` 转换为 PyG `HeteroData`
- **节点类型**：`atom`（55 维）、`pharmacophore`（194 维）、`reaction`（34 维）
- **边类型**：化学键、药效团↔原子、反应性↔原子之间的交叉连接
- **核心模块**：
  - 原子级 GNN 消息传递层
  - 表面活性剂注意力（头基/尾链掩码引导）
  - 多视图交叉注意力（MVMP）
  - 层级化 readout
- **输出**：pCMC 预测值（标量）
- 权重文件格式：`.pth`（PyTorch checkpoint，含 `state_dict`、`params`、`metrics`）

---

## 6. 模型加载器（model_loader.py）

提供 5 个专用加载函数 + 1 个通用调度函数：

```python
load_model(model_name, weights_dir='weights/', device='cpu')
```

- `model_name` 可选值：`catboost_pharmhgt`、`xgboost_pharmhgt`、`lightgbm_pharmhgt`、`pharmhgt_gnn`、`catboost_all`
- 权重文件缺失时会抛出 `FileNotFoundError` 并提示解决方法
- 可通过 `get_available_models()` 检查哪些模型可用

### 权重文件对应关系

| 模型名 | 权重文件名 |
|--------|-----------|
| `catboost_pharmhgt` | `catboost_pharmhgt_model.pkl` |
| `xgboost_pharmhgt` | `xgboost_pharmhgt_model.pkl` |
| `lightgbm_pharmhgt` | `lightgbm_pharmhgt_model.pkl` |
| `pharmhgt_gnn` | `pharmhgt_best_model.pth` |
| `catboost_all` | `catboost_all_features_model.pkl` |

---

## 7. 重新训练缺失模型

如果 `--list-models` 显示 `catboost_all` 模型缺失，可以运行：

```bash
python models/predictor/retrain_catboost_all.py
```

该脚本会：
1. 从 `data/surfpro_imputed.csv` 加载训练数据
2. 计算所有 RDKit 描述符（~209 维）
3. 使用 Optuna 进行 50 次超参数搜索（5 折交叉验证）
4. 在完整训练集上训练最终模型
5. 评估测试集性能并输出 Top-20 特征重要性
6. 将模型保存至 `weights/catboost_all_features_model.pkl`

---

## 8. 依赖项

- `rdkit` — 分子处理与特征化
- `torch` + `torch_geometric` — PharmHGT GNN
- `catboost` / `xgboost` / `lightgbm` — 树模型
- `joblib` — 模型序列化
- `numpy` / `pandas` — 数据处理
- `optuna` — 超参数优化
- `scikit-learn` — 评估指标与交叉验证

可通过项目根目录的 `pyproject.toml` 安装全部依赖：

```bash
pip install -e .
```