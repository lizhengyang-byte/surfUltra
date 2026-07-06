# SurfPredict — 表面活性剂性质预测

利用分子结构（SMILES）预测表面活性剂（surfactant）的界面性质，涵盖传统机器学习与图神经网络多种建模策略。

## 目标变量

| 变量 | 说明 | 缺失率 |
|---|---|---|
| **pCMC** | log CMC（临界胶束浓度，对数） | ~9.8% |
| AW_ST_CMC | 表面张力 at CMC | ~53.8% |
| Gamma_max | 最大表面过剩 | ~57.8% |
| Area_min | 最小分子面积 | ~57.8% |
| Pi_CMC | 表面压 at CMC | ~38.3% |
| pC20 | 表面活性效率 | ~32.5% |

> **pCMC 为首选目标** —— 缺失率最低、近似正态分布；与 pC20 高度相关 (r=0.77)。

## 数据

| 文件 | 行数 | 说明 |
|---|---|---|
| `data/surfpro_train.csv` | 1335 | 训练集 |
| `data/surfpro_test.csv` | 140 | 测试集 |
| `data/surfpro_literature.csv` | 1503 | 文献数据 |
| `data/surfpro_imputed.csv` | 1476 | 插补后完整数据 |

每条记录包含 SMILES、表面活性剂类型（阴/阳/非离子）、温度、6 个目标变量及交叉验证折标记。

## 建模策略

### 特征化方案

| 方式 | 维度 | 对应模块 |
|---|---|---|
| **RDKit 精选描述符** | ~62 | `smiles_to_features.py` — 分子量、LogP、TPSA、电荷、拓扑、VSA 分布等 |
| **RDKit 全部描述符** | ~217 | `smiles_to_features_all.py` — RDKit 所有可用描述符 |
| **Morgan 圆形指纹 (ECFP4)** | 2048 | `001.py` / `002.py` 内置 `scikit-mol` 转换器 |
| **AttentiveFP 分子图** | 原子 39 维 + 键 11 维 | `train_gnn.py` 内置 `smiles_to_graph()` |

### 模型一览

| 脚本 | 特征 | 模型 | 框架 | 超参数搜索 |
| --- | --- | --- | --- | --- |
| `train_xgboost.py` | RDKit 描述符 (~62) | XGBoost | xgboost | Optuna 60 trials |
| `train_LightGBM.py` | RDKit 描述符 (~62) | LightGBM | lightgbm | Optuna 60 trials |
| `train_SVR.py` | RDKit 描述符 (~62) | SVR (RBF/Poly/Sigmoid) | scikit-learn | Optuna 60 trials |
| `train_rnn.py` | RDKit 描述符 (~62) | MLP (Dense+BN+Dropout) | TensorFlow/Keras | Optuna 30 trials |
| `train_mlp.py` | RDKit 描述符 (~62) | MLP (PyTorch) | PyTorch | Optuna 30 trials |
| `train_mlp_use_all_features.py` | RDKit 全部描述符 (~217) | MLP (PyTorch) | PyTorch | Optuna 30 trials |
| `train_rnn_use_all_features.py` | RDKit 全部描述符 (~217) | LSTM (PyTorch) | PyTorch | Optuna 30 trials |
| `train_gnn.py` | 分子图 (原子 39+键 11) | **AttentiveFP** (GNN) | PyTorch Geometric | Optuna 30 trials |
| `001.py` | Morgan 指纹 (2048) | Ridge（快速基线） | scikit-mol | — |
| `002.py` | Morgan 指纹 (2048) | 多模型基准 (Ridge, RF, GBR, SVR, KNN, XGB, LGB) | scikit-mol + sklearn | — |

### 当前最佳结果（pCMC）

| 模型 | 特征 | Test R² | 来源 |
|------|------|---------|------|
| **MLP (PyTorch)** | 全部 RDKit 描述符 (~217) | **0.8650** | `train_mlp_use_all_features.py` |
| **LSTM (PyTorch)** | 全部 RDKit 描述符 (~217) | *待运行* | `train_rnn_use_all_features.py` |
| MLP (PyTorch) | RDKit 精选描述符 (~62) | 0.8399 | `train_mlp.py` |
| MLP (Keras) | RDKit 精选描述符 (~62) | 0.8399 | `train_rnn.py` |
| XGBoost | RDKit 精选描述符 (~62) | 0.8401 (Val) | `train_xgboost.py` |
| SVR (RBF) | Morgan 指纹 (2048) | 0.6227 | `002.py` |

## 快速开始

```bash
# 运行任意训练脚本
python train_mlp.py

# 各脚本会自动完成：数据加载 → 特征化 → 划分 → 基线训练
# → Optuna 搜索 → 最终训练 → 评估 → 保存预测图至 reports/
```

所有依赖已预装：RDKit 2026.3, PyTorch 2.6, PyG 2.8, TensorFlow 2.21, XGBoost 3.3, LightGBM 4.6, Optuna 4.9, scikit-learn, pandas 3.0, matplotlib, seaborn。

## 输出

预测-真实值散点图保存至 `reports/` 目录，文件名如 `gnn_pred_vs_true.png`。示例（AttentiveFP GNN 训练日志）：

```
[Train] RMSE: 0.2800, R²: 0.9321
[Val]   RMSE: 0.4878, R²: 0.8011
[Test]  RMSE: 0.4474, R²: 0.8399
```

## 项目结构

```
├── data/                       # CSV 数据文件
│   ├── surfpro_train.csv       # 训练集 (1335)
│   ├── surfpro_test.csv        # 测试集 (140)
│   ├── surfpro_literature.csv  # 文献数据 (1503)
│   └── surfpro_imputed.csv     # 插补后完整数据 (1476)
├── reports/                    # 预测图 + 日志
├── smiles_to_features.py       # RDKit 精选描述符提取 (~62 维)
├── smiles_to_features_all.py   # RDKit 全部描述符提取 (~217 维)
├── train_xgboost.py            # XGBoost + Optuna
├── train_LightGBM.py           # LightGBM + Optuna
├── train_SVR.py                # SVR + Optuna
├── train_rnn.py                # Keras MLP (62维) + Optuna
├── train_mlp.py                # PyTorch MLP (62维) + Optuna
├── train_mlp_use_all_features.py  # PyTorch MLP (217维全描述符) + Optuna
├── train_rnn_use_all_features.py  # PyTorch LSTM (217维全描述符) + Optuna
├── train_gnn.py                # AttentiveFP GNN + Optuna
├── 001.py                      # Morgan 指纹 + Ridge 快速基线
└── 002.py                      # Morgan 指纹多模型 benchmark
```

> **注意：** 没有单元测试，各脚本独立运行（无共享模块），图数据在 `train_gnn.py` 中即时转换、无预存 `.pt` 文件。
