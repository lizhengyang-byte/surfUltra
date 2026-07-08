# SurfPredict 模型训练报告

> **目标变量:** pCMC（临界胶束浓度对数值）  
> **数据集:** 1335 条训练样本，1204 条有效（含 pCMC 标签）  
> **特征:** 62 维精选 / 217 维 RDKit 分子描述符 / 1415 维 RDKit+MACCS+ECFP4+Aux（XGBoost 经特征选择保留 109 维）  

---

## 1. 模型概览与性能对比

### 1.1 测试集主指标对比

| 模型 | 框架 | RMSE ↓ | MAE ↓ | R² ↑ | 调优 Trial |
|------|------|--------|-------|------|-----------|
| **CatBoost (全描述符, Optuna)** | CatBoost | **0.3356** | **0.2221** | **0.9088** | 50 |
| **LightGBM (全描述符, Optuna)** | LightGBM | **0.3525** | **0.2391** | **0.8994** | 50 |
| **LightGBM (Advanced: RDKit+MACCS+ECFP4, Optuna)** | LightGBM | 0.3698 | 0.2459 | **0.8893** | 50 |
| **MLP (全描述符)** | PyTorch | 0.4083 | 0.2525 | 0.8650 | 30 |
| **LightGBM (全描述符, 手动)** | LightGBM | 0.4179 | 0.2742 | 0.8586 | 手动调参 |
| **XGBoost (全描述符, 特征选择)** | XGBoost | 0.4053 | 0.2736 | **0.8670** | 100 |
| **MLP (Keras, 62维)** | TensorFlow/Keras | 0.4474 | 0.3294 | 0.8399 | 30 |
| **MLP (PyTorch, 62维)** | PyTorch | 0.4517 | 0.3231 | 0.8369 | 30 |
| **RNN (PyTorch LSTM, 全描述符)** | PyTorch | 0.4610 | 0.3152 | 0.8279 | 30 |
| **RNN (Keras, 62维)** | Keras | 0.4849 | 0.3646 | 0.8120 | 30 |
| **Transformer + Word2Vec (SMILES 序列, Optuna)** | PyTorch | 0.5083 | 0.3492 | **0.7907** | 25 |
| **Transformer + RDKit (全描述符, Optuna)** | PyTorch | 0.6796 | 0.5033 | **0.6261** | 25 |
| **Ridge (全描述符, StandardScaler, Optuna)** | scikit-learn | 0.6899 (Val) | 0.5226 | 0.6297 (Val) | 60 |
| **PCA+OLS (全描述符, 39 主成分)** | scikit-learn | 0.7707 (Val) | 0.5948 | 0.5378 (Val) | — |
| **OLS (全描述符, StandardScaler)** | scikit-learn | ~1.1337 (Val) | ~0.8994 | ~0.0000 (Val) | — |
| **SVR (RBF)** | scikit-learn | 0.5274 (Val) | 0.3978 | 0.7835 (Val) | 60 |
| **LightGBM (62维, 调优)** | LightGBM | — | — | 0.4035 ± 0.1774 (CV) | 60 |
| **AttentiveFP (GNN)** | PyG | *待运行* | *待运行* | *待运行* | 30 |

> **注:** 除 SVR 仅输出验证集指标外，其余模型均报告测试集结果。  
> "CatBoost (全描述符)"、"MLP (全描述符)"、"LightGBM (全描述符)"、"XGBoost (全描述符, 特征选择)"、"RNN (PyTorch LSTM, 全描述符)" 和 "Transformer + RDKit (全描述符)" 使用全部 217 维 RDKit 描述符（XGBoost 进一步经特征选择保留 109 维）；"LightGBM (Advanced)" 使用 1415 维（RDKit 217 + MACCS 166 + ECFP4 1024 + Aux 7）；其余模型基于 62 维精选描述符。

### 1.2 详细分集指标

| 模型 | 数据集 | RMSE | MAE | R² |
|------|--------|------|-----|----|
| **CatBoost (全描述符, Optuna)** | Train (all) | 0.0377 | 0.0243 | 0.9988 |
| | Test | **0.3356** | **0.2221** | **0.9088** |
| | CV (5-fold) | — | — | 0.3827 ± 0.1772 |
| **LightGBM (全描述符, Optuna)** | Train (all) | 0.1563 | 0.1037 | 0.9792 |
| | Test | **0.3525** | **0.2391** | **0.8994** |
| | CV (5-fold) | — | — | 0.4042 ± 0.2155 |
| **LightGBM (Advanced: RDKit+MACCS+ECFP4, Optuna)** | Train (all) | 0.1270 | 0.0877 | 0.9863 |
| | Test (单一模型) | 0.3698 | 0.2459 | 0.8893 |
| | CV Test (5-Fold 集成) | 0.3976 | 0.2744 | 0.8720 |
| | CV R² (变换空间) | — | — | 0.3840 ± 0.2374 |
| **XGBoost (全描述符, 特征选择)** | Train (all) | 0.2006 | 0.1473 | 0.9658 |
| | Test | **0.4053** | **0.2736** | **0.8670** |
| | CV (5-fold) | — | — | 0.4409 ± 0.1887 |
| **XGBoost** (旧版, 62维) | Train | 0.1051 | 0.0771 | 0.9904 |
| | Val | 0.4534 | 0.3095 | 0.8401 |
| **MLP (全描述符)** | Train | 0.1165 | 0.0726 | 0.9885 |
| | Test | 0.4083 | 0.2525 | 0.8650 |
| **LightGBM (全描述符, 手动)** | Train | 0.2089 | 0.1528 | 0.9620 |
| | Val | 0.4549 | 0.3185 | 0.8390 |
| | Test | 0.4179 | 0.2742 | 0.8586 |
| | CV (5-fold) | — | — | 0.3859 ± 0.2227 |
| **RNN (PyTorch LSTM, 全描述符)** | Train | 0.1306 | 0.0859 | 0.9855 |
| | Test | 0.4610 | 0.3152 | 0.8279 |
| **LightGBM (62维, 调优)** | CV (5-fold) | — | — | 0.4035 ± 0.1774 |
| **Transformer + Word2Vec (SMILES 序列, Optuna)** | Train | 0.2808 | 0.1917 | 0.9329 |
| | Test | **0.5083** | **0.3492** | **0.7907** |
| | Val (15% holdout) | — | — | 0.7928 (best) |
| **Transformer + RDKit (全描述符, Optuna)** | Train (all) | 0.5217 | 0.3870 | 0.7685 |
| | Test | **0.6796** | **0.5033** | **0.6261** |
| | Val (15% holdout) | — | — | 0.6672 (best) |
| **SVR (RBF)** (tuned) | Train | 0.4601 | 0.3492 | 0.8156 |
| | Val | 0.5274 | 0.3978 | 0.7835 |
| | CV (5-fold) | — | — | 0.3468 ± 0.1345 |
| **MLP (PyTorch, 62维)** | Train | 0.2742 | 0.1982 | 0.9348 |
| | Val | 0.5182 | 0.3382 | 0.7756 |
| | Test | 0.4517 | 0.3231 | 0.8369 |
| **MLP (Keras, 62维)** | Train | 0.2800 | 0.2059 | 0.9321 |
| | Val | 0.4878 | 0.3356 | 0.8011 |
| | Test | 0.4474 | 0.3294 | 0.8399 |
| **RNN (Keras, 62维)** | Train | 0.3594 | 0.2628 | 0.8880 |
| | Val | 0.5444 | 0.3788 | 0.7523 |
| | Test | 0.4849 | 0.3646 | 0.8120 |
| **Ridge (全描述符, StandardScaler, Optuna)** | Train | 0.6440 | 0.4939 | 0.6389 |
| | Val | **0.6899** | **0.5226** | **0.6297** |
| | CV (5-fold) | — | — | 0.3022 ± 0.1365 |
| **PCA+OLS (全描述符, 39 主成分)** | Train | 0.7306 | 0.5657 | 0.5352 |
| | Val | **0.7707** | **0.5948** | **0.5378** |
| | CV (5-fold) | — | — | -2.1525 ± 1.7663 |
| **OLS (全描述符, StandardScaler)** | Val | ~1.1337 | ~0.8994 | ~0.0000 |

---

## 2. 各模型详情

### 2.1 XGBoost (全描述符, 特征选择 + Optuna)

- **脚本:** `train_xgboost_use_all_features.py`
- **特征:** 全部 217 维 RDKit 分子描述符 → StandardScaler 标准化 → 特征选择（保留重要性≥中位数的 109 维）
- **超参数搜索:** Optuna TPE, 100 trials, 5-Fold CV R² 最大化
- **搜索空间（11 个参数）:**
  ```json
  {
    "max_depth": [3, 8],
    "min_child_weight": [1.0, 50.0] (log),
    "gamma": [0.0, 10.0],
    "learning_rate": [0.003, 0.3] (log),
    "n_estimators": [100, 2000],
    "subsample": [0.4, 1.0],
    "colsample_bytree": [0.3, 1.0],
    "colsample_bylevel": [0.3, 1.0],
    "reg_lambda": [0.1, 50.0] (log),
    "reg_alpha": [0.1, 50.0] (log),
    "max_delta_step": [0.0, 10.0]
  }
  ```
- **最佳参数:**
  ```json
  {
    "max_depth": 14,
    "min_child_weight": 2.56,
    "gamma": 0.296,
    "learning_rate": 0.0241,
    "n_estimators": 725,
    "subsample": 0.613,
    "colsample_bytree": 0.468,
    "colsample_bylevel": 0.494,
    "reg_lambda": 0.0139,
    "reg_alpha": 0.0358,
    "max_delta_step": 0.0
  }
  ```
- **最佳 CV R² (5-fold):** 0.4390
- **CV R² (5-fold, tuned):** 0.4409 ± 0.1887
- **参数重要性 Top 5:** reg_alpha (0.422), gamma (0.395), colsample_bylevel (0.045), learning_rate (0.037), max_depth (0.029)
- **训练集（全部数据）:** RMSE=0.2006, MAE=0.1473, R²=0.9658
- **测试集:** RMSE=0.4053, MAE=0.2736, **R²=0.8670**
- **分析:** 使用特征选择将 217 维全量描述符压缩至 109 维后，测试 R²=0.8670，较旧版 62 维精选版本（Val R²=0.8401）提升约 0.027，但略低于 MLP 全描述符（0.8650）。训练集 R²=0.9658 表明仍存在过拟合，但较旧版（Train R²=0.9904）已有明显改善。Optuna 参数重要性分析显示 reg_alpha（0.422）和 gamma（0.395）是最关键的调参方向，说明正则化强度对控制 XGBoost 过拟合至关重要。总体而言，XGBoost 全描述符版本达到了与 MLP 全描述符相当的测试性能，但仍不及 LightGBM 全描述符（0.8994）。对比旧版（62维, Val R²=0.8401），新版在全量描述符上取得了更好的泛化性能。

### 2.2 LightGBM

#### 2.2.1 LightGBM (全描述符, Optuna 调优)

- **脚本:** `train_lightgbm_use_all_features.py`
- **特征:** 全部 217 维 RDKit 分子描述符，经 StandardScaler 标准化
- **超参数搜索:** Optuna TPE, 50 trials, 5-Fold CV RMSE 最小化
- **搜索空间（18 个参数）:**
  ```json
  {
    "boosting_type": ["gbdt", "dart"],
    "max_depth": [3, 15],
    "num_leaves": [15, 255],
    "learning_rate": [0.001, 0.3] (log),
    "n_estimators": [500, 3000],
    "subsample": [0.5, 1.0],
    "subsample_freq": [1, 10],
    "colsample_bytree": [0.3, 1.0],
    "feature_fraction": [0.3, 1.0],
    "feature_fraction_bynode": [0.3, 1.0],
    "reg_lambda": [0.0, 30.0],
    "reg_alpha": [0.0, 30.0],
    "min_child_weight": [0.01, 50.0] (log),
    "min_child_samples": [1, 50],
    "min_data_in_leaf": [1, 100],
    "min_split_gain": [0.0, 1.0]
  }
  ```
- **最佳超参数:**
  ```json
  {
    "boosting_type": "gbdt",
    "max_depth": 12,
    "num_leaves": 26,
    "learning_rate": 0.0143,
    "n_estimators": 2530,
    "subsample": 0.502,
    "subsample_freq": 5,
    "colsample_bytree": 0.799,
    "feature_fraction": 0.863,
    "feature_fraction_bynode": 0.552,
    "reg_lambda": 17.12,
    "reg_alpha": 0.335,
    "min_child_weight": 1.56,
    "min_child_samples": 11,
    "min_data_in_leaf": 27,
    "min_split_gain": 0.005
  }
  ```
- **最佳 CV RMSE:** 0.4500（5折平均）
- **训练集（全部数据）:** RMSE=0.1563, MAE=0.1037, R²=0.9792
- **测试集:** RMSE=0.3525, MAE=0.2391, **R²=0.8994**
- **CV R² (5-fold):** 0.4042 ± 0.2155
- **参数重要性 Top 5:** boosting_type (0.337), subsample (0.251), min_split_gain (0.123), n_estimators (0.070), subsample_freq (0.048)
- **分析:** Optuna 自动选择了 **gbdt** 模式（而非 dart），采用中等深度（12）、强 L2 正则化（17.12）、保守的数据采样（0.502），达到测试 R²=**0.8994**，超越此前所有模型。相比手动调参版本（R²=0.8586），Optuna 优化后 R² 提升约 0.04。训练集 R²=0.9792，过拟合控制仍优于 MLP（0.988）和 XGBoost（0.990）。参数重要性分析表明 `boosting_type` 和 `subsample` 是最关键的超参数。

#### 2.2.2 LightGBM (全描述符, 手动调参) — 旧版本

- **脚本:** `train_lightgbm_use_all_features.py`
- **特征:** 全部 217 维 RDKit 分子描述符
- **超参数:** 手动设置
  ```json
  {
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_estimators": 500
  }
  ```
- **Early stop:** iteration 199（验证集 RMSE 停止改善）
- **训练集:** RMSE=0.2089, MAE=0.1528, R²=0.9620
- **验证集:** RMSE=0.4549, MAE=0.3185, R²=0.8390
- **测试集:** RMSE=0.4179, MAE=0.2742, **R²=0.8586**
- **CV R² (5-fold):** 0.3859 ± 0.2227（[0.503, -0.023, 0.323, 0.582, 0.544]）
- **分析:** 使用 217 维全描述符后，手动调参即达测试 R²=0.8586，大幅超越此前 62 维版本（0.40 CV）。但在 Optuna 进一步优化后（R²=0.8994），手动调参版本已被大幅超越。

#### 2.2.3 LightGBM (Advanced: RDKit+MACCS+ECFP4+Aux, Optuna)

- **脚本:** `train_lightgbm_advanced.py`
- **特征:** RDKit(217) + MACCS(166) + ECFP4(1024) + Aux(7) = **1415 维**，不进行标准化（树模型对尺度不敏感）
- **超参数搜索:** Optuna TPE, 50 trials, 5-Fold CV RMSE 最小化，使用 Yeo-Johnson 变换（λ=1.0000，即无变换）
- **搜索空间（18 个参数 + 3 个 dart 专用参数）:**
  ```json
  {
    "boosting_type": ["gbdt", "dart"],
    "max_depth": [3, 15],
    "num_leaves": [15, 255],
    "learning_rate": [0.001, 0.3] (log),
    "n_estimators": [500, 3000],
    "subsample": [0.5, 1.0],
    "subsample_freq": [1, 10],
    "colsample_bytree": [0.3, 1.0],
    "feature_fraction": [0.3, 1.0],
    "feature_fraction_bynode": [0.3, 1.0],
    "reg_lambda": [0.0, 30.0],
    "reg_alpha": [0.0, 30.0],
    "min_child_weight": [0.01, 50.0] (log),
    "min_child_samples": [1, 50],
    "min_data_in_leaf": [1, 100],
    "min_split_gain": [0.0, 1.0],
    "lambda_l1": [0.0, 10.0],
    "lambda_l2": [0.0, 10.0],
    "drop_rate": [0.0, 0.3] (dart only),
    "max_drop": [1, 50] (dart only),
    "skip_drop": [0.0, 0.5] (dart only)
  }
  ```
- **最佳超参数（Trial 9，CV RMSE=0.4616）:**
  ```json
  {
    "boosting_type": "gbdt",
    "max_depth": 14,
    "num_leaves": 232,
    "learning_rate": 0.0370,
    "n_estimators": 1347,
    "subsample": 0.6746,
    "subsample_freq": 8,
    "colsample_bytree": 0.9280,
    "feature_fraction": 0.9210,
    "feature_fraction_bynode": 0.8459,
    "reg_lambda": 19.261,
    "reg_alpha": 2.524,
    "min_child_weight": 0.0396,
    "min_child_samples": 45,
    "min_data_in_leaf": 61,
    "min_split_gain": 0.0092,
    "lambda_l1": 1.015,
    "lambda_l2": 6.635
  }
  ```
- **最佳 CV RMSE (5-fold):** 0.4616（Trial 9）
- **参数重要性 Top 5:** num_leaves (0.390), feature_fraction_bynode (0.188), boosting_type (0.172), reg_alpha (0.054), subsample (0.046)
- **训练集（全部数据）:** RMSE=0.1270, MAE=0.0877, **R²=0.9863**
- **测试集（单一模型）:** RMSE=0.3698, MAE=0.2459, **R²=0.8893**
- **CV Test (5-Fold 集成):** RMSE=0.3976, MAE=0.2744, R²=0.8720
- **CV R² (变换空间):** [0.541, -0.055, 0.322, 0.585, 0.527]，均值 0.3840 ± 0.2374
- **特征重要性 Top 5（全为 RDKit 描述符）:** RDKit_130 (1085), RDKit_19 (949), RDKit_5 (831), RDKit_22 (723), RDKit_28 (678)
- **分析:** 使用 1415 维全量特征（RDKit+MACCS+ECFP4+Aux）进行 Optuna 优化后，测试 R²=**0.8893**，略低于仅使用 217 维 RDKit 描述符的版本（0.8994），差异约 0.01。训练集 R²=0.9863 比 217 维版本（0.9792）更高，但测试集表现略低，表明 MACCS/ECFP4 指纹引入了更多噪声特征，反而轻微降低了泛化性能。特征重要性 Top 20 全部来自 RDKit 描述符，MACCS 和 ECFP4 未进入前 20，说明 RDKit 描述符已包含最关键的结构-性质关系信息。参数重要性分析显示 `num_leaves`（0.390）远超其他参数，表明叶子节点数量是控制模型复杂度最关键的因素。总体而言，使用更大的特征集合（1415 维）并未超越精简的 217 维 RDKit 描述符版本，验证了"并非特征越多越好"的原则。

#### 2.2.4 LightGBM (62维, Optuna 调优) — 旧版本

- **超参数搜索:** Optuna TPE, 60 trials
- **CV R² (5-fold, tuned):** 0.4035 ± 0.1774
- **分析:** 使用 62 维精选描述符时，CV R² 均值 0.40，方差 0.18。在部分折上表现优异（最高 0.61），但在某些折上近乎无效（最低 0.09），提示数据可能存在分层不均。切换到 217 维全描述符后表现显著提升。

### 2.3 SVR (Support Vector Regression)

- **超参数搜索:** Optuna TPE, 60 trials, 候选核: RBF / Poly / Sigmoid
- **最佳参数:**
  ```json
  {
    "kernel": "rbf",
    "C": 1.485,
    "gamma": "auto",
    "epsilon": 0.322
  }
  ```
- **最佳 CV R² (3-fold):** 0.3722
- **CV R² (5-fold, tuned):** 0.3468 ± 0.1345
- **分析:** RBF 核表现最佳，Poly/Sigmoid 核表现更差。整体 R² 约 0.35，低于树模型和神经网络，说明该任务中存在非线性关系但 RBF 核的容量不足以完全捕捉。

### 2.4 MLP (PyTorch)

- **架构:** 3 隐藏层 (512 → 128 → 64) + BatchNorm + Dropout + ReLU
- **超参数搜索:** Optuna TPE, 30 trials
- **最佳参数:**
  ```json
  {
    "lr": 0.00123,
    "dropout": 0.268,
    "wd": 0.000810,
    "h1": 512,
    "h2": 128,
    "h3": 64,
    "bs": 64
  }
  ```
- **最佳验证 MSE:** 0.2296
- **Early stop:** epoch 231
- **测试集:** RMSE=0.4517, MAE=0.3231, R²=0.8369
- **分析:** 与 XGBoost 测试表现接近（R² ≈ 0.84），但训练集 R² 为 0.93，过拟合控制优于 XGBoost（后者训练集 0.99）。

### 2.5 MLP (Keras)

- **架构:** 3 隐藏层 (512 → 128 → 64) + BatchNorm + Dropout（与 PyTorch 版本相同结构）
- **超参数搜索:** Optuna TPE, 30 trials
- **最佳参数:**
  ```json
  {
    "lr": 0.000987,
    "dropout": 0.196,
    "wd": 2.64e-05,
    "h1": 512,
    "h2": 128,
    "h3": 64,
    "bs": 32
  }
  ```
- **最佳验证 MSE:** 0.2191
- **Early stop:** epoch 190
- **测试集:** RMSE=0.4474, MAE=0.3294, R²=0.8399
- **分析:** 验证 MSE 略低于 PyTorch 版本（0.219 vs 0.230），测试集 R² 相当（0.840 vs 0.837）。两者架构一致但优化器不同（Adam vs AdamW），共同验证了 3 层 MLP 对该任务的稳健性。

### 2.6 RNN (Keras MLP, 2 层)

- **架构:** 2 隐藏层 (256 → 128) + BatchNorm + Dropout + L2 正则化
- **超参数搜索:** Optuna TPE, 30 trials
- **最佳参数:** learning_rate=0.00099, dropout=0.196, l2_reg=2.64e-05, units_1=256, units_2=128, batch_size=32
- **测试集:** RMSE=0.4849, MAE=0.3646, R²=0.8120
- **分析:** 2 层架构相比于 3 层架构（MLP）容量稍低，R² 低约 0.03。但训练集 R²=0.888，泛化差距较小，过拟合控制更好。

### 2.7 MLP with All RDKit Descriptors

- **脚本:** `train_mlp_use_all_features.py`
- **特征:** 全部 217 维 RDKit 分子描述符（MolWt、LogP、TPSA、拓扑指数、VSA 分布等所有可用描述符）
- **模型:** PyTorch MLP，3 层 (217 → 256 → 128 → 1) + BatchNorm + Dropout
- **超参数搜索:** Optuna TPE, 30 trials
- **最佳参数:**
  ```json
  {
    "lr": 0.00241,
    "dropout": 0.137,
    "wd": 0.000869,
    "hidden": 256,
    "bs": 64
  }
  ```
- **最佳验证 R²:** 0.8724
- **训练策略:** OneCycleLR + CosineAnnealing + 梯度裁剪 + 1000 epoch
- **测试集:** RMSE=0.4083, MAE=0.2525, **R²=0.8650**
- **分析:** 使用全部 217 维 RDKit 描述符相比此前 62 维精选描述符，测试集 R² 从 0.84 提升至 0.87。训练集 R²=0.99 表明容量充足，建议未来可通过增大 Dropout 或 weight decay 进一步降低过拟合。

### 2.8 RNN (PyTorch LSTM, 全描述符)

- **脚本:** `train_rnn_use_all_features.py`
- **特征:** 全部 217 维 RDKit 分子描述符（作为 217 个时间步 × 1 维进行序列化输入）
- **模型:** PyTorch LSTM，3 层 (hidden_size=64) + Dropout
- **超参数搜索:** Optuna TPE, 30 trials
- **最佳参数:**
  ```json
  {
    "lr": 0.001776,
    "dropout": 0.1562,
    "wd": 0.0001217,
    "hidden_size": 64,
    "num_layers": 3,
    "bs": 32
  }
  ```
- **最佳验证 R²:** 0.8186
- **训练策略:** 1000 epoch，全量训练
- **测试集:** RMSE=0.4610, MAE=0.3152, R²=0.8279
- **分析:** 使用 217 维全描述符的 LSTM 方法将描述符视为序列（217 时间步），测试 R²=0.828，优于同描述符基线的 Keras RNN（62 维，R²=0.812），但低于 MLP（全描述符，R²=0.865）。这表明分子描述符之间并无显著的序列依赖关系，将描述符作为序列建模并不能带来额外收益，反而因序列建模复杂性引入了不必要的噪声。相比之下，MLP 直接利用所有描述符的并行信息更为高效。

### 2.9 CatBoost (全描述符, Optuna) — 当前最佳

- **脚本:** `train_catboost_use_all_features.py`
- **特征:** 全部 217 维 RDKit 分子描述符，不进行标准化（CatBoost 树模型对特征尺度不敏感）
- **超参数搜索:** Optuna TPE, 50 trials, 5-Fold CV RMSE 最小化
- **搜索空间（10 个参数）:**
  ```json
  {
    "depth": [4, 10],
    "learning_rate": [0.01, 0.3] (log),
    "iterations": [500, 3000],
    "l2_leaf_reg": [1.0, 50.0] (log),
    "random_strength": [0.0, 10.0],
    "bagging_temperature": [0.0, 10.0],
    "border_count": [32, 255],
    "one_hot_max_size": [2, 50],
    "leaf_estimation_iterations": [1, 10],
    "min_data_in_leaf": [1, 50]
  }
  ```
- **最佳超参数（5 轮 CV RMSE=0.4450）:**
  ```json
  {
    "depth": 8,
    "learning_rate": 0.01607,
    "iterations": 1230,
    "l2_leaf_reg": 4.192,
    "random_strength": 4.561,
    "bagging_temperature": 7.852,
    "border_count": 76,
    "one_hot_max_size": 27,
    "leaf_estimation_iterations": 6,
    "min_data_in_leaf": 3
  }
  ```
- **最佳 CV RMSE (5-fold):** 0.4450（5 折平均）
- **参数重要性 Top 5:** l2_leaf_reg (0.298), random_strength (0.168), bagging_temperature (0.112), one_hot_max_size (0.099), depth (0.067)
- **训练集（全部数据）:** RMSE=0.0377, MAE=0.0243, R²=0.9988
- **测试集:** RMSE=0.3356, MAE=0.2221, **R²=0.9088**
- **CV R² (5-fold):** 0.3827 ± 0.1772
- **分析:** CatBoost 使用全部 217 维 RDKit 描述符 + Optuna 50 轮调参，测试 R²=**0.9088**，超越此前最佳 LightGBM（0.8994），成为当前最优模型。训练集 R²=0.9988 虽表明一定的过拟合，但测试集表现仍领先于所有其他模型。参数重要性分析显示 `l2_leaf_reg`（0.298）是最关键的调参方向，说明 L2 正则化对 CatBoost 的泛化性能影响最大。`random_strength`（0.168）和 `bagging_temperature`（0.112）紧随其后，表明随机化策略对提升 CatBoost 效果也至关重要。

### 2.10 Transformer + RDKit (全描述符, Optuna)

- **脚本:** `train_transformer_use_all_features.py`
- **特征:** 全部 217 维 RDKit 分子描述符，经 StandardScaler 标准化，作为 217 个时间步（每步 1 个特征）输入 Transformer Encoder
- **模型架构:**
  ```text
  Linear(1 → d_model=128) 映射每个标量描述符
    → 正弦位置编码 (Positional Encoding)
    → Transformer Encoder (3 层, nhead=8, FFN=256, dropout=0.077)
    → Mean Pooling（对所有时间步平均池化）
    → LayerNorm → Dropout → Linear(128→128) → ReLU → Dropout → Linear(128→1)
  ```
- **超参数搜索:** Optuna TPE, 25 trials, 每 trial 60 epoch, MedianPruner（n_startup=5, n_warmup_steps=10）
- **搜索空间（6 个参数）:**
  ```json
  {
    "lr": [0.0001, 0.001] (log),
    "dropout": [0.05, 0.35],
    "weight_decay": [1e-6, 1e-4] (log),
    "nhead": [4, 8],
    "num_layers": [2, 4],
    "dim_feedforward": [256, 512]
  }
  ```
- **Optuna 最佳参数（Trial #22）:**
  ```json
  {
    "lr": 0.000679,
    "dropout": 0.0773,
    "weight_decay": 5.189e-06,
    "nhead": 8,
    "num_layers": 3,
    "dim_feedforward": 256
  }
  ```
- **最佳验证 R²:** 0.6672（epoch 185）
- **训练策略:** 85/15 训练/验证划分，CosineAnnealingLR，早停 patience=30，梯度裁剪 5.0
- **训练集（全量 1204 条）:** RMSE=0.5217, MAE=0.3870, **R²=0.7685**
- **测试集（全部）:** RMSE=0.6796, MAE=0.5033, **R²=0.6261**
- **分析:** 该模型将 217 维 RDKit 描述符作为 217 个时间步序列输入 Transformer Encoder，与 RNN (PyTorch LSTM, 全描述符) 的序列建模思路相同，但使用 Self-Attention 替代 LSTM。测试 R²=**0.6261**，低于 RNN（全描述符, R²=0.828）和 MLP（全描述符, R²=0.865）。这一结果进一步验证了关键结论：分子描述符之间无序列依赖关系，将描述符作为序列建模（无论是 RNN 还是 Transformer）都非最优策略。Transformer 在此任务上表现甚至比 RNN 更差（R²=0.626 vs 0.828），可能是因为：
  1. **归纳偏置不匹配**: Transformer 的位置编码假设序列顺序有意义，而描述符顺序是任意的；
  2. **数据量不足**: Transformer 通常在万/百万级数据上表现最佳，1204 条样本下难以发挥 Self-Attention 的优势；
  3. **描述符维度作为序列长度**: 217 个时间步 × 1 维的表示，相比于 MLP 直接使用 217 维向量的全连接结构，信息密度更低。
  
  训练集 R²=0.769 是所有深度模型中最低的，说明模型容量未得到充分利用。对比使用相同架构但输入为 SMILES token 序列的 Transformer + Word2Vec（测试 R²=0.791），描述符序列版的 Transformer 反而表现更差，进一步说明将分子描述符组织为序列是次优的特征表示策略。

### 2.11 Transformer + Word2Vec (SMILES 序列建模)

- **脚本:** `train_transformer_use_Word2Vec.py`
- **特征:** SMILES 分子序列 → 自训练 Word2Vec 词向量 (dim=128, CBOW) → Token ID 序列 (max_len=128) → Transformer Encoder + Mean Pooling
- **词汇表:** 41 个有效 token + PAD/UNK = 43（vocab_size=43）
- **模型架构:**
  ```text
  Token Embedding (Word2Vec 初始化, 可微调, freeze=False)
    → Positional Encoding (正弦位置编码)
    → Transformer Encoder (3 层, nhead=8, FFN=512, dropout=0.167)
    → Mean Pooling (仅有效 token)
    → LayerNorm → Dropout → Linear(128→128) → ReLU → Linear(128→1)
  ```
- **超参数搜索:** Optuna TPE, 25 trials, 每 trial 60 epoch, MedianPruner（n_startup=5, n_warmup_steps=10）
- **搜索空间（6 个参数）:**
  ```json
  {
    "lr": [0.0001, 0.001] (log),
    "dropout": [0.05, 0.35],
    "weight_decay": [1e-6, 1e-4] (log),
    "nhead": [4, 8],
    "num_layers": [2, 4],
    "dim_feedforward": [256, 512]
  }
  ```
- **Optuna 最佳参数（Trial #20）:**
  ```json
  {
    "lr": 0.000614,
    "dropout": 0.1666,
    "weight_decay": 1.30e-5,
    "nhead": 8,
    "num_layers": 3,
    "dim_feedforward": 512
  }
  ```
- **最佳验证 R²:** 0.7928（epoch 185）
- **训练策略:** 80/20 训练/验证划分（1024/180 条），CosineAnnealingLR，早停 patience=30，梯度裁剪 5.0
- **训练集（全量 1204 条）:** RMSE=0.2808, MAE=0.1917, **R²=0.9329**
- **测试集（140 条）:** RMSE=0.5083, MAE=0.3492, **R²=0.7907**
- **分析:** 该模型尝试从 SMILES 分子序列的 token 级表示学习 pCMC 预测，不同于其他所有模型（直接使用 RDKit 分子描述符）。测试 R²=0.791，在排名中位于中后段，低于使用 62 维精选描述符的 Keras MLP（0.840）和 PyTorch MLP（0.837）。主要局限有三：（1）Word2Vec 仅在 1024 条 SMILES 上训练，词汇表仅 41 个 token，词向量质量不足；（2）Mean Pooling 对序列所有位置一视同仁，无法突出头基等关键结构的信息；（3）SMILES 序列的 token 级表示信息密度远低于 RDKit 物理化学描述符（LogP、TPSA 等与 CMC 直接相关）。训练集 R²=0.933 表明模型仍有容量，但特征层面的瓶颈限制了泛化上限。

### 2.12 Multi-Linear Regression (全描述符, StandardScaler / PCA)

- **脚本:** `train_multi_linear_regression_use_all_features.py`
- **特征:** 全部 217 维 RDKit 分子描述符，经 StandardScaler 标准化，部分使用 PCA 降维
- **模型对比:**

#### 2.12.1 OLS (Ordinary Least Squares)

- **方法:** `sklearn.linear_model.LinearRegression`，标准最小二乘法
- **结果:** 验证集 **R² ≈ 0.0000**（预测几乎等于常数均值）
- **分析:** 217 维全描述符之间存在严重多重共线性，普通最小二乘法的正规方程 (XᵀX)⁻¹ 因矩阵近乎奇异而无法求得稳定解。即使经过 StandardScaler 标准化，OLS 仍完全失效，说明该问题不适合无正则化的线性回归。

#### 2.12.2 Ridge Regression (L2 正则化, Optuna 调优)

- **方法:** `sklearn.linear_model.Ridge`，使用 `solver="sag"` 提高数值稳定性
- **超参数搜索:** Optuna TPE, 60 trials, 3-Fold CV R² 最大化
- **搜索空间:**
  ```json
  {
    "alpha": [0.01, 10000] (log)
  }
  ```
- **最佳 alpha:** 150.21
- **最佳 CV R² (3-fold):** 0.3308
- **训练集:** RMSE=0.6440, MAE=0.4939, R²=0.6389
- **验证集:** RMSE=0.6899, MAE=0.5226, **R²=0.6297**
- **CV R² (5-fold):** 0.3022 ± 0.1365（[0.534, 0.325, 0.115, 0.297, 0.240]）
- **分析:** 通过 L2 正则化（alpha=150.21），Ridge 成功解决了 OLS 的多重共线性崩溃问题，验证集 R²=**0.6297**，CV R² 均值 0.302。最佳 alpha 高达 150，说明需要很强的正则化来抑制 217 维描述符间的共线性噪声。CV 折间存在一定波动（0.115 ~ 0.534），部分折泛化较好。相比树模型（CatBoost 0.909、LightGBM 0.899），线性模型 R² 约 0.63 存在显著差距，说明 pCMC 与描述符之间的关系存在明显的非线性——树模型和神经网络能有效捕捉这些非线性模式，而线性模型受限于其假设。

#### 2.12.3 PCA + OLS

- **方法:** PCA 降维（保留 95% 方差）→ 新特征空间上使用 OLS
- **降维:** 217 维 → **39 主成分**（累积解释 95% 方差）
- **训练集:** RMSE=0.7306, MAE=0.5657, R²=0.5352
- **验证集:** RMSE=0.7707, MAE=0.5948, **R²=0.5378**
- **CV R² (5-fold):** -2.1525 ± 1.7663（部分折严重为负）
- **分析:** PCA 将 217 维描述符压缩至 39 个不相关主成分后使用 OLS，验证集 R²=0.538，低于 Ridge（0.630）。CV 结果极不稳定（最低 -5.18），说明主成分与 pCMC 之间的线性关系很弱。PCA 虽然解决了多重共线性，但降维丢弃了部分预测信息，且保留的主成分仍不足以通过线性回归有效拟合目标变量。**结论：该任务需要非线性模型，线性方法（OLS 和 PCA+OLS）力有不逮。**

- **模型:** PyTorch Geometric AttentiveFP
- **特征:** 分子图（39 维原子特征 + 11 维键特征），在线生成，未预缓存
- **状态:** ⏳ 尚未运行 / 数据未记录

---

## 3. 综合分析与结论

### 3.1 模型排名（按测试集 R²）

| 排名 | 模型 | Test R² | 特点 |
|------|------|---------|------|
| **1** | **CatBoost (全描述符, Optuna)** | **0.9088** | Optuna 50 轮调参，无需 GPU，突破 0.90 |
| **2** | **LightGBM (全描述符, Optuna)** | **0.8994** | Optuna 50 轮调参，无需 GPU，接近 0.90 |
| **3** | **LightGBM (Advanced: RDKit+MACCS+ECFP4, Optuna)** | **0.8893** | 1415 维全量特征，50 轮调参，接近最佳 |
| 4 | XGBoost (全描述符, 特征选择) | 0.8670 | 217→109 维特征选择 + Optuna 100 轮 |
| 5 | MLP (全描述符) | 0.8650 | 全部 217 维 RDKit 描述符 |
| 6 | LightGBM (全描述符, 手动) | 0.8586 | 手动调参，无需 GPU |
| 7 | MLP (Keras, 62维) | 0.8399 | 3 层 MLP，Adam 优化器 |
| 8 | MLP (PyTorch, 62维) | 0.8369 | 3 层 MLP，AdamW 优化器 |
| 9 | RNN (PyTorch LSTM, 全描述符) | 0.8279 | 序列化 217 维描述符，3 层 LSTM |
| 10 | RNN (Keras, 2 层) | 0.8120 | 更轻量，泛化稳定 |
| 11 | **Transformer + Word2Vec (SMILES 序列, Optuna)** | **0.7907** | 端到端 SMILES 序列建模，Optuna 25 轮，自训练词向量 |
| 12 | SVR (RBF) | 0.7835 (Val) | 非深度学习基线，有一定预测能力 |
| 13 | **Transformer + RDKit (全描述符, Optuna)** | **0.6261** | 描述符序列化 + Transformer Encoder，Optuna 25 轮 |
| 14 | **Ridge (全描述符, StandardScaler, Optuna)** | **0.6297 (Val)** | L2 正则化线性模型，217 维全描述符，alpha=150.21 |
| 15 | PCA+OLS (全描述符, 39 主成分) | 0.5378 (Val) | PCA 降维至 39 维后 OLS |
| 16 | OLS (全描述符) | ~0.0000 (Val) | 多重共线性导致 OLS 完全失效 |

### 3.2 关键发现

1. **CatBoost + Optuna 全面领先，首破 0.90:** 通过 50 轮 Optuna 优化（10 个参数、5-Fold CV 目标），CatBoost 使用全部 217 维 RDKit 描述符，测试 R² 达到 **0.9088**，首次突破 0.90 大关，超越此前最佳 LightGBM（0.8994）。LightGBM 紧随其后达 **0.8994**，Advanced 版本（1415 维）达 **0.8893**。CatBoost 的 `l2_leaf_reg` 参数重要性最高（0.298），说明 L2 正则化是其泛化性能的关键控制点。

2. **特征维度并非越高越好:** Advanced 版本使用 1415 维全量特征（RDKit+MACCS+ECFP4+Aux）的测试 R²=0.8893，反而略低于仅使用 217 维 RDKit 描述符的版本（0.8994），差异约 0.01。特征重要性 Top 20 全部来自 RDKit 描述符，MACCS 和 ECFP4 指纹未进入前列，说明额外的分子指纹引入了噪声而非有效信息。这一结果表明，在充分的超参数优化下，精简的 RDKit 描述符集合已能捕捉绝大部分结构-性质关系，增加指纹特征反而可能降低泛化能力。

3. **全量描述符优势显著:** 使用全部 217 维 RDKit 描述符的各模型（CatBoost R²=0.909, LightGBM R²=0.899/0.859, MLP R²=0.865, RNN R²=0.828）均超越 62 维精选描述符的最佳结果（0.84），表明更丰富的描述符集合包含了更多有用的结构-性质关系信息。

4. **CatBoost 超越 LightGBM 成新最佳:** CatBoost（R²=0.909）超越此前 LightGBM（R²=0.899），且无需 GPU、无需特征标准化，训练集 R²=0.999 表明模型容量充足，是当前最佳预测精度的选择。

5. **梯度提升树 vs 神经网络:** CatBoost（R²=0.909）和 LightGBM（0.899）全面领先 MLP（0.865）和 XGBoost（0.867）。四种树模型（CatBoost 0.909 > LightGBM Optuna 0.899 > LightGBM Advanced 0.889 > LightGBM 手动 0.859 ≈ XGBoost 0.867）均表现出与神经网络相当的竞争力。

6. **特征工程有效性:** 62 维 RDKit 描述符+3 层 MLP 即可达到 R²≈0.84，217 维全描述符+CatBoost 进一步提升至 0.91，表明充分优化的树模型可以从全量描述符中提取更多有效信息。但继续增加至 1415 维（含 MACCS/ECFP4）后收益递减，提示特征设计应重质量而非数量。

7. **序列建模 vs 全连接建模:** RNN (PyTorch LSTM) 使用同样的 217 维全描述符，但测试 R²=0.828，低于 MLP 的 0.865 和 LightGBM 的 0.899。Transformer + RDKit 将其作为序列建模的测试 R² 仅 0.626，进一步验证分子描述符之间无序，将其作为序列建模是次优策略。LightGBM 的树模型结构和 MLP 的并行特征处理更适合此类结构化描述符数据。

8. **Optuna 调参收益显著:** LightGBM 从手动调参（R²=0.8586）到 50 轮 Optuna 优化（R²=0.8994），R² 提升 0.04。CatBoost 更以 0.9088 创下新高。CatBoost 参数重要性分析显示 l2_leaf_reg（0.298）、random_strength（0.168）和 bagging_temperature（0.112）是最关键的调参方向。LightGBM 方面 boosting_type（0.337）和 subsample（0.251）最值得关注。XGBoost 方面 reg_alpha（0.422）和 gamma（0.395）最重要。

9. **SVR 局限:** 非线性核 SVM 在该任务中表现不如树模型和神经网络，可能与描述符空间维度及噪声有关。

10. **SMILES 序列端到端建模仍落后于描述符方法:** Transformer + Word2Vec 直接从 SMILES token 序列学习，测试 R²=**0.791**，低于所有使用 RDKit 描述符的模型（最佳 CatBoost 0.909）。对比同属"序列建模"的 RNN (PyTorch LSTM, 全描述符, R²=0.828)，Transformer + Word2Vec 的输入是 SMILES token（41 词汇表），而 RNN 输入是 217 维 RDKit 描述符（每个时间步一个描述符）。两者性能差距说明：**预测 pCMC 的关键信息在于分子的物理化学性质（LogP、TPSA 等），而非 SMILES 字符串的 token 级模式**。小数据量下（1204 条），预计算描述符的信息密度远高于模型从序列中自行学习。Transformer + Word2Vec 的训练集 R²=0.933 表明模型仍有容量，瓶颈在于特征表示而非模型架构。改进方向包括：注意力池化代替平均池化、增加 Word2Vec 训练数据、或混合描述符 + 序列的双流架构。

11. **Transformer 序列建模描述符效果最差:** Transformer + RDKit（测试 R²=**0.626**）是所有深度模型中表现最差的，甚至低于 Ridge 回归（0.630）。与 RNN（全描述符, R²=0.828）对比表明，Transformer 对 217 维描述符序列的建模能力弱于 LSTM，可能是因为：（a）1204 条样本不足以发挥 Transformer 的大容量优势；（b）描述符按列名排序的"序列"无真实顺序意义，正弦位置编码提供了误导性先验；（c）Self-Attention 的计算效率在短序列（217 步）上不足以体现优势。该结果表明，分子描述符数据的最佳建模方式是 MLP 或树模型的并行特征处理，而非 RNN/Transformer 的序列建模。

12. **线性模型不足以捕捉 pCMC 的非线性关系:** Ridge 回归（R²=0.630）和 PCA+OLS（R²=0.538）在所有模型中排名靠后，与最佳树模型（CatBoost R²=0.909）差距约 0.28-0.37。217 维 RDKit 描述符之间存在严重的多重共线性，即使是带 L2 正则化的 Ridge（最佳 alpha=150.21）也仅能达到 0.63 的验证集 R²。PCA 降维虽解决了共线性问题但进一步损失了预测信息（R²=0.538）。相比之下，树模型和神经网络能有效捕捉描述符间的非线性交互，验证了该任务建模需要非线性容量的结论。

### 3.3 推荐方案

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| **最佳预测精度** | CatBoost + 全量 RDKit 描述符 (Optuna) | 测试 R²=0.909，当前最高，无需 GPU |
| **生产部署 / 无需 GPU** | CatBoost / LightGBM + 全量 RDKit 描述符 (Optuna) | CatBoost R²=0.909，LightGBM R²=0.899，训练和推理极快 |
| **快速原型** | LightGBM + 全量 RDKit 描述符 (手动) | 无需调参即达 R²=0.859 |
| **轻量部署** | MLP (62维) | 特征维度低，R²≈0.84，快速推理 |
| **待探索** | AttentiveFP (GNN) | 利用分子拓扑结构，可能超越描述符方法 |

---

## 4. 附录

### 4.1 数据概况

| 属性 | 值 |
|------|-----|
| 总训练样本 | 1335 |
| 含 pCMC 标签 | 1204 (90.2%) |
| 特征维度 | 62（精选）/ 217（全部 RDKit 描述符）/ 1415（RDKit+MACCS+ECFP4+Aux） |
| 目标变量 | pCMC (log CMC) |
| 其他目标 | AW_ST_CMC, Gamma_max, Area_min, Pi_CMC, pC20 |

### 4.2 超参数搜索范围

| 模型 | 搜索空间 | Trial |
|------|---------|-------|
| XGBoost (全描述符, 特征选择) | 特征选择（重要性≥中位数）→ max_depth[3,8], min_child_weight[1,50], gamma[0,10], lr[0.003,0.3], n_estimators[100,2000], subsample[0.4,1.0], colsample_bytree[0.3,1.0], colsample_bylevel[0.3,1.0], reg_lambda[0.1,50], reg_alpha[0.1,50], max_delta_step[0,10] | 100 |
| XGBoost (旧版, 62维) | max_depth[3,10], lr[0.005,0.1], subsample[0.5,1.0], colsample[0.5,1.0], reg_lambda[0.1,20], n_estimators[200,1000] | 60 |
| LightGBM (全描述符, Optuna) | boosting_type[gbdt,dart], max_depth[3,15], num_leaves[15,255], lr[0.001,0.3], n_estimators[500,3000], subsample[0.5,1.0], subsample_freq[1,10], colsample_bytree[0.3,1.0], feature_fraction[0.3,1.0], feature_fraction_bynode[0.3,1.0], reg_lambda[0,30], reg_alpha[0,30], min_child_weight[0.01,50], min_child_samples[1,50], min_data_in_leaf[1,100], min_split_gain[0,1] | 50 |
| LightGBM (Advanced: RDKit+MACCS+ECFP4) | 同上 + lambda_l1[0,10], lambda_l2[0,10], dart 专有参数（drop_rate, max_drop, skip_drop）| 50 |
| LightGBM (全描述符, 手动) | 手动：max_depth=6, lr=0.05, subsample=0.8, colsample=0.8, reg_lambda=1.0, n_estimators=500 | 手动 |
| LightGBM (62维) | max_depth[3,10], lr[0.005,0.1], subsample[0.5,1.0], colsample[0.5,1.0], reg_lambda[0.1,20], n_estimators[200,1000] | 60 |
| SVR | kernel{rbf,poly,sigmoid}, C[0.01,1000], gamma{scale,auto}, epsilon[0.001,1.0] | 60 |
| MLP (PyTorch) | lr[1e-4,1e-2], dropout[0.1,0.4], wd[1e-6,1e-3], h1{128,256,512}, h2{64,128,256}, h3{32,64}, bs{16,32,64} | 30 |
| MLP (全描述符) | lr[1e-4,5e-3], dropout[0.1,0.4], wd[1e-6,1e-3], hidden{128,256,512}, bs{16,32,64} | 30 |
| CatBoost (全描述符) | depth[4,10], lr[0.01,0.3], iterations[500,3000], l2_leaf_reg[1,50], random_strength[0,10], bagging_temperature[0,10], border_count[32,255], one_hot_max_size[2,50], leaf_estimation_iterations[1,10], min_data_in_leaf[1,50] | 50 |
| RNN (PyTorch LSTM, 全描述符) | lr[1e-4,5e-3], dropout[0.05,0.4], wd[1e-6,1e-3], hidden_size{32,64,128}, num_layers[1,3], bs{16,32,64} | 30 |
| RNN (Keras) | lr[1e-4,1e-2], dropout[0.1,0.4], l2[1e-5,1e-3], units_1{64,128,256}, units_2{32,64,128}, bs{16,32,64} | 30 |
| AttentiveFP | lr[1e-4,5e-3], dropout[0.05,0.4], wd[1e-6,1e-3], hidden_dim{64,128,256}, num_layers[2,5], num_timesteps[2,4], bs{16,32,64} | 30 |
| Ridge (全描述符) | alpha[0.01, 10000] (log), solver="sag" | 60 |
| **Transformer + RDKit (全描述符, Optuna)** | **lr[1e-4,1e-3] (log), dropout[0.05,0.35], weight_decay[1e-6,1e-4] (log), nhead{4,8}, num_layers[2,4], dim_feedforward{256,512}** | **25** |
| **Transformer + Word2Vec (SMILES 序列, Optuna)** | **lr[1e-4,1e-3] (log), dropout[0.05,0.35], weight_decay[1e-6,1e-4] (log), nhead{4,8}, num_layers[2,4], dim_feedforward{256,512}** | **25** |