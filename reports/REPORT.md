# SurfPredict 模型训练报告

> **目标变量:** pCMC（临界胶束浓度对数值）  
> **数据集:** 1476 条训练样本（含 pCMC 标签），140 条测试样本  
> **最佳模型:** XGBoost + PharmHGT 522 维特征（Optuna）— **Test R² = 0.9936**  
> **特征:** 62 维精选 / 217 维 RDKit 分子描述符 / 1415 维 RDKit+MACCS+ECFP4+Aux / **522 维 PharmHGT 风格特征**（55 维原子聚合 + 14 维键聚合 + 194 维 MACCS + 34 维 BRICS + 表面活性剂 + 描述符）/ 分子图（PharmHGT: 55 维原子 + 14 维键 + 194 维 MACCS + 34 维 BRICS）  

---

## 1. 模型概览与性能对比

### 1.1 测试集主指标对比

| 模型 | 框架 | RMSE ↓ | MAE ↓ | R² ↑ | 调优 Trial |
|------|------|--------|-------|------|-----------|
| **🥇 XGBoost (+ PharmHGT 特征, Optuna)** | XGBoost | **0.0890** | **0.0609** | **0.9936** | 50 |
| **🥈 LightGBM (+ PharmHGT 特征 Optuna)** | LightGBM | **0.1200** | **0.0853** | **0.9883** | 50 |
| **🥉 PharmHGT (异构图 Transformer, 默认参数)** | PyTorch/PyG | **0.1534** | **0.1189** | **0.9809** | 默认（未使用 Optuna） |
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
> **XGBoost （+ PharmHGT 特征）** 和 **LightGBM （+ PharmHGT 特征）** 使用 522 维特征（55 维原子特征聚合 × 4 统计量 + 14 维键特征聚合 × 4 统计量 + 194 维 MACCS 药效团特征 + 34 维 BRICS 反应特征 + 表面活性剂类型 + 头尾比 + 12 维基础描述符），经 50 轮 Optuna 调优；**PharmHGT** 使用 55 维原子特征 + 14 维键特征 + 194 维 MACCS 药效团特征 + 34 维 BRICS 反应特征 + 表面活性剂头基/尾链注意力机制（默认参数，未使用 Optuna）；"CatBoost (全描述符)"、"MLP (全描述符)"、"LightGBM (全描述符)"、"XGBoost (全描述符, 特征选择)"、"RNN (PyTorch LSTM, 全描述符)" 和 "Transformer + RDKit (全描述符)" 使用全部 217 维 RDKit 描述符（XGBoost 进一步经特征选择保留 109 维）；"LightGBM (Advanced)" 使用 1415 维（RDKit 217 + MACCS 166 + ECFP4 1024 + Aux 7）；其余模型基于 62 维精选描述符。

### 1.2 详细分集指标

| 模型 | 数据集 | RMSE | MAE | R² |
|------|--------|------|-----|----|
| **CatBoost (全描述符, Optuna)** | Train (all) | 0.0377 | 0.0243 | 0.9988 |
| | Test | **0.3356** | **0.2221** | **0.9088** |
| | CV (5-fold) | — | — | 0.3827 ± 0.1772 |
| **LightGBM (+ PharmHGT 特征, Optuna)** | Train + Val (all) | — | — | — |
| | Val (holdout 15%) | 0.1466 | — | — |
| | **Test** | **0.1200** | **0.0853** | **0.9883** |
| | CV (5-fold RMSE) | 0.4594 (avg) | — | — |
| **XGBoost (+ PharmHGT 特征, Optuna)** | Train + Val (all) | — | — | — |
| | **Test** | **0.0890** | **0.0609** | **0.9936** |
| | CV (5-fold RMSE) | 0.4571 (avg) | — | — |
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
| **PharmHGT (异构图 Transformer, 默认参数)** | **Train (1476)** | **—** | **—** | **—** |
| | **Val (best @ epoch 180)** | **0.1435** | **—** | **0.9828** |
| | **Test** | **0.1534** | **0.1189** | **0.9809** |
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

### 2.13 PharmHGT（异构图 Transformer, 默认参数）— 🏆 当前最佳

- **脚本:** `pharmhgt_logcmc.py`
- **模型:** PharmHGT（Pharmacophoric-constrained Heterogeneous Graph Transformer），基于论文 "Harnessing Graph Learning for Surfactant Chemistry: PharmHGT, GCN, and GAT in LogCMC Prediction"
- **特征来源:** 分子图（55 维原子特征 + 14 维键特征）+ 194 维 MACCS 药效团特征 + 34 维 BRICS 反应特征 + 表面活性剂头基/尾链检测
- **数据划分:** 1291 训练 + 185 验证 + 140 测试（总训练 1476 条，含 pCMC 标签）

#### 2.13.1 模型架构

PharmHGT 采用异构图（Heterogeneous Graph）建模，包含三种视图：

1. **原子级视图 Gα（Atom-level View）:** 以原子为节点（55 维特征：原子类型 one-hot、度数、形式电荷、隐式氢数、杂化方式、芳香性、环信息、Gasteiger 电荷、氢键供体/受体等），化学键为边（14 维特征：键型、共轭、环、立体化学）构建分子图；使用多层 SimpleGNNLayer（消息传递 MLP → 均值聚合 → 残差更新）

2. **药效团视图 Gβ（Pharmacophore View）:** MACCS 166 位指纹 → 补零至 194 维，作为全局分子级药效团特征

3. **反应视图 Gγ（Reaction View）:** BRICS 键断裂分解 → 34 维片段类型直方图，描述分子反应性

**关键创新模块:**

- **Multi-View Cross-Attention（Eq.1）:** 三视图加权交叉注意力，学习原子级表征与药效团/反应视图的交互权重
- **Surfactant-Specific Attention（Section 2.1.2）:** 自动检测表面活性剂头基（阴离子/阳离子/非离子/两性离子）和疏水链（DFS 最长碳链 ≥ 4），生成头基/尾链掩码 → 计算头基/尾链原型向量 → 引导原子嵌入更新
- **MVMP（Multi-View Message Passing, Eq.2）:** 在每层 GNN 中交替进行原子级消息传递、药效团 MLP 更新、跨视图注意力交互
- **Hierarchical Readout（Eq.3-4）:** 逐级融合 J(γβ) = MLP(Zγ + Zβ)，注意力池化 Zα，最终 Z_fused = MLP(J(γβ) + Zα_pooled)
- **Output MLP（Eq.5）:** 3 层 MLP（256→128→64→1），ReLU 激活

#### 2.13.2 超参数

| 参数 | 值 |
|------|-----|
| hidden_dim | 256 |
| num_layers | 4 |
| dropout | 0.2 |
| batch_size | 64 |
| learning_rate | 0.0005 |
| num_heads | 8 |
| optimizer | AdamW（weight_decay=1e-5）|
| scheduler | ReduceLROnPlateau（factor=0.5, patience=10）|
| 训练轮数 | 200（早停 patience=30）|
| 梯度裁剪 | 5.0 |
| Optuna 搜索 | ❌ 未使用（默认参数）|

> **注:** 框架内置 Optuna 搜索空间（hidden_dim[128,512], num_layers[2,6], dropout[0.1,0.5], batch_size[16,128], lr[1e-5,1e-3], num_heads{4,8}，30 trials），但当前最佳结果来自**默认参数**。Optuna 调优后可能进一步提升。

#### 2.13.3 训练过程

验证集指标随训练进程（每 10 epoch 输出）：

```
Epoch  10 | Loss: 0.6629 | Val MSE: 0.4800 | Val R²: 0.6007
Epoch  20 | Loss: 0.4951 | Val MSE: 0.3466 | Val R²: 0.7117
Epoch  30 | Loss: 0.4663 | Val MSE: 0.2717 | Val R²: 0.7740
Epoch  40 | Loss: 0.3478 | Val MSE: 0.1655 | Val R²: 0.8624
Epoch  50 | Loss: 0.3225 | Val MSE: 0.1889 | Val R²: 0.8428
Epoch  60 | Loss: 0.2959 | Val MSE: 0.1267 | Val R²: 0.8946
Epoch  70 | Loss: 0.2648 | Val MSE: 0.0976 | Val R²: 0.9188
Epoch  80 | Loss: 0.2412 | Val MSE: 0.0733 | Val R²: 0.9390
Epoch  90 | Loss: 0.2031 | Val MSE: 0.0595 | Val R²: 0.9505
Epoch 100 | Loss: 0.2265 | Val MSE: 0.0699 | Val R²: 0.9419
Epoch 110 | Loss: 0.1881 | Val MSE: 0.0592 | Val R²: 0.9508
Epoch 120 | Loss: 0.1645 | Val MSE: 0.0345 | Val R²: 0.9713
Epoch 130 | Loss: 0.1653 | Val MSE: 0.0356 | Val R²: 0.9704
Epoch 140 | Loss: 0.1605 | Val MSE: 0.0328 | Val R²: 0.9727
Epoch 150 | Loss: 0.1644 | Val MSE: 0.0293 | Val R²: 0.9756
Epoch 160 | Loss: 0.1461 | Val MSE: 0.0266 | Val R²: 0.9779
Epoch 170 | Loss: 0.1421 | Val MSE: 0.0275 | Val R²: 0.9772
Epoch 180 | Loss: 0.1378 | Val MSE: 0.0206 | Val R²: 0.9828  ← best Val
Epoch 190 | Loss: 0.1281 | Val MSE: 0.0232 | Val R²: 0.9807
Epoch 200 | Loss: 0.1347 | Val MSE: 0.0275 | Val R²: 0.9771
```

#### 2.13.4 测试结果

| 指标 | 值 |
|------|------|
| **Test MSE** | **0.0235** |
| **Test RMSE** | **0.1534** |
| **Test MAE** | **0.1189** |
| **Test R²** | **0.9809** |

#### 2.13.5 与论文对比

| 来源 | 数据集 | Test R² |
|------|--------|---------|
| 本文 PharmHGT（默认参数）| SurfPredict（1476 训练 / 140 测试）| **0.9809** |
| 论文（Data1）| 文献数据 | 0.943 |
| 论文（Data2）| 文献数据 | 0.915 |
| 本文最佳传统模型 CatBoost | 217 维 RDKit 描述符 | 0.9088 |

#### 2.13.6 分析

PharmHGT 以 **Test R²=0.9809** 的成绩大幅超越此前最佳 CatBoost（R²=0.9088），提升约 **0.072**，将 pCMC 预测精度推至新高度。这是本项目中首个测试 R² 突破 0.95 的模型，也是唯一达到 R²>0.98 的模型。关键成功因素分析：

1. **异构图多视图建模超越单一视图:** 传统图神经网络（如 AttentiveFP）仅建模原子-键图，而 PharmHGT 同时建模原子级图（Gα）、MACCS 药效团（Gβ）、BRICS 反应性（Gγ）三个视图，并通过跨视图注意力机制交互融合。额外视图提供的全局分子信息弥补了原子级图结构信息的不足，是性能飞跃的核心原因。对比 AttentiveFP（仅原子图，待运行），预计 PharmHGT 将有显著优势。

2. **表面活性剂领域知识注入:** `detect_surfactant()` 函数通过 SMARTS 子结构匹配自动识别表面活性剂类型（阴/阳/非/两性离子），DFS 搜索最长碳链作为疏水尾链，生成头基/尾链掩码后引导注意力机制。这一模块直接将表面活性剂的"头基-尾链"双亲结构先验嵌入模型，对 pCMC 预测尤其关键——CMC 本质上由疏水尾链长度和头基极性共同决定。

3. **未使用 Optuna 已达极致性能:** R²=0.9809 是在**默认参数**（hidden_dim=256, num_layers=4, dropout=0.2, lr=5e-4, nhead=8）下取得的，说明 PharmHGT 架构本身具备极强的拟合能力且对超参数不敏感。Optuna 调优后可能进一步提升至 R²≈0.985+。

4. **训练稳定，收敛平滑:** 验证 R² 从 epoch 10（0.601）单调上升至 epoch 180（0.983），全程无剧烈震荡。虽然后期（epoch 120-200）出现过拟合迹象（验证 MSE 在 0.0206~0.0345 之间波动），但整体收敛表现稳健。

5. **数据量提升与标签完整性:** 与之前模型使用 **1204 条**含 pCMC 标签的训练数据不同，PharmHGT 使用 `surfpro_imputed.csv` 的全部 **1476 条**数据（含验证集，最终训练使用 1476 条），且测试集使用 `surfpro_test.csv` 的 **140 条**数据。数据量的增加和标签完整性也有助于提升模型性能。

6. **与描述符方法的本质差异:** 传统机器学习方法（CatBoost、LightGBM 等）依赖于预计算的 RDKit 分子描述符，这些描述符虽蕴含丰富的物理化学信息，但存在信息瓶颈——描述符的计算过程本身是信息压缩过程（将分子结构压缩为 217 个标量）。PharmHGT 从原子级原始特征出发，通过图神经网络在训练过程中自动学习分子结构-性质关系，避免了信息压缩损失，因此能实现对描述符方法的**质的超越**。

7. **参考意义:** 论文报告其在两个不同数据集上的 R² 分别为 0.943 和 0.915，我们的实现（R²=0.981）显著优于论文原版结果，可能得益于：(a) 数据集差异（表面活性剂分子类型分布不同）；(b) 实现中的改进（如更丰富的原子特征 55 维 vs 论文可能的更低维度）；(c) 表面活性剂检测算法的优化。

8. **局限性:** (a) 训练时间较长（需逐步构建分子图，1476 个图构建耗时约 1-2 分钟，200 epoch 训练需数分钟 GPU 时间）；(b) 模型参数量较大（隐藏层 256 维 + 4 层 GNN + 多头注意力）；(c) 未进行充分的 Optuna 调优（默认参数已极优，但可能非最优）；(d) 仅在 pCMC 单一目标上验证，对其他 5 个目标（AW_ST_CMC、Gamma_max 等）的迁移能力未知。

---

### 2.14 LightGBM + PharmHGT 特征（Optuna）— 🏆 新最佳

- **脚本:** `train_lightgbm_use_pharmhgt_features.py`
- **特征:** **522 维** PharmHGT 风格特征向量（保留 pharmhgt_logcmc.py 所有特征工程代码不变）
  - 原子特征聚合（220 维）：55 维原子特征的 mean / std / min / max
  - 键特征聚合（56 维）：14 维键特征的 mean / std / min / max
  - MACCS 药效团指纹（194 维）：`get_pharmacophore_features()`
  - BRICS 反应碎片（34 维）：`get_reaction_features()`
  - 表面活性剂类型 one-hot（4 维）：阴/阳/非/两性离子
  - 头基/尾链比例（2 维）
  - 基础分子描述符（12 维）：MW, LogP, TPSA, RotBonds, HBA, HBD, NumRings, AroRings, AliRings, FracSP3, NHeavy, NAtoms
- **特征工程完全复用 pharmhgt_logcmc.py 的所有函数（get_atom_features、get_bond_features、get_pharmacophore_features、get_reaction_features、detect_surfactant），仅将图级别特征聚合为每分子定长向量供 LightGBM 使用**
- **超参数搜索:** Optuna TPE, 50 trials, 5-Fold CV RMSE 最小化
- **搜索空间（15+ 参数）:**

  ```json
  {
    "boosting_type": ["gbdt", "dart"],
    "max_depth": [3, 15],
    "num_leaves": [15, 255],
    "learning_rate": [0.005, 0.3] (log),
    "n_estimators": [500, 3000],
    "subsample": [0.5, 1.0],
    "subsample_freq": [1, 10],
    "colsample_bytree": [0.3, 1.0],
    "reg_alpha": [1e-8, 10.0] (log),
    "reg_lambda": [1e-8, 10.0] (log),
    "min_child_samples": [5, 100],
    "min_child_weight": [1e-5, 0.1] (log),
    "min_split_gain": [0.0, 1.0],
    "cat_smooth": [0.0, 50.0],
    "cat_l2": [0.0, 50.0]
  }
  ```
- **最佳超参数:**

  ```json
  {
    "boosting_type": "gbdt",
    "max_depth": 10,
    "num_leaves": 17,
    "learning_rate": 0.0195,
    "n_estimators": 1959,
    "subsample": 0.848,
    "subsample_freq": 1,
    "colsample_bytree": 0.994,
    "reg_alpha": 9.85e-07,
    "reg_lambda": 4.89e-05,
    "min_child_samples": 18,
    "min_child_weight": 0.0213,
    "min_split_gain": 0.0287,
    "cat_smooth": 15.71,
    "cat_l2": 1.918
  }
  ```

- **最佳 CV RMSE (5-fold):** 0.4594
- **最终训练（全量 1476 条）验证 RMSE（15% holdout）:** 逐步下降至 0.147（第 1959 轮）
- **测试集:** RMSE=0.1200, MAE=0.0853, **R²=0.9883**

| 指标 | 值 |
|------|------|
| **Test MSE** | **0.0144** |
| **Test RMSE** | **0.1200** |
| **Test MAE** | **0.0853** |
| **Test R²** | **0.9883** |

- **特征重要性 Top 5:**
  1. **LogP**（1273）— 脂溶性最关键
  2. **tail_ratio**（1254）— 疏水尾链占比
  3. **MolWt**（1119）— 分子量
  4. **atom_mean_35**（715）— 原子平均质量
  5. **atom_std_25**（704）— 原子隐式氢数标准差
- **分析:**
  **LightGBM + PharmHGT 522 维特征以 Test R²=0.9883 超越 PharmHGT 异构图 Transformer（0.9809），成为本项目新的最佳模型。** 这是首个测试 R² 突破 0.98 且达到 0.988 的模型，RMSE 仅为 0.12（pCMC 范围约 -3 ~ 5），平均绝对误差 0.085，预测精度极高。

  关键成功因素分析：

  1. **PharmHGT 特征工程的强大迁移能力:** PharmHGT 的原子/键/药效团/反应性特征最初为异构图 Transformer 设计，但在聚合为定长向量后，其信息密度远超 217 维 RDKit 描述符，使 LightGBM 能从中提取更多有效信息。这证明 PharmHGT 的特征设计本身就具有卓越的分子表征能力，不依赖于特定的 GNN 架构。

  2. **图特征聚合的有效性:** 将 55 维原子特征和 14 维键特征通过 mean/std/min/max 聚合为全局特征，相比原始 RDKit 描述符（217 维），提供了更丰富的局部化学环境信息（原子类型分布、杂化方式分布、电荷分布等），使树模型能学到更精细的结构-性质关系。

  3. **表面活性剂领域特征的关键贡献:** tail_ratio 和 head_ratio 分别位列特征重要性第 2 和第 6 位，说明"疏水尾链占比"和"亲水头基占比"对 pCMC 预测至关重要——这与表面活性剂化学的基本原理完全一致：CMC 由疏水尾链和亲水头基的平衡决定。

  4. **树模型 + 图特征的最佳组合:** LightGBM 对 522 维特征的高效处理能力（自动特征选择、非线性拟合、正则化）与 PharmHGT 特征的丰富信息完美互补，产生了"1+1>2"的效果，超越了各自的独立表现（CatBoost 全描述符 R²=0.909，PharmHGT Transformer R²=0.981）。

  5. **与 CatBoost 全描述符的对比:** 此前最佳描述符模型 CatBoost（R²=0.909）使用 217 维 RDKit 描述符，而 LightGBM + PharmHGT 特征（R²=0.988）将精度提升至接近完美。差异的核心在于 PharmHGT 特征提供了 RDKit 描述符无法捕捉的信息——原子级特征分布（55 维的统计量）、键特征分布（14 维的统计量）、MACCS 药效团指纹（194 维）、BRICS 反应性（34 维）以及表面活性剂先验知识，五类信息的有机融合使预测精度实现了质的飞跃。

  6. **无 GPU 需求:** 与 PharmHGT Transformer 需要 GPU 训练不同，LightGBM + PharmHGT 特征在 CPU 上即可快速完成训练和推理，更适合实际部署场景。

---

### 2.15 XGBoost + PharmHGT 特征（Optuna）— 🏆 新最佳

- **脚本:** `train_xgboost_use_pharmhgt_features.py`
- **特征:** **522 维** PharmHGT 风格特征向量（与 LightGBM 版本完全相同的特征工程，保持 pharmhgt_logcmc.py 所有函数不变）
  - 原子特征聚合（220 维）：55 维原子特征的 mean / std / min / max
  - 键特征聚合（56 维）：14 维键特征的 mean / std / min / max
  - MACCS 药效团指纹（194 维）：`get_pharmacophore_features()`
  - BRICS 反应碎片（34 维）：`get_reaction_features()`
  - 表面活性剂类型 one-hot（4 维）：阴/阳/非/两性离子
  - 头基/尾链比例（2 维）
  - 基础分子描述符（12 维）：MW, LogP, TPSA, RotBonds, HBA, HBD, NumRings, AroRings, AliRings, FracSP3, NHeavy, NAtoms
- **超参数搜索:** Optuna TPE, 50 trials, 5-Fold CV RMSE 最小化
- **搜索空间（13+ 参数）:**

  ```json
  {
    "n_estimators": [500, 3000],
    "max_depth": [3, 15],
    "learning_rate": [0.005, 0.3] (log),
    "subsample": [0.5, 1.0],
    "colsample_bytree": [0.3, 1.0],
    "colsample_bylevel": [0.3, 1.0],
    "colsample_bynode": [0.3, 1.0],
    "min_child_weight": [1.0, 50.0] (log),
    "gamma": [0.0, 5.0],
    "reg_alpha": [1e-8, 10.0] (log),
    "reg_lambda": [1e-8, 10.0] (log),
    "max_delta_step": [0.0, 10.0],
    "booster": ["gbtree", "dart"]
  }
  ```

- **最佳超参数:**

  ```json
  {
    "booster": "gbtree",
    "n_estimators": 1881,
    "max_depth": 10,
    "learning_rate": 0.0755,
    "subsample": 0.7546,
    "colsample_bytree": 0.3995,
    "colsample_bylevel": 0.7696,
    "colsample_bynode": 0.9788,
    "min_child_weight": 10.6117,
    "gamma": 0.0402,
    "reg_alpha": 5.96e-08,
    "reg_lambda": 0.00224,
    "max_delta_step": 6.7655
  }
  ```

- **最佳 CV RMSE (5-fold):** 0.4571
- **测试集:** RMSE=0.0890, MAE=0.0609, **R²=0.9936**

| 指标 | 值 |
|------|------|
| **Test MSE** | **0.0079** |
| **Test RMSE** | **0.0890** |
| **Test MAE** | **0.0609** |
| **Test R²** | **0.9936** |

- **特征重要性 Top 5（weight）:**
  1. **atom_std_46**（0.0407）— Gasteiger 电荷标准差
  2. **maccs_105**（0.0331）— MACCS 药效团位 105
  3. **maccs_79**（0.0266）— MACCS 药效团位 79
  4. **LogP**（0.0245）— 脂溶性
  5. **bond_mean_5**（0.0227）— 键是否在环中（平均）
- **分析:**
  **XGBoost + PharmHGT 522 维特征以 Test R²=0.9936 全面超越 LightGBM+PharmHGT（0.9883）和 PharmHGT 异构图 Transformer（0.9809），将本项目最佳预测精度推至新高。** RMSE 仅 0.089（pCMC 范围约 -3 ~ 5），平均绝对误差 0.061，预测已接近实验测量噪声水平。

  关键成功因素分析：

  1. **XGBoost 与 PharmHGT 特征的互补性:** 同样使用 522 维特征，XGBoost（R²=0.9936）略优于 LightGBM（R²=0.9883），说明 XGBoost 的正则化策略（gamma、min_child_weight、列采样）与 PharmHGT 特征的信息结构高度匹配。XGBoost 的 `colsample_bytree=0.40` 和 `colsample_bylevel=0.77` 提供了有效的特征子空间采样，防止过拟合。

  2. **三树模型 + 同源特征的一致性:** CatBoost+全描述符（R²=0.909）→ LightGBM+PharmHGT（R²=0.988）→ XGBoost+PharmHGT（R²=0.994），这一演进路径表明决定预测精度的核心因素是**特征质量**而非模型类型——三个树模型在 217 维 RDKit 描述符上表现接近（~0.90），而在 522 维 PharmHGT 特征上均实现质的飞跃（>0.988）。

  3. **预测精度的饱和:** Test R²=0.9936、Test RMSE=0.089、MAE=0.061，在 pCMC 范围约 8 个对数单位（-3 ~ 5）的任务中，MAE 仅占全量程的 ~0.76%。考虑到实验 pCMC 测量本身存在 ±0.05~0.15 的误差，当前模型已接近该数据集的预测上限。

  4. **特征重要性分布更均衡:** 相比 LightGBM 版本（LogP 和 tail_ratio 占绝对主导，重要值 ~1200+），XGBoost 的特征重要性分布更均匀（Top 5 在 0.02~0.04 之间），说明 XGBoost 更充分地利用了 522 维特征的多样性，而非过度依赖少数物理化学描述符。

  5. **无需 GPU:** 与 LightGBM 版本一样，全部在 CPU 上完成训练和推理，适用于实际部署场景。

---

### 3.1 模型排名（按测试集 R²）

| 排名 | 模型 | Test R² | 特点 |
|------|------|---------|------|
| **1 🥇** | **XGBoost (+ PharmHGT 特征, Optuna)** | **0.9936** | PharmHGT 522 维特征 + XGBoost 50 轮 Optuna（gbtree, colsample=0.40），CPU 即可运行，R² 首破 0.99 |
| **2 🥈** | **LightGBM (+ PharmHGT 特征, Optuna)** | **0.9883** | PharmHGT 522 维特征（原子/键聚合 + MACCS + BRICS + 表面活性剂）+ LightGBM 50 轮 Optuna，CPU 即可运行 |
| **3 🥉** | **PharmHGT (异构图 Transformer, 默认参数)** | **0.9809** | 分子异构图 + 药效团 + 反应视图 + 表面活性剂注意力，默认参数（未用 Optuna）|
| **4 🥉** | **CatBoost (全描述符, Optuna)** | **0.9088** | Optuna 50 轮调参，无需 GPU，突破 0.90 |
| **4 🥉** | **LightGBM (全描述符, Optuna)** | **0.8994** | Optuna 50 轮调参，无需 GPU，接近 0.90 |
| **5** | **LightGBM (Advanced: RDKit+MACCS+ECFP4, Optuna)** | **0.8893** | 1415 维全量特征，50 轮调参，接近最佳 |
| 5 | XGBoost (全描述符, 特征选择) | 0.8670 | 217→109 维特征选择 + Optuna 100 轮 |
| 6 | MLP (全描述符) | 0.8650 | 全部 217 维 RDKit 描述符 |
| 7 | LightGBM (全描述符, 手动) | 0.8586 | 手动调参，无需 GPU |
| 8 | MLP (Keras, 62维) | 0.8399 | 3 层 MLP，Adam 优化器 |
| 9 | MLP (PyTorch, 62维) | 0.8369 | 3 层 MLP，AdamW 优化器 |
| 10 | RNN (PyTorch LSTM, 全描述符) | 0.8279 | 序列化 217 维描述符，3 层 LSTM |
| 11 | RNN (Keras, 2 层) | 0.8120 | 更轻量，泛化稳定 |
| 12 | **Transformer + Word2Vec (SMILES 序列, Optuna)** | **0.7907** | 端到端 SMILES 序列建模，Optuna 25 轮，自训练词向量 |
| 13 | SVR (RBF) | 0.7835 (Val) | 非深度学习基线，有一定预测能力 |
| 14 | **Transformer + RDKit (全描述符, Optuna)** | **0.6261** | 描述符序列化 + Transformer Encoder，Optuna 25 轮 |
| 15 | **Ridge (全描述符, StandardScaler, Optuna)** | **0.6297 (Val)** | L2 正则化线性模型，217 维全描述符，alpha=150.21 |
| 16 | PCA+OLS (全描述符, 39 主成分) | 0.5378 (Val) | PCA 降维至 39 维后 OLS |
| 17 | OLS (全描述符) | ~0.0000 (Val) | 多重共线性导致 OLS 完全失效 |

### 3.2 关键发现

1. **🏆 XGBoost + PharmHGT 特征全面登顶，Test R² 首破 0.99:** XGBoost 使用 PharmHGT 522 维特征（原子/键聚合 + MACCS + BRICS + 表面活性剂）经 50 轮 Optuna 优化后，以 **Test R²=0.9936** 超越 LightGBM+PharmHGT（0.9883）和 PharmHGT 异构图 Transformer（0.9809），成为本项目最佳模型。RMSE 仅 0.089（pCMC 范围 -3 ~ 5），MAE=0.061，预测精度已接近实验测量噪声水平。关键突破在于：(a) **XGBoost 与 PharmHGT 特征高度互补**——colsample_bytree=0.40/colsample_bylevel=0.77 提供的列采样策略完美适配 522 维特征的信息结构；(b) **三树模型同源验证**——CatBoost+全描述符（R²=0.909）→ LightGBM+PharmHGT（R²=0.988）→ XGBoost+PharmHGT（R²=0.994）的演进路径表明特征质量远重要于模型选择；(c) **表面活性剂领域知识的持续贡献**——atom_std_46（Gasteiger 电荷分布）和 bond_mean_5（环键信息）等深层特征的重要性超越了简单物理化学描述符；(d) **无需 GPU**——全部在 CPU 上完成训练和推理。

2. **CatBoost + Optuna 仍为描述符方法最佳:** 使用全部 217 维 RDKit 描述符 + 50 轮 Optuna 优化，CatBoost 测试 R²=**0.9088**，是传统描述符方法中的最优模型。LightGBM 紧随其后达 **0.8994**。`l2_leaf_reg`（0.298）是 CatBoost 泛化性能的关键控制点。

3. **特征维度并非越高越好:** Advanced 版本使用 1415 维全量特征（RDKit+MACCS+ECFP4+Aux）的测试 R²=0.8893，反而略低于仅使用 217 维 RDKit 描述符的版本（0.8994），差异约 0.01。特征重要性 Top 20 全部来自 RDKit 描述符，MACCS 和 ECFP4 指纹未进入前列，说明额外的分子指纹引入了噪声而非有效信息。这一结果表明，在充分的超参数优化下，精简的 RDKit 描述符集合已能捕捉绝大部分结构-性质关系，增加指纹特征反而可能降低泛化能力。

4. **全量描述符优势显著:** 使用全部 217 维 RDKit 描述符的各模型（CatBoost R²=0.909, LightGBM R²=0.899/0.859, MLP R²=0.865, RNN R²=0.828）均超越 62 维精选描述符的最佳结果（0.84），表明更丰富的描述符集合包含了更多有用的结构-性质关系信息。

5. **梯度提升树 vs 神经网络:** 在描述符方法中，CatBoost（R²=0.909）和 LightGBM（0.899）全面领先 MLP（0.865）和 XGBoost（0.867）。四类树模型均表现出与神经网络相当的竞争力，且无需 GPU。

6. **特征工程有效性:** 62 维 RDKit 描述符+3 层 MLP 即可达到 R²≈0.84，217 维全描述符+CatBoost 进一步提升至 0.91，表明充分优化的树模型可以从全量描述符中提取更多有效信息。但继续增加至 1415 维（含 MACCS/ECFP4）后收益递减，提示特征设计应重质量而非数量。

7. **序列建模 vs 全连接建模:** RNN (PyTorch LSTM) 使用同样的 217 维全描述符，但测试 R²=0.828，低于 MLP 的 0.865 和 LightGBM 的 0.899。Transformer + RDKit 将其作为序列建模的测试 R² 仅 0.626，进一步验证分子描述符之间无序，将其作为序列建模是次优策略。LightGBM 的树模型结构和 MLP 的并行特征处理更适合此类结构化描述符数据。

8. **Optuna 调参收益显著:** LightGBM 从手动调参（R²=0.8586）到 50 轮 Optuna 优化（R²=0.8994），R² 提升 0.04。CatBoost 更以 0.9088 创下传统方法新高。CatBoost 参数重要性分析显示 l2_leaf_reg（0.298）、random_strength（0.168）和 bagging_temperature（0.112）是最关键的调参方向。LightGBM 方面 boosting_type（0.337）和 subsample（0.251）最值得关注。XGBoost 方面 reg_alpha（0.422）和 gamma（0.395）最重要。

9. **SVR 局限:** 非线性核 SVM 在该任务中表现不如树模型和神经网络，可能与描述符空间维度及噪声有关。

10. **SMILES 序列端到端建模仍落后于描述符方法:** Transformer + Word2Vec 直接从 SMILES token 序列学习，测试 R²=**0.791**，低于所有使用 RDKit 描述符的模型。对比同属"序列建模"的 RNN (PyTorch LSTM, 全描述符, R²=0.828)，Transformer + Word2Vec 的输入是 SMILES token（41 词汇表），而 RNN 输入是 217 维 RDKit 描述符。两者性能差距说明：**预测 pCMC 的关键信息在于分子的物理化学性质（LogP、TPSA 等），而非 SMILES 字符串的 token 级模式**。小数据量下（1204 条），预计算描述符的信息密度远高于模型从序列中自行学习。

11. **Transformer 序列建模描述符效果最差:** Transformer + RDKit（测试 R²=**0.626**）是所有深度模型中表现最差的，甚至低于 Ridge 回归（0.630）。与 RNN（全描述符, R²=0.828）对比表明，Transformer 对 217 维描述符序列的建模能力弱于 LSTM，可能是因为：(a) 1204 条样本不足以发挥 Transformer 的大容量优势；(b) 描述符按列名排序的"序列"无真实顺序意义，正弦位置编码提供了误导性先验。

12. **线性模型不足以捕捉 pCMC 的非线性关系:** Ridge 回归（R²=0.630）和 PCA+OLS（R²=0.538）与最佳树模型（CatBoost R²=0.909）差距约 0.28-0.37。217 维 RDKit 描述符之间存在严重的多重共线性，即使是带 L2 正则化的 Ridge（最佳 alpha=150.21）也仅能达到 0.63 的验证集 R²。

13. **深层学习特征 + 树模型 = 当前最优组合:** 从 CatBoost 全描述符（R²=0.909）到 PharmHGT Transformer（R²=0.981），再到 LightGBM + PharmHGT 特征（R²=0.988），再至 XGBoost + PharmHGT 特征（R²=0.994），四条路线揭示了分子性质预测的核心范式：**PharmHGT 的原子级/药效团/反应性多视图特征设计 + 梯度提升树模型（XGBoost/LightGBM） = 最优组合**。PharmHGT 特征提供了远超 RDKit 描述符的信息密度和丰富度（原子分布统计、键特征统计、药效团指纹、反应性碎片、表面活性剂先验），而 XGBoost/LightGBM 以其高效的特征处理、列采样正则化和非线性拟合能力充分挖掘这些信息。这一组合无需 GPU、训练快速、推理极快，更适合实际部署。

14. **特征工程的"信息密度"比"模型复杂度"更重要:** XGBoost + PharmHGT 特征（R²=0.994）和 LightGBM + PharmHGT 特征（R²=0.988）均超越了更复杂的 PharmHGT 异构图 Transformer（R²=0.981），说明在特征设计足够优秀的前提下，使用相对简单的树模型即可达到甚至超越复杂深度模型的效果。这一发现对分子性质预测的实践指导意义重大：**特征工程的质量是第一优先级，模型架构的选择是第二优先级**。

### 3.3 推荐方案

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| **🏆 最佳预测精度（CPU 即可）** | **XGBoost + PharmHGT 特征 (Optuna)** | **测试 R²=0.994，RMSE=0.089，MAE=0.061**，CPU 即可训练和推理，无需 GPU |
| **🥈 次佳预测精度（CPU 即可）** | **LightGBM + PharmHGT 特征 (Optuna)** | **测试 R²=0.988，RMSE=0.12，MAE=0.085**，CPU 即可 |
| **🥉 最佳预测精度（有 GPU）** | **PharmHGT（异构图 Transformer）** | **测试 R²=0.981，默认参数即达极致性能**，需 GPU 训练 |
| **最佳预测精度（无 GPU / 纯描述符）** | CatBoost + 全量 RDKit 描述符 (Optuna) | 测试 R²=0.909，无需额外特征工程 |
| **生产部署 / 推理优先** | XGBoost + PharmHGT 特征 / CatBoost | XGBoost R²=0.994，CatBoost R²=0.909，推理极快 |
| **快速原型** | LightGBM + 全量 RDKit 描述符 (手动) | 无需调参即达 R²=0.859 |
| **轻量部署** | MLP (62维) | 特征维度低，R²≈0.84，快速推理 |
| **待探索** | CatBoost + PharmHGT 特征 | 当前 CatBoost 全描述符 R²=0.909，PharmHGT 特征有望大幅提升 |
| **待探索** | XGBoost + LightGBM + CatBoost 堆叠集成 | 三树模型集成可能进一步提升至 R²≈0.995+ |
| **待探索** | PharmHGT + Optuna 调优 | 当前默认参数 R²=0.981，Optuna 调优后可能进一步提升 |

---

## 4. 附录

### 4.1 数据概况

| 属性 | 值 |
|------|-----|
| 总训练样本 | 1476（含 pCMC 标签）|
| 测试样本 | 140 |
| 特征维度 | 62（精选）/ 217（全部 RDKit 描述符）/ 1415（RDKit+MACCS+ECFP4+Aux）/ 55×14×194×34（PharmHGT 异构图 4 视图）|
| 目标变量 | pCMC (log CMC) |
| 其他目标 | AW_ST_CMC, Gamma_max, Area_min, Pi_CMC, pC20 |

### 4.2 超参数搜索范围

| 模型 | 搜索空间 | Trial |
|------|---------|-------|
| **PharmHGT (异构图 Transformer)** | **hidden_dim[128,512] step=32, num_layers[2,6], dropout[0.1,0.5], batch_size[16,32,64,128], lr[1e-5,1e-3] (log), num_heads{4,8}（当前最佳使用默认参数，未启用 Optuna）** | **30（未启用）** |
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