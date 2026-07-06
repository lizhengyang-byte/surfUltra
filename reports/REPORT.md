# SurfPredict 模型训练报告

> **目标变量:** pCMC（临界胶束浓度对数值）  
> **数据集:** 1335 条训练样本，1204 条有效（含 pCMC 标签）  
> **特征:** 62 维精选 / 217 维全量 RDKit 分子描述符（XGBoost 经特征选择保留 109 维）  

---

## 1. 模型概览与性能对比

### 1.1 测试集主指标对比

| 模型 | 框架 | RMSE ↓ | MAE ↓ | R² ↑ | 调优 Trial |
|------|------|--------|-------|------|-----------|
| **LightGBM (全描述符, Optuna)** | LightGBM | **0.3525** | **0.2391** | **0.8994** | 50 |
| **MLP (全描述符)** | PyTorch | 0.4083 | 0.2525 | 0.8650 | 30 |
| **LightGBM (全描述符, 手动)** | LightGBM | 0.4179 | 0.2742 | 0.8586 | 手动调参 |
| **XGBoost (全描述符, 特征选择)** | XGBoost | 0.4053 | 0.2736 | **0.8670** | 100 |
| **MLP (Keras, 62维)** | TensorFlow/Keras | 0.4474 | 0.3294 | 0.8399 | 30 |
| **MLP (PyTorch, 62维)** | PyTorch | 0.4517 | 0.3231 | 0.8369 | 30 |
| **RNN (PyTorch LSTM, 全描述符)** | PyTorch | 0.4610 | 0.3152 | 0.8279 | 30 |
| **RNN (Keras, 62维)** | Keras | 0.4849 | 0.3646 | 0.8120 | 30 |
| **SVR (RBF)** | scikit-learn | 0.5274 (Val) | 0.3978 | 0.7835 (Val) | 60 |
| **LightGBM (62维, 调优)** | LightGBM | — | — | 0.4035 ± 0.1774 (CV) | 60 |
| **AttentiveFP (GNN)** | PyG | *待运行* | *待运行* | *待运行* | 30 |

> **注:** 除 SVR 仅输出验证集指标外，其余模型均报告测试集结果。  
> "MLP (全描述符)"、"LightGBM (全描述符)"、"XGBoost (全描述符, 特征选择)" 和 "RNN (PyTorch LSTM, 全描述符)" 使用全部 217 维 RDKit 描述符（XGBoost 进一步经特征选择保留 109 维）；其余模型基于 62 维精选描述符。

### 1.2 详细分集指标

| 模型 | 数据集 | RMSE | MAE | R² |
|------|--------|------|-----|----|
| **LightGBM (全描述符, Optuna)** | Train (all) | 0.1563 | 0.1037 | 0.9792 |
| | Test | **0.3525** | **0.2391** | **0.8994** |
| | CV (5-fold) | — | — | 0.4042 ± 0.2155 |
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

#### 2.2.1 LightGBM (全描述符, Optuna 调优) — 当前最佳

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

#### 2.2.3 LightGBM (62维, Optuna 调优) — 旧版本

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

### 2.9 AttentiveFP (GNN)

- **模型:** PyTorch Geometric AttentiveFP
- **特征:** 分子图（39 维原子特征 + 11 维键特征），在线生成，未预缓存
- **状态:** ⏳ 尚未运行 / 数据未记录

---

## 3. 综合分析与结论

### 3.1 模型排名（按测试集 R²）

| 排名 | 模型 | Test R² | 特点 |
|------|------|---------|------|
| **1** | **LightGBM (全描述符, Optuna)** | **0.8994** | Optuna 50 轮调参，无需 GPU，接近 0.90 |
| 2 | XGBoost (全描述符, 特征选择) | 0.8670 | 217→109 维特征选择 + Optuna 100 轮 |
| 3 | MLP (全描述符) | 0.8650 | 全部 217 维 RDKit 描述符 |
| 4 | LightGBM (全描述符, 手动) | 0.8586 | 手动调参，无需 GPU |
| 5 | MLP (Keras, 62维) | 0.8399 | 3 层 MLP，Adam 优化器 |
| 6 | MLP (PyTorch, 62维) | 0.8369 | 3 层 MLP，AdamW 优化器 |
| 7 | RNN (PyTorch LSTM, 全描述符) | 0.8279 | 序列化 217 维描述符，3 层 LSTM |
| 8 | RNN (Keras, 2 层) | 0.8120 | 更轻量，泛化稳定 |
| 9 | SVR (RBF) | 0.7835 (Val) | 非深度学习基线，有一定预测能力 |

### 3.2 关键发现

1. **LightGBM + Optuna 全面领先:** 通过 50 轮 Optuna 优化（18 个参数、5-Fold CV 目标），LightGBM 测试 R² 达到 **0.8994**，较手动调参版本（0.8586）提升约 0.04，较此前最佳 MLP 全描述符（0.8650）提升约 0.034。Optuna 自动选择了 gbdt 模式 + 强正则化策略，实现了当前最佳泛化性能。

2. **全量描述符优势显著:** 使用全部 217 维 RDKit 描述符的各模型（LightGBM R²=0.899/0.859, MLP R²=0.865, RNN R²=0.828）均超越 62 维精选描述符的最佳结果（0.84），表明更丰富的描述符集合包含了更多有用的结构-性质关系信息。

3. **LightGBM 全描述符表现亮眼:** LightGBM 手动调参即达测试 R²=**0.8586**，Optuna 优化后进一步提升至 **0.8994**，且无需 GPU，训练集 R²=0.979 过拟合控制优于 MLP（0.988）和 XGBoost（0.990），是生产环境的最优选择。

4. **梯度提升树 vs 神经网络:** XGBoost 全描述符版本测试 R²=0.867，略高于 MLP 全描述符（0.865），但 LightGBM（Optuna）以 0.899 全面领先。三种树模型（LightGBM Optuna 0.899 > LightGBM 手动 0.859 ≈ XGBoost 0.867）均表现出与神经网络相当的竞争力。

5. **特征工程有效性:** 62 维 RDKit 描述符+3 层 MLP 即可达到 R²≈0.84，217 维全描述符+LightGBM（Optuna）进一步提升至 0.90，表明充分优化的树模型可以从全量描述符中提取更多有效信息。

6. **序列建模 vs 全连接建模:** RNN (PyTorch LSTM) 使用同样的 217 维全描述符，但测试 R²=0.828，低于 MLP 的 0.865 和 LightGBM 的 0.899。这证实分子描述符之间无序，将其作为序列建模是次优策略。LightGBM 的树模型结构和 MLP 的并行特征处理更适合此类结构化描述符数据。

7. **Optuna 调参收益显著:** 从手动调参（R²=0.8586）到 50 轮 Optuna 优化（R²=0.8994），R² 提升 0.04。参数重要性分析显示 boosting_type（0.337）和 subsample（0.251）是最关键的调参方向，未来可针对性地进一步搜索。XGBoost 方面，reg_alpha（0.422）和 gamma（0.395）是最重要的超参数，说明正则化对控制 XGBoost 过拟合至关重要。

8. **SVR 局限:** 非线性核 SVM 在该任务中表现不如树模型和神经网络，可能与描述符空间维度及噪声有关。

### 3.3 推荐方案

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| **最佳预测精度** | LightGBM + 全量描述符 (Optuna) | 测试 R²=0.899，当前最高，无需 GPU |
| **生产部署 / 无需 GPU** | LightGBM + 全量描述符 (Optuna) | 测试 R²=0.899，训练和推理极快 |
| **快速原型** | LightGBM + 全量描述符 (手动) | 无需调参即达 R²=0.859 |
| **轻量部署** | MLP (62维) | 特征维度低，R²≈0.84，快速推理 |
| **待探索** | AttentiveFP (GNN) | 利用分子拓扑结构，可能超越描述符方法 |

---

## 4. 附录

### 4.1 数据概况

| 属性 | 值 |
|------|-----|
| 总训练样本 | 1335 |
| 含 pCMC 标签 | 1204 (90.2%) |
| 特征维度 | 62（精选）/ 217（全部 RDKit 描述符） |
| 目标变量 | pCMC (log CMC) |
| 其他目标 | AW_ST_CMC, Gamma_max, Area_min, Pi_CMC, pC20 |

### 4.2 超参数搜索范围

| 模型 | 搜索空间 | Trial |
|------|---------|-------|
| XGBoost (全描述符, 特征选择) | 特征选择（重要性≥中位数）→ max_depth[3,8], min_child_weight[1,50], gamma[0,10], lr[0.003,0.3], n_estimators[100,2000], subsample[0.4,1.0], colsample_bytree[0.3,1.0], colsample_bylevel[0.3,1.0], reg_lambda[0.1,50], reg_alpha[0.1,50], max_delta_step[0,10] | 100 |
| XGBoost (旧版, 62维) | max_depth[3,10], lr[0.005,0.1], subsample[0.5,1.0], colsample[0.5,1.0], reg_lambda[0.1,20], n_estimators[200,1000] | 60 |
| LightGBM (全描述符, Optuna) | boosting_type[gbdt,dart], max_depth[3,15], num_leaves[15,255], lr[0.001,0.3], n_estimators[500,3000], subsample[0.5,1.0], subsample_freq[1,10], colsample_bytree[0.3,1.0], feature_fraction[0.3,1.0], feature_fraction_bynode[0.3,1.0], reg_lambda[0,30], reg_alpha[0,30], min_child_weight[0.01,50], min_child_samples[1,50], min_data_in_leaf[1,100], min_split_gain[0,1] | 50 |
| LightGBM (全描述符, 手动) | 手动：max_depth=6, lr=0.05, subsample=0.8, colsample=0.8, reg_lambda=1.0, n_estimators=500 | 手动 |
| LightGBM (62维) | max_depth[3,10], lr[0.005,0.1], subsample[0.5,1.0], colsample[0.5,1.0], reg_lambda[0.1,20], n_estimators[200,1000] | 60 |
| SVR | kernel{rbf,poly,sigmoid}, C[0.01,1000], gamma{scale,auto}, epsilon[0.001,1.0] | 60 |
| MLP (PyTorch) | lr[1e-4,1e-2], dropout[0.1,0.4], wd[1e-6,1e-3], h1{128,256,512}, h2{64,128,256}, h3{32,64}, bs{16,32,64} | 30 |
| MLP (全描述符) | lr[1e-4,5e-3], dropout[0.1,0.4], wd[1e-6,1e-3], hidden{128,256,512}, bs{16,32,64} | 30 |
| RNN (PyTorch LSTM, 全描述符) | lr[1e-4,5e-3], dropout[0.05,0.4], wd[1e-6,1e-3], hidden_size{32,64,128}, num_layers[1,3], bs{16,32,64} | 30 |
| RNN (Keras) | lr[1e-4,1e-2], dropout[0.1,0.4], l2[1e-5,1e-3], units_1{64,128,256}, units_2{32,64,128}, bs{16,32,64} | 30 |
| AttentiveFP | lr[1e-4,5e-3], dropout[0.05,0.4], wd[1e-6,1e-3], hidden_dim{64,128,256}, num_layers[2,5], num_timesteps[2,4], bs{16,32,64} | 30 |