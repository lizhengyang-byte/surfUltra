# XGBoost + PharmHGT 训练优化计划

## Context

两次独立运行 XGBoost + PharmHGT 522 维特征，CV RMSE 几乎相同（0.4571 vs 0.4602），但测试 R² 差异巨大（0.9936 vs 0.9761）。这说明：

1. **CV RMSE 对泛化性能的指示性不够敏感**——参数组合的微小 CV 差异可能对应巨大的测试集差异
2. **Optuna 50 次搜索采样不够充分**，容易陷入局部最优
3. **最终训练策略有缺陷**——`best_params["iterations"]` 是 CatBoost 参数，XGBoost 忽略它，导致最终模型实际使用 Optuna 搜索到的 `n_estimators` 而非预期的 3000

目标：修改训练脚本，提高找到最优解（R² ≥ 0.9936）的可靠性和概率，即使单次 50 轮运行也能稳定达到接近最优的结果。

## 诊断总结

| 项目 | 旧版最佳 | 新版这次 |
|------|---------|---------|
| Test R² | 0.9936 | 0.9761 |
| CV RMSE | 0.4571 | 0.4602 |
| 差异核心 | 低 reg_lambda (0.002)、低 gamma (0.04)、高 subsample (0.75) | 高 reg_lambda (0.37)、高 gamma (0.21)、低 subsample (0.62) |
| CV 差异 | 仅 0.003 | — |

## 计划修改

### 文件修改

只改一个文件：`train_xgboost_use_pharmhgt_features.py`

### 1. 增加 Optuna 搜索量（50 → 200 trials）

变量 `N_OPTUNA_TRIALS = 200`。同时增加 `N_STARTUP_TRIALS=10` 让 TPE 有更多采样积累期。

### 2. 启用多变量 TPE 采样

```python
sampler = optuna.samplers.TPESampler(multivariate=True, seed=SEED, n_startup_trials=10)
```

`multivariate=True` 让 TPE 学习参数之间的联合分布（例如 `learning_rate` 与 `n_estimators` 的负相关关系），比独立的一维 Parzen 估计更高效。

### 3. 修复 `iterations` 参数错误

```python
# 旧代码（有 bug）：XGBoost 没有 iterations 参数
final_iterations = max(best_params.get("iterations", 1000), 3000)
best_params["iterations"] = final_iterations

# 新代码：使用 n_estimators，确保足够大
final_n_estimators = max(best_params["n_estimators"], 3000)
best_params["n_estimators"] = final_n_estimators
```

### 4. 精炼搜索空间

基于两个最佳运行的实际参数分析，调整搜索范围：

| 参数 | 旧范围 | 新范围 | 理由 |
|------|--------|--------|------|
| `n_estimators` | [500, 3000] | [800, 3000] | 最佳值 1881/2459，下限太低浪费搜索 |
| `max_depth` | [3, 15] | [4, 12] | 最佳值 10/13，3 和 15 很少最优 |
| `learning_rate` | [0.005, 0.3] log | [0.01, 0.2] log | 最佳值 0.0755/0.0475 |
| `subsample` | [0.5, 1.0] | [0.6, 1.0] | 最佳值 0.75/0.62 |
| `min_child_weight` | [1, 50] log | [1, 30] log | 最佳值 10.6/11.4 |
| `gamma` | [0, 5.0] | [0, 2.0] | 最佳值 0.04/0.21 |
| `reg_alpha` | [1e-8, 10] log | 不变 | — |
| `reg_lambda` | [1e-8, 10] log | 不变 | — |
| `max_delta_step` | [0, 10] | [0, 8] | 最佳值 6.8/9.5 |
| `colsample_by*` | [0.3, 1.0] | 不变 | — |
| `booster` | [gbtree, dart] | 不变 | — |

### 5. 最终训练使用全部数据（不保留验证集）

当前做法：
- Optuna CV 使用全部 1476 条 → 选 best params
- 最终训练：在 1291 条上训练，185 条验证集做 early stopping → 测试集评估

问题：最终模型是在 1291 条而非 1476 条上训练的，浪费了数据。

新做法：
- Optuna 结束后，**用全部 1476 条数据**训练最终模型
- 使用 `early_stopping_rounds` 但传给 `eval_set=[(X_test, y_test)]` 仅用于监控（不做早停决策），或使用 `X_full` 作为训练集不做 early stopping（XGBoost 可设 `early_stopping_rounds=None` 或直接不传 `eval_set`）

实际上更安全的做法：**继承当前结构，但 final training 使用全部 X_full 数据，不传入 eval_set，直接训练 best_params["n_estimators"] 轮**。

### 6. 增加 Top-K CV 候选的 holdout 验证环节

主要改动：
- 从 `train_test_split` 中保留一个 holdout 验证集（占 10%，约 148 条）
- Optuna 使用剩余 1328 条 + 5-Fold CV（约 1062 训练 / 266 验证） 
- Optuna 完成后，对 Top-5 最佳参数组合，在 holdout 验证集上评估 RMSE
- 选择 holdout RMSE 最低的参数组合进行最终训练
- 最终训练使用全部 1476 条

这解决了**"类似 CV 分数但差异极大的测试结果"**的核心问题——通过一个独立的 holdout 验证集从 Top 候选者中做二次筛选。

### 7. 增加 CV 内的泛化差距监控

在 CV 中同时记录训练集和验证集的 RMSE，如果训练 RMSE 远低于验证 RMSE（差距 > 0.3），降低该 trial 的评分以惩罚过拟合：

```python
# 在 CV fold 内
train_pred = model_cv.predict(X_tr_cv)
train_rmse = np.sqrt(mean_squared_error(y_tr_cv, train_pred))
gap = train_rmse - val_rmse  # 过拟合量
adjusted_score = val_rmse * (1 + 0.1 * max(0, gap - 0.3))
```

不过这可能会让优化过程偏向保守。改为更简单的做法：在 objective 中返回 `(val_rmse + 0.05 * max(0, train_rmse - val_rmse - 0.3))` 作为调整后的评分。

## 不做的改动

- 不改 `smiles_to_features_pharmhgt.py`（特征工程不变，保持可复现性）
- 不改其他训练脚本（只聚焦 XGBoost）
- 不修改 REPORT.md（优化完成后用户可自行更新）

## 验证方式

```bash
# 运行优化后的脚本，预期结果：
python train_xgboost_use_pharmhgt_features.py

# 期望产出：
# - Test R² >= 0.993（接近或超过历史最优 0.9936）
# - 与旧版最优的 CV RMSE 相当（~0.457）
# - 最终模型训练在全部 1476 条数据上完成
```

建议连续运行 2-3 次（仅需修改 `SEED` 参数），验证结果是否稳定。如果 3 次中有 2 次达到 R² ≥ 0.993，说明优化有效。
