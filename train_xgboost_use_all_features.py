import pandas as pd
import numpy as np
from rdkit import Chem
from smiles_to_features_all import compute_all_descriptors
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

# ==================== XGBoost ====================
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

# ========== 特征标准化 ==========
scaler = StandardScaler()
X_train_all_scaled = scaler.fit_transform(X_train_all)
X_test_scaled = scaler.transform(X_test)


def evaluate(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  [{name}] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    return rmse, mae, r2


# ========== 特征选择：通过基线模型筛选重要特征 ==========
print("\n" + "=" * 60)
print("步骤1: 训练基线模型进行特征选择 ...")

# 先用一个有较强正则化的模型来选特征
selector_model = xgb.XGBRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
    random_state=42, verbosity=0,
)
selector_model.fit(X_train_all_scaled, y_train_full)

# 选择重要性高于中位数的特征
importance = selector_model.feature_importances_
median_importance = np.median(importance)
top_feature_idx = np.where(importance >= median_importance)[0]

print(f"全部特征数: {X_train_all_scaled.shape[1]}")
print(f"选择特征数(重要性≥中位数): {len(top_feature_idx)}")

X_train_selected = X_train_all_scaled[:, top_feature_idx]
X_test_selected = X_test_scaled[:, top_feature_idx]

# ========== 划分训练/验证 ==========
X_train, X_val, y_train_val, y_val = train_test_split(
    X_train_selected, y_train_full, test_size=0.2, random_state=42
)

# ========== 基线模型（用于对比）==========
base_model = xgb.XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    early_stopping_rounds=50, verbosity=0,
)
base_model.fit(X_train, y_train_val, eval_set=[(X_val, y_val)], verbose=0)

print("\n===== 基线模型（特征选择后）=====")
evaluate(y_train_val, base_model.predict(X_train), "Train")
evaluate(y_val, base_model.predict(X_val), "Val")

cv_scores = cross_val_score(
    xgb.XGBRegressor(
        **{k: v for k, v in base_model.get_params().items()
           if k != "early_stopping_rounds"}
    ), X_train_selected, y_train_full, cv=5, scoring="r2",
)
print(f"CV R²: {cv_scores}")
print(f"CV R² 均值: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ==================== 超参数调优（Optuna） ====================
print("\n" + "=" * 60)
print("步骤2: Optuna 超参数搜索（基于特征选择后的数据）...")

import optuna

N_TRIALS = 100


def objective(trial):
    params = {
        # 核心参数 - 控制模型复杂度
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "gamma":            trial.suggest_float("gamma", 0.0, 10.0),
        "learning_rate":    trial.suggest_float("learning_rate", 0.003, 0.3, log=True),
        "n_estimators":     trial.suggest_int("n_estimators", 100, 2000),
        # 采样
        "subsample":        trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),
        # 正则化 - 强力防过拟合
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 50.0, log=True),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.1, 50.0, log=True),
        "max_delta_step":   trial.suggest_float("max_delta_step", 0.0, 10.0),
    }
    model = xgb.XGBRegressor(random_state=42, verbosity=0, **params)
    scores = cross_val_score(model, X_train_selected, y_train_full, cv=5, scoring="r2", n_jobs=-1)
    return scores.mean()


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_cfg = study.best_params
print(f"\n最佳参数: {best_cfg}")
print(f"最佳 CV R² (5折): {study.best_value:.4f}")

print("\n参数重要性:")
try:
    importances = optuna.importance.get_param_importances(study)
    for param, importance in importances.items():
        print(f"  {param}: {importance:.4f}")
except Exception:
    pass

# ========== 在全部训练数据上训练最终模型 ==========
print("\n" + "=" * 60)
print("步骤3: 使用最佳超参数在全部训练集上训练最终模型 ...")

final_cfg = {k: v for k, v in best_cfg.items() if k != "n_estimators"}
final_model = xgb.XGBRegressor(
    random_state=42, verbosity=0,
    n_estimators=best_cfg["n_estimators"],
    **final_cfg,
)
final_model.fit(X_train_selected, y_train_full, verbose=0)

# ========== 测试集评估 ==========
print("\n测试集评估 (最终模型):")
test_pred = final_model.predict(X_test_selected)
test_rmse, test_mae, test_r2 = evaluate(y_test_orig, test_pred, "Test")

# 训练集评估
train_pred_all = final_model.predict(X_train_selected)
train_rmse_all = np.sqrt(mean_squared_error(y_train_full, train_pred_all))
train_mae_all  = mean_absolute_error(y_train_full, train_pred_all)
train_r2_all   = r2_score(y_train_full, train_pred_all)

# 5折CV验证
cv_best = cross_val_score(
    xgb.XGBRegressor(random_state=42, verbosity=0, **best_cfg),
    X_train_selected, y_train_full, cv=5, scoring="r2",
)
print(f"CV R² (tuned): {cv_best}")
print(f"CV R² 均值 (tuned): {cv_best.mean():.4f} ± {cv_best.std():.4f}")

# ========== 结果汇总 ==========
print(f"\n{'Model':<45} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print("-" * 73)
print(f"{'XGBoost (Train All, feature selected + tuned)':<45} {train_rmse_all:>8.4f} {train_mae_all:>8.4f} {train_r2_all:>8.4f}")
print(f"{'XGBoost (Test, feature selected + tuned)':<45} {test_rmse:>8.4f} {test_mae:>8.4f} {test_r2:>8.4f}")

print("-" * 73)
print(f"\n✅ 全部 RDKit 描述符 ({X_train_all.shape[1]} 维 → 筛选后 {X_train_selected.shape[1]} 维)")
print(f"   + XGBoost (Optuna 调参 {N_TRIALS} 轮)")
print(f"   训练集 R² = {train_r2_all:.4f}")
print(f"   测试集 R² = {test_r2:.4f}")

if test_r2 >= 0.95:
    print(f"\n🎯 目标达成！测试集 R² = {test_r2:.4f} >= 0.95")
else:
    print(f"\n⚠️ 当前测试集 R² = {test_r2:.4f}，目标为 0.95，仍需改进")
    print("建议尝试: 集成多个模型 / 使用 GNN / 特征工程优化")