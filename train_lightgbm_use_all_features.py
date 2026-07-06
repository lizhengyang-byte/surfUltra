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

# ========== LightGBM 训练 ==========
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import optuna

# ========== 特征标准化 ==========
scaler = StandardScaler()
X_train_all_scaled = scaler.fit_transform(X_train_all)
X_test_scaled = scaler.transform(X_test)

# ========== Optuna 超参数优化（基于 5-Fold CV）==========
print("\n" + "=" * 60)
print("Optuna 超参数调优（50 轮，基于 5-Fold CV RMSE）...")

def objective(trial):
    # 核心参数
    boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "dart"])
    params = {
        "boosting_type": boosting_type,
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
        # 数据采样
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        # 特征采样
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0),
        "feature_fraction_bynode": trial.suggest_float("feature_fraction_bynode", 0.3, 1.0),
        # 正则化
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 30.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 30.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.01, 50.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 50),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        # Dart 特有
        "drop_rate": trial.suggest_float("drop_rate", 0.0, 0.3) if boosting_type == "dart" else 0.0,
        "max_drop": trial.suggest_int("max_drop", 1, 50) if boosting_type == "dart" else 50,
        "skip_drop": trial.suggest_float("skip_drop", 0.0, 0.5) if boosting_type == "dart" else 0.0,
    }

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rmse_scores = []
    for train_idx, val_idx in kfold.split(X_train_all_scaled):
        X_cv_train, X_cv_val = X_train_all_scaled[train_idx], X_train_all_scaled[val_idx]
        y_cv_train, y_cv_val = y_train_full[train_idx], y_train_full[val_idx]

        model_cv = lgb.LGBMRegressor(random_state=42, verbose=-1, **params)
        model_cv.fit(
            X_cv_train, y_cv_train,
            eval_set=[(X_cv_val, y_cv_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        pred = model_cv.predict(X_cv_val)
        cv_rmse_scores.append(np.sqrt(mean_squared_error(y_cv_val, pred)))

    mean_cv_rmse = np.mean(cv_rmse_scores)
    std_cv_rmse = np.std(cv_rmse_scores)
    trial.set_user_attr("cv_rmse_std", std_cv_rmse)
    trial.set_user_attr("cv_rmse_scores", cv_rmse_scores)
    return mean_cv_rmse


study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50)

# 打印所有 trial 的 CV 详情
print("\n" + "-" * 60)
print("各 Trial CV 结果:")
print(f"{'Trial':>5} {'Mean RMSE':>10} {'Std RMSE':>9}  {'CV Scores'}")
print("-" * 60)
for t in study.trials:
    if t.values is not None:
        scores = t.user_attrs.get("cv_rmse_scores", [])
        scores_str = ", ".join(f"{s:.4f}" for s in scores)
        print(f"{t.number:>5} {t.values[0]:>10.4f} {t.user_attrs.get('cv_rmse_std', 0):>9.4f}  [{scores_str}]")
print("-" * 60)

best_params = study.best_params.copy()
print(f"\n最佳超参数（基于 5-Fold CV RMSE = {study.best_value:.4f}）:")
for k, v in best_params.items():
    print(f"  {k} = {v}")

print(f"\n各参数重要性:")
# 打印 Optuna 参数重要性（如果可用）
try:
    importances = optuna.importance.get_param_importances(study)
    for param, importance in importances.items():
        print(f"  {param}: {importance:.4f}")
except Exception:
    pass

# ========== 使用最佳参数在全部训练数据上训练最终模型 ==========
print("\n" + "=" * 60)
print("使用最佳超参数在全部训练数据上训练最终模型 ...")

# 最终模型使用更多迭代次数，利用早停确定最优轮数
final_n_estimators = max(best_params.get("n_estimators", 1000), 3000)
best_params_for_final = best_params.copy()
best_params_for_final["n_estimators"] = final_n_estimators

model = lgb.LGBMRegressor(random_state=42, verbose=-1, **best_params_for_final)
model.fit(
    X_train_all_scaled, y_train_full,
    eval_set=[(X_train_all_scaled, y_train_full)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
)


def evaluate(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  [{name}] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    return rmse, mae, r2


train_pred = model.predict(X_train_all_scaled)
train_rmse, train_mae, train_r2 = evaluate(y_train_full, train_pred, "Train (all)")

# 5-Fold CV 评估最终参数
cv_scores = cross_val_score(
    lgb.LGBMRegressor(random_state=42, verbose=-1, **best_params),
    X_train_all_scaled, y_train_full, cv=5, scoring="r2",
)
print(f"\nCV R²: {cv_scores}")
print(f"CV R² 均值: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ========== 测试集评估 ==========
print("\n" + "=" * 60)
test_pred = model.predict(X_test_scaled)
test_rmse = np.sqrt(mean_squared_error(y_test_orig, test_pred))
test_mae  = mean_absolute_error(y_test_orig, test_pred)
test_r2   = r2_score(y_test_orig, test_pred)
print(f"  [Test] RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")

# ========== 结果汇总 ==========
print(f"\n{'Model':<35} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print("-" * 62)
print(f"{'LightGBM (Train All)':<35} {train_rmse:>8.4f} {train_mae:>8.4f} {train_r2:>8.4f}")
print(f"{'LightGBM (Test)':<35} {test_rmse:>8.4f} {test_mae:>8.4f} {test_r2:>8.4f}")

print("-" * 62)
print(f"\n✅ 全部 RDKit 描述符 ({X_train_all.shape[1]} 维) + LightGBM (Optuna 调参，50 轮)")
print(f"   训练集 R² = {train_r2:.4f}")
print(f"   测试集 R² = {test_r2:.4f}")

# 特征重要性
print(f"\n📊 特征重要性（Top 20）:")
feature_importance = model.feature_importances_
top20_idx = np.argsort(feature_importance)[-20:][::-1]
print(f"{'Rank':>4} {'Feature':<6} {'Importance':>10}")
print("-" * 24)
for rank, idx in enumerate(top20_idx, 1):
    print(f"{rank:>4} {'idx_'+str(idx):<6} {feature_importance[idx]:>10}")

# 与其他模型对比
print(f"\n📊 与其他模型对比:")
print(f"{'Model':<35} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print("-" * 62)
print(f"{'MLP (全描述符)':<35} {'0.4083':>8} {'0.2525':>8} {'0.8650':>8}")
print(f"{'LightGBM (本模型)':<35} {test_rmse:>8.4f} {test_mae:>8.4f} {test_r2:>8.4f}")