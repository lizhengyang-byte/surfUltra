import pandas as pd
import numpy as np
from smiles_to_features import smiles_to_features

data_train_file_path = 'data/surfpro_train.csv'
data_test_file_path = 'data/surfpro_test.csv'

# 数据清洗
df_train = pd.read_csv(data_train_file_path)
df_train = df_train.dropna(subset=['pCMC'])

# 将SMILES逐条转换为特征
x_train = np.array([smiles_to_features(smi) for smi in df_train["SMILES"]])
y_train = df_train['pCMC'].values

# ==================== SVR 训练 ====================
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# SVR 对特征尺度敏感，先标准化
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

X_train, X_val, y_train_val, y_val = train_test_split(
    x_train_scaled, y_train, test_size=0.2, random_state=42
)

model = SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=0.1)
model.fit(X_train, y_train_val)


def evaluate(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  [{name}] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")


evaluate(y_train_val, model.predict(X_train), "Train")
evaluate(y_val, model.predict(X_val), "Val")

cv_scores = cross_val_score(
    SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=0.1),
    x_train_scaled, y_train, cv=5, scoring="r2",
)
print(f"CV R²: {cv_scores}")
print(f"CV R² 均值: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ==================== 超参数调优（Optuna） ====================
print("\n" + "=" * 60)
print("超参数搜索: Optuna (TPE) ...")

import optuna

N_TRIALS = 60


def objective(trial):
    kernel = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])
    C = trial.suggest_float("C", 0.01, 1000, log=True)
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    epsilon = trial.suggest_float("epsilon", 0.001, 1.0, log=True)

    if kernel == "poly":
        degree = trial.suggest_int("degree", 2, 5)
        svr = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon, degree=degree)
    else:
        svr = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)

    scores = cross_val_score(svr, x_train_scaled, y_train, cv=3, scoring="r2", n_jobs=-1)
    return scores.mean()


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_cfg = study.best_params
print(f"\n最佳参数: {best_cfg}")
print(f"最佳 CV R² (3折): {study.best_value:.4f}")

kernel = best_cfg.pop("degree", None)
best = SVR(**best_cfg)
if kernel is not None:
    best = SVR(**best_cfg, degree=kernel)
best.fit(X_train, y_train_val)

evaluate(y_train_val, best.predict(X_train), "Train (tuned)")
evaluate(y_val, best.predict(X_val), "Val (tuned)")

tune_params = {k: v for k, v in study.best_params.items()}
deg = tune_params.pop("degree", None)
best_cv = SVR(**tune_params)
if deg is not None:
    best_cv = SVR(**tune_params, degree=deg)

cv_best = cross_val_score(best_cv, x_train_scaled, y_train, cv=5, scoring="r2")
print(f"CV R² (tuned): {cv_best}")
print(f"CV R² 均值 (tuned): {cv_best.mean():.4f} ± {cv_best.std():.4f}")