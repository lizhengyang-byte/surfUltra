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

# ==================== XGBoost 训练 ====================
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X_train, X_val, y_train_val, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)

model = xgb.XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    early_stopping_rounds=50, verbosity=0,
)
model.fit(X_train, y_train_val, eval_set=[(X_val, y_val)], verbose=0)

def evaluate(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  [{name}] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

evaluate(y_train_val, model.predict(X_train), "Train")
evaluate(y_val, model.predict(X_val), "Val")

cv_scores = cross_val_score(
    xgb.XGBRegressor(
        **{k: v for k, v in model.get_params().items()
           if k != "early_stopping_rounds"}
    ), x_train, y_train, cv=5, scoring="r2",
)
print(f"CV R²: {cv_scores}")
print(f"CV R² 均值: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ==================== 超参数调优（Optuna） ====================
print("\n" + "=" * 60)
print("超参数搜索: Optuna (TPE) ...")

import optuna

N_TRIALS = 60


def objective(trial):
    params = {
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
        "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
    }
    model = xgb.XGBRegressor(random_state=42, verbosity=0, **params)
    scores = cross_val_score(model, x_train, y_train, cv=3, scoring="r2", n_jobs=-1)
    return scores.mean()


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_cfg = study.best_params
print(f"\n最佳参数: {best_cfg}")
print(f"最佳 CV R² (3折): {study.best_value:.4f}")

best = xgb.XGBRegressor(random_state=42, verbosity=0, early_stopping_rounds=50, **best_cfg)
best.fit(X_train, y_train_val, eval_set=[(X_val, y_val)], verbose=0)

evaluate(y_train_val, best.predict(X_train), "Train (tuned)")
evaluate(y_val, best.predict(X_val), "Val (tuned)")

cv_best = cross_val_score(
    xgb.XGBRegressor(random_state=42, verbosity=0, **best_cfg),
    x_train, y_train, cv=5, scoring="r2",
)
print(f"CV R² (tuned): {cv_best}")
print(f"CV R² 均值 (tuned): {cv_best.mean():.4f} ± {cv_best.std():.4f}")

