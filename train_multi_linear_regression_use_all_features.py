import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==================== 数据加载 ====================
data_train_file_path = 'data/surfpro_train.csv'
df_train = pd.read_csv(data_train_file_path)
df_train = df_train.dropna(subset=['pCMC'])
print(f"Training samples (after pCMC NaN drop): {len(df_train)}")

# ==================== 特征工程 ====================
from smiles_to_features_all import smiles_to_features_all, get_all_descriptor_names

print("Converting SMILES to all RDKit descriptors (217-dim) ...")
x_train = np.array([smiles_to_features_all(smi) for smi in df_train["SMILES"]])
y_train = df_train['pCMC'].values

desc_names = get_all_descriptor_names()
print(f"Feature matrix: {x_train.shape}, Descriptors: {len(desc_names)}")
assert not np.any(np.isnan(x_train)), "NaN in feature matrix!"
assert not np.any(np.isinf(x_train)), "Inf in feature matrix!"

# ==================== 数据集划分 ====================
X_train, X_val, y_train_val, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape}, Val: {X_val.shape}")


# ==================== 评估函数 ====================
def evaluate(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  [{name}] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


# ==================== 标准化 ====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
x_all_scaled = scaler.fit_transform(x_train)  # for CV


# ==================== 1. OLS on scaled features ====================
print("\n" + "=" * 60)
print("1. OLS (Ordinary Least Squares, after scaling)")

ols = LinearRegression()
ols.fit(X_train_scaled, y_train_val)
evaluate(y_train_val, ols.predict(X_train_scaled), "Train")
evaluate(y_val, ols.predict(X_val_scaled), "Val")

# Note: OLS will still struggle due to near-singular matrix
# (217 features >> 1204 samples is fine, but multicollinearity is severe)


# ==================== 2. Ridge + Optuna ====================
print("\n" + "=" * 60)
print("2. Ridge Regression + Optuna tuning (after scaling)")

import optuna

N_TRIALS = 60

def objective(trial):
    alpha = trial.suggest_float("alpha", 1e-2, 1e4, log=True)
    model = Ridge(alpha=alpha, solver="sag", random_state=42)
    try:
        scores = cross_val_score(model, x_all_scaled, y_train, cv=3, scoring="r2", n_jobs=-1)
        return scores.mean()
    except Exception:
        return -1e30

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_alpha = study.best_params["alpha"]
print(f"\nBest alpha: {best_alpha:.4f}")
print(f"Best CV R2 (3-fold): {study.best_value:.4f}")

ridge_best = Ridge(alpha=best_alpha, solver="sag", random_state=42)
ridge_best.fit(X_train_scaled, y_train_val)
evaluate(y_train_val, ridge_best.predict(X_train_scaled), "Train (tuned)")
evaluate(y_val, ridge_best.predict(X_val_scaled), "Val (tuned)")

cv_ridge = cross_val_score(
    Ridge(alpha=best_alpha, solver="sag", random_state=42),
    x_all_scaled, y_train, cv=5, scoring="r2",
)
print(f"CV R2 (tuned): {cv_ridge}")
valid = cv_ridge[np.isfinite(cv_ridge) & (cv_ridge > -10)]
if len(valid) > 0:
    print(f"CV R2 mean valid folds: {valid.mean():.4f} +/- {valid.std():.4f}")


# ==================== 3. PCA + LinearRegression ====================
print("\n" + "=" * 60)
print("3. PCA + Linear Regression")

pca = PCA(n_components=0.95).fit(X_train_scaled)
n_components = pca.n_components_
print(f"PCA retains {n_components} components (95% variance)")

X_train_pca = pca.transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
x_all_pca = pca.transform(x_all_scaled)

lr_pca = LinearRegression()
lr_pca.fit(X_train_pca, y_train_val)
evaluate(y_train_val, lr_pca.predict(X_train_pca), "Train")
evaluate(y_val, lr_pca.predict(X_val_pca), "Val")

cv_pca = cross_val_score(LinearRegression(), x_all_pca, y_train, cv=5, scoring="r2")
print(f"CV R2: {cv_pca}")
print(f"CV R2 mean: {cv_pca.mean():.4f} +/- {cv_pca.std():.4f}")
