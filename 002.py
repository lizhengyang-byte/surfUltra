from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from rdkit import Chem
from scikit_mol.fingerprints import MorganFingerprintTransformer
import pandas as pd
import numpy as np

# ========== 加载数据 ==========
# df_train = pd.read_csv('data/surfpro_train.csv').dropna(subset=['pCMC'])
df_train = pd.read_csv('data/surfpro_imputed.csv').dropna(subset=['pCMC'])
df_test  = pd.read_csv('data/surfpro_test.csv').dropna(subset=['pCMC'])

y_train = df_train['pCMC'].values
y_test  = df_test['pCMC'].values

# SMILES → Mol 对象
def smiles_to_mols(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid = [(m, i) for i, m in enumerate(mols) if m is not None]
    return [v[0] for v in valid], [v[1] for v in valid]

mol_train, idx_train = smiles_to_mols(df_train["SMILES"].tolist())
mol_test,  idx_test  = smiles_to_mols(df_test["SMILES"].tolist())
y_train = y_train[idx_train]
y_test  = y_test[idx_test]

# ========== 定义所有模型 ==========
models = {
    "Ridge": Pipeline([
        ('mol_transformer', MorganFingerprintTransformer(radius=2, nBits=2048)),
        ('regressor', Ridge())
    ]),
    "RandomForest": Pipeline([
        ('mol_transformer', MorganFingerprintTransformer(radius=2, nBits=2048)),
        ('regressor', RandomForestRegressor(n_estimators=300, random_state=42))
    ]),
    "GradientBoosting": Pipeline([
        ('mol_transformer', MorganFingerprintTransformer(radius=2, nBits=2048)),
        ('regressor', GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42))
    ]),
    "SVR": Pipeline([
        ('mol_transformer', MorganFingerprintTransformer(radius=2, nBits=2048)),
        ('scaler', StandardScaler()),
        ('regressor', SVR(kernel='rbf', C=10, gamma='scale'))
    ]),
    "KNN": Pipeline([
        ('mol_transformer', MorganFingerprintTransformer(radius=2, nBits=2048)),
        ('scaler', StandardScaler()),
        ('regressor', KNeighborsRegressor(n_neighbors=5))
    ]),
    "XGBoost": Pipeline([
        ('mol_transformer', MorganFingerprintTransformer(radius=2, nBits=2048)),
        ('regressor', __import__('xgboost').XGBRegressor(
            n_estimators=300, learning_rate=0.1, random_state=42, verbosity=0
        ))
    ]),
    "LightGBM": Pipeline([
        ('mol_transformer', MorganFingerprintTransformer(radius=2, nBits=2048)),
        ('regressor', __import__('lightgbm').LGBMRegressor(
            n_estimators=300, learning_rate=0.1, random_state=42, verbose=-1
        ))
    ]),
}

# ========== 逐个训练 & 评估 ==========
results = []
print(f"{'Model':<20} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print("-" * 48)

for name, pipe in models.items():
    pipe.fit(mol_train, y_train)
    y_pred = pipe.predict(mol_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))
    results.append((name, rmse, mae, r2))

    print(f"{name:<20} {rmse:>8.4f} {mae:>8.4f} {r2:>8.4f}")

# ========== 按 RMSE 排序 ==========
print("\n📊 模型排名（RMSE 升序）:")
results.sort(key=lambda x: x[1])
print(f"{'Rank':>4} {'Model':<20} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print("-" * 52)
for rank, (name, rmse, mae, r2) in enumerate(results, 1):
    print(f"{rank:>4} {name:<20} {rmse:>8.4f} {mae:>8.4f} {r2:>8.4f}")