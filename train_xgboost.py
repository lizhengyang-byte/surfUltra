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

