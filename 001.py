from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from rdkit import Chem
from scikit_mol.fingerprints import MorganFingerprintTransformer
import pandas as pd

import pandas as pd
import numpy as np
from smiles_to_features import smiles_to_features

data_train_file_path = 'data/surfpro_train.csv'
data_test_file_path = 'data/surfpro_test.csv'

# 数据清洗
df_train = pd.read_csv(data_train_file_path)
df_train = df_train.dropna(subset=['pCMC'])

y_train = df_train['pCMC'].values

# 将SMILES转换为分子对象（用于指纹生成）
smiles_list_train = df_train["SMILES"].tolist()
mol_list_train = [Chem.MolFromSmiles(s) for s in smiles_list_train]

# 剔除解析失败的分子（若有）
valid = [(m, i) for i, m in enumerate(mol_list_train) if m is not None]
mol_list_train = [v[0] for v in valid]
y_train = y_train[[v[1] for v in valid]]

# 将SMILES逐条转换为特征（用于描述符分析，此处未使用但保留）
x_features_train = np.array([smiles_to_features(smi) for smi in smiles_list_train])

# 构建流水线：先进行指纹转换，再进行回归
pipe = Pipeline([
    ('mol_transformer', MorganFingerprintTransformer(radius=2, nBits=2048)),
    ('regressor', Ridge())
])

# 训练模型
pipe.fit(mol_list_train, y_train)

# ========== 测试集评估 ==========
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df_test = pd.read_csv(data_test_file_path)
df_test = df_test.dropna(subset=['pCMC'])

smiles_list_test = df_test["SMILES"].tolist()
mol_list_test = [Chem.MolFromSmiles(s) for s in smiles_list_test]

valid_test = [(m, i) for i, m in enumerate(mol_list_test) if m is not None]
mol_list_test = [v[0] for v in valid_test]
y_test = df_test['pCMC'].values[[v[1] for v in valid_test]]

y_pred = pipe.predict(mol_list_test)

rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae = float(mean_absolute_error(y_test, y_pred))
r2 = float(r2_score(y_test, y_pred))

print(f"Test  RMSE: {rmse:.4f}")
print(f"Test  MAE:  {mae:.4f}")
print(f"Test  R^2:   {r2:.4f}")

# 预测新分子
