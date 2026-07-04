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

# ==================== LightGBM 训练 ====================

