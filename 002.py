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

# 描述性统计

## pCMC 分布
print("训练数据描述性统计:")
print(f"  样本数量: {len(y_train)}")
print(f"  pCMC 均值: {y_train.mean():.4f}")
print(f"  pCMC 标准差: {y_train.std():.4f}")
print(f"  pCMC 最小值: {y_train.min():.4f}")
print(f"  pCMC 最大值: {y_train.max():.4f}")

## x_train 分布
print(f"  特征维度: {x_train.shape[1]}")
print(f"  特征均值: {x_train.mean():.4f}")
print(f"  特征标准差: {x_train.std():.4f}")

