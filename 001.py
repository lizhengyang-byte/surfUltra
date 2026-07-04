from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from rdkit import Chem
from scikit_mol.fingerprints import MorganFingerprintTransformer
import pandas as pd

data_train_file_path = 'data/surfpro_train.csv'
data_test_file_path = 'data/surfpro_test.csv'

smiles_list = pd.read_csv(data_train_file_path)['SMILES'].tolist()
y_train = pd.read_csv(data_train_file_path)['pCMC'].tolist()

# 将SMILES转换为分子对象
mol_list_train = [Chem.MolFromSmiles(s) for s in smiles_list]

# 构建流水线：先进行指纹转换，再进行回归
pipe = Pipeline([
    ('mol_transformer', MorganFingerprintTransformer(radius=2, nBits=2048)),
    ('regressor', Ridge())
])

# 训练模型
pipe.fit(mol_list_train, y_train)

# 预测新分子
new_mol = Chem.MolFromSmiles('c1ccccc1C(=O)C')
prediction = pipe.predict([new_mol])
print(f"预测值: {prediction[0]}")