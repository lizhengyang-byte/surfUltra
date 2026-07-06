from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np

def compute_all_descriptors(mol):
    """计算分子所有可用的 RDKit 描述符，返回特征向量和描述符名称列表"""
    descriptors = []
    names = []
    for desc_name, func in Descriptors.descList:
        try:
            val = func(mol)
            # 确保数值有效（有些可能返回 None 或 NaN）
            if val is not None and np.isfinite(val):
                descriptors.append(val)
            else:
                descriptors.append(0.0)  # 或者使用 np.nan，但后续可能需要填充
        except Exception:
            descriptors.append(0.0)  # 计算失败则补 0
        names.append(desc_name)
    return np.array(descriptors), names

# 1. 从 SMILES 创建分子对象
smiles = "CC(C)(C)OCCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]"
mol = Chem.MolFromSmiles(smiles)

if mol:
    # 2. 计算所有描述符
    feature_vector, desc_names = compute_all_descriptors(mol)
    
    # 可选：打印每个描述符的名称和值（便于查看）
    # for name, val in zip(desc_names, feature_vector):
    #     print(f"{name}: {val}")
    
    print(f"特征向量维度: {len(feature_vector)}")
    print(f"特征向量 (前10个): {feature_vector[:10]}")
else:
    print("无效的 SMILES 字符串")