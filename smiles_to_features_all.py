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
                descriptors.append(0.0)
        except Exception:
            descriptors.append(0.0)  # 计算失败则补 0
        names.append(desc_name)
    return np.array(descriptors), names


def smiles_to_features_all(smiles: str) -> np.ndarray:
    """从 SMILES 字符串计算所有 RDKit 分子描述符并返回特征向量。

    使用 RDKit 中所有可用的分子描述符（约 209 维）。

    Args:
        smiles: SMILES 字符串。

    Returns:
        所有描述符的 NumPy 数组。无效 SMILES 返回全零向量。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        feature_vector, _ = compute_all_descriptors(Chem.MolFromSmiles("C"))
        return np.zeros(len(feature_vector))
    feature_vector, _ = compute_all_descriptors(mol)
    return feature_vector


def get_all_descriptor_names() -> list:
    """返回所有描述符名称列表，与 smiles_to_features_all 的输出一一对应。"""
    mol = Chem.MolFromSmiles("C")
    _, names = compute_all_descriptors(mol)
    return names


if __name__ == "__main__":
    # 测试示例
    smiles = "CC(C)(C)OCCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]"
    mol = Chem.MolFromSmiles(smiles)

    if mol:
        feature_vector, desc_names = compute_all_descriptors(mol)
        print(f"特征向量维度: {len(feature_vector)}")
        print(f"特征向量 (前10个): {feature_vector[:10]}")
    else:
        print("无效的 SMILES 字符串")