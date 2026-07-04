from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np


# 所有可用 RDKit 描述符列表: [(name, func), ...]
_ALL_DESCRIPTORS = Descriptors.descList


def smiles_to_features(smiles: str) -> np.ndarray:
    """从 SMILES 字符串计算全部 RDKit 分子描述符并返回特征向量。

    使用 rdkit.Chem.Descriptors.descList 中的所有描述符（约 200+ 个）。

    Args:
        smiles: SMILES 字符串。

    Returns:
        包含所有描述符值的 NumPy 数组（长度 = len(_ALL_DESCRIPTORS)）。
        如果 SMILES 无效，返回全零数组。
        单个描述符计算失败时以 0.0 填充。
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return np.zeros(len(_ALL_DESCRIPTORS))

    features = []
    for _, func in _ALL_DESCRIPTORS:
        try:
            val = func(mol)
            if val is None:
                val = 0.0
        except Exception:
            val = 0.0
        features.append(val)

    return np.array(features)


def get_descriptor_names() -> list:
    """返回描述符名称列表，与 smiles_to_features 的输出一一对应。"""
    return [name for name, _ in _ALL_DESCRIPTORS]


if __name__ == "__main__":
    # 测试示例
    test_smiles = "CC(C)(C)OCCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]"
    result = smiles_to_features(test_smiles)
    print(f"特征维度: {result.shape[0]}")
    print(f"描述符数量: {len(get_descriptor_names())}")
    print(f"前 10 个值: {result[:10]}")
    if result.size > 0:
        print(f"有效特征，非零个数: {np.count_nonzero(result)}")
    else:
        print("无效的 SMILES 字符串")


if __name__ == "__main__":
    # 测试示例
    test_smiles = "CC(C)(C)OCCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]"
    result = smiles_to_features(test_smiles)
    if result.size > 0:
        print(f"特征向量: {result}")
    else:
        print("无效的 SMILES 字符串")