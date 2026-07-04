from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np


# 精选描述符列表（覆盖大小、疏水、极性、电荷、拓扑、官能团）
_SELECTED_DESCRIPTORS = [
    # --- 分子大小 / 分子量 ---
    "MolWt", "HeavyAtomMolWt", "ExactMolWt",
    "NumHeavyAtoms", "HeavyAtomCount",
    "NumRotatableBonds", "RingCount",
    "NumAromaticRings", "NumSaturatedRings", "NumAliphaticRings",
    "NumSaturatedCarbocycles", "NumSaturatedHeterocycles",
    "NumAromaticCarbocycles", "NumAromaticHeterocycles",
    # --- 疏水性 ---
    "MolLogP",
    # --- 极性 / 表面积 ---
    "TPSA", "LabuteASA",
    "NumHAcceptors", "NumHDonors", "NumHeteroatoms",
    "NHOHCount", "NOCount",
    # --- 电荷 / 电子 ---
    "MaxPartialCharge", "MinPartialCharge",
    "NumValenceElectrons",
    # --- 拓扑 / 形状 ---
    "FractionCsp3", "BertzCT", "HallKierAlpha",
    "Kappa1", "Kappa2", "Kappa3",
    "Chi0", "Chi1",
    "Chi0n", "Chi1n",
    "Chi0v", "Chi1v",
    "BalabanJ",
    # --- 部分电荷 VSA（极性分布） ---
    "PEOE_VSA1", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4",
    "PEOE_VSA5", "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8",
    "PEOE_VSA9", "PEOE_VSA10", "PEOE_VSA11", "PEOE_VSA12",
    "PEOE_VSA13", "PEOE_VSA14",
    # --- LogP VSA（疏水分布） ---
    "SlogP_VSA1", "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4",
    "SlogP_VSA5", "SlogP_VSA6", "SlogP_VSA7", "SlogP_VSA8",
    "SlogP_VSA9", "SlogP_VSA10", "SlogP_VSA11", "SlogP_VSA12",
]

# 按名称索引，避免每次遍历 descList
_DESC_MAP = {name: func for name, func in Descriptors.descList if name in _SELECTED_DESCRIPTORS}
_SELECTED_NAMES = list(_SELECTED_DESCRIPTORS)  # 保持有序


def smiles_to_features(smiles: str) -> np.ndarray:
    """从 SMILES 字符串计算精选 RDKit 分子描述符并返回特征向量。

    使用约 60 个手工挑选的描述符，覆盖分子大小、疏水性、极性、
    电荷、拓扑形状和官能团分布。

    Args:
        smiles: SMILES 字符串。

    Returns:
        精选描述符的 NumPy 数组。无效 SMILES 返回全零向量。
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return np.zeros(len(_SELECTED_NAMES))

    features = []
    for name in _SELECTED_NAMES:
        func = _DESC_MAP.get(name)
        if func is None:
            features.append(0.0)
            continue
        try:
            val = func(mol)
            if val is None or not np.isfinite(val):
                val = 0.0
        except Exception:
            val = 0.0
        features.append(val)

    arr = np.array(features, dtype=np.float64)
    np.nan_to_num(arr, copy=False)
    np.clip(arr, -1e10, 1e10, out=arr)
    return arr


def get_descriptor_names() -> list:
    """返回描述符名称列表，与 smiles_to_features 的输出一一对应。"""
    return list(_SELECTED_NAMES)


if __name__ == "__main__":
    # 测试示例
    test_smiles = "CC(C)(C)OCCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]"
    result = smiles_to_features(test_smiles)
    print(f"特征维度: {result.shape[0]}")
    print(f"描述符数量: {len(get_descriptor_names())}")
    print(f"前 10 个值: {result[:10]}")
    if result.size > 0:
        print(f"有效特征，非零个数: {np.count_nonzero(result)}")