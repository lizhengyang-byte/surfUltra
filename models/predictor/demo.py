"""
demo.py — Usage examples for the pCMC prediction API.

Demonstrates:
  - Batch CSV prediction via predictor()
  - Single SMILES prediction via predictor_single()
  - Multi-SMILES list prediction via predictor_single()
  - Empty list handling via predictor_single()
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from models.predictor.API_predictor import predictor, predictor_single
from models.predictor.model_loader import get_available_models

print("=" * 60)
print("Available models:", get_available_models())
print("=" * 60)

# ===========================================================================
# 1. Batch CSV prediction (original workflow)
# ===========================================================================
# print("\n>>> 1. Batch CSV prediction (predictor)")
# input_path = Path(__file__).parent / "input.csv.csv"

# # 使用全部模型（集成均值）
# predictor(str(input_path), "output_all.csv", "all")

# # 指定单个模型
# predictor(str(input_path), "output_catboost_pharmhgt.csv",
#           "catboost_pharmhgt", device="cpu")

# # 自定义 SMILES 列名
# predictor(str(input_path), "out_all_default_SMILES_column.csv",
#           "all", smiles_col="SMILES")

# ===========================================================================
# 2. Single SMILES string prediction (new)
# ===========================================================================
print("\n>>> 2. Single SMILES prediction (predictor_single)")

# 单个分子式预测 — 传入字符串
result = predictor_single("CCO", "catboost_pharmhgt", device="cpu")
print(f"Result: {result['predicted_pCMC'][0]:.4f}")

# 单个分子式 + 全部模型集成
result = predictor_single("CCCCCCCCCCCCCS(=O)(=O)[O-].[Na+]",
                          "all", device="cpu")
print(f"Ensemble: {result['predicted_pCMC'][0]:.4f}")

# ===========================================================================
# 3. Multi-SMILES list prediction (new)
# ===========================================================================
print("\n>>> 3. Multi-SMILES list prediction (predictor_single)")

test_smiles = [
    "CCO",                              # ethanol
    "CCCCCCCCCCCCCS(=O)(=O)[O-].[Na+]",  # SDS
    "CC(=O)O",                          # acetic acid
    #"invalid_smiles_xxx",               # intentionally invalid
]
result = predictor_single(test_smiles, "catboost_pharmhgt", device="cpu")
for smi, val in zip(result['smiles'], result['predicted_pCMC']):
    status = f"{val:.4f}" if not np.isnan(val) else "NaN"
    print(f"  {smi:40s} -> {status}")

# ===========================================================================
# 4. Empty SMILES list (new)
# ===========================================================================
print("\n>>> 4. Empty SMILES list (predictor_single)")
empty_result = predictor_single([], "catboost_pharmhgt", device="cpu")
print(f"  Empty result: {empty_result}")
