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

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from models.predictor.API_predictor import predictor, predictor_single
from models.predictor.model_loader import get_available_models


# 单个分子式预测 — 传入字符串
# print("\n>>> 同分子第1次预测")
print("=" * 60)
print("catboost_pharmhgt:S(C1C=CC(=CC=1)CCCCCCCCCC(C)C)(=O)(=O)O")
result = predictor_single("S(C1C=CC(=CC=1)CCCCCCCCCC(C)C)(=O)(=O)O", "catboost_pharmhgt", device="cpu")
print(f"Result: {result['predicted_pCMC'][0]:.4f}")

# print("\n>>> 同分子第2次预测")
# result = predictor_single("S(C1C=CC(=CC=1)CCCCCCCCCC(C)C)(=O)(=O)O", "catboost_pharmhgt", device="cpu")
# print(f"Result: {result['predicted_pCMC'][0]:.4f}")
    
# print("\n>>> 同分子第3次预测")
# result = predictor_single("S(C1C=CC(=CC=1)CCCCCCCCCC(C)C)(=O)(=O)O", "catboost_pharmhgt", device="cpu")
# print(f"Result: {result['predicted_pCMC'][0]:.4f}")

# print("\n>>> 同分子第4次预测")
# result = predictor_single("S(C1C=CC(=CC=1)CCCCCCCCCC(C)C)(=O)(=O)O", "catboost_pharmhgt", device="cpu")
# print(f"Result: {result['predicted_pCMC'][0]:.4f}")



# # 单个分子式预测多次预测 — 传入字符串
# def test_predictor_times(times, smiles, model_name, device="cpu"):
#     results = []
#     for i in range(times):
#         result = predictor_single(smiles, model_name, device=device)
#         results.append(result['predicted_pCMC'][0])
#     return results
# print("S(C1C=CC(=CC=1)CCCCCCCCCC(C)C)(=O)(=O)O")
# print("=" * 60)
# print(test_predictor_times(5, "S(C1C=CC(=CC=1)CCCCCCCCCC(C)C)(=O)(=O)O", "catboost_pharmhgt", device="cpu"))


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from models.predictor.API_predictor import predictor, predictor_single
from models.predictor.model_loader import get_available_models

print("=" * 60)
print("Available models:", get_available_models())
print("=" * 60)