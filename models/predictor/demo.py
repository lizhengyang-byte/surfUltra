import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.predictor.API_predictor import predictor
from models.predictor.model_loader import get_available_models

# 查看可用模型
print(get_available_models())

input_path = Path(__file__).parent / "input.csv.csv"

# 使用默认参数（全部模型，CPU）
predictor(str(input_path), "output_all.csv", "all")

# 指定单个模型和 GPU
predictor(str(input_path), "output_catboost_pharmhgt.csv", "catboost_pharmhgt", device="cpu")

# 自定义 SMILES 列名
predictor(str(input_path), "out_all_default_SMILES_column.csv", "all", smiles_col="SMILES")