"""
retrain_catboost_all.py — Re-train CatBoost with all RDKit descriptors (model 5)
and save to weights/catboost_all_features_model.pkl.

Adapted from train_catboost_use_all_features.py with:
  - Increased Optuna trials (5 → 50)
  - Uses surfpro_imputed.csv (pre-imputed, more data)
  - Saves model to weights/ directory

Usage:
    python models/predictor/retrain_catboost_all.py
"""

import os, sys, random, warnings

# Add project root to path for standalone execution
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
from rdkit import Chem
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import joblib

from models.predictor.featurizer import smiles_to_features_all, get_all_descriptor_names

warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def main():
    N_TRIALS = 50
    N_FOLDS = 5
    VAL_FRAC = 0.125

    # ========== Load data ==========
    print("=" * 60)
    print("Loading data ...")
    df_train = pd.read_csv('data/surfpro_imputed.csv').dropna(subset=['pCMC'])
    df_test = pd.read_csv('data/surfpro_test.csv').dropna(subset=['pCMC'])

    y_train_full = df_train['pCMC'].values
    y_test = df_test['pCMC'].values

    # ========== Featurize ==========
    print("Computing all RDKit descriptors ...")

    def featurize(df):
        features, indices = [], []
        for i, smi in enumerate(df['SMILES']):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            vec = smiles_to_features_all(smi)
            features.append(vec)
            indices.append(i)
        return np.array(features, dtype=np.float64), indices

    X_full, train_idx = featurize(df_train)
    y_full = y_train_full[train_idx]
    X_test, test_idx = featurize(df_test)
    y_test = y_test[test_idx]

    print(f"  Train features: {X_full.shape}")
    print(f"  Test features:  {X_test.shape}")

    # ========== Train/Val split ==========
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=VAL_FRAC, random_state=SEED)

    # ========== Optuna ==========
    print(f"\n{'=' * 60}")
    print(f"Optuna hyperparameter tuning ({N_TRIALS} trials, {N_FOLDS}-Fold CV) ...")

    def objective(trial):
        params = {
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'iterations': trial.suggest_int('iterations', 500, 3000),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 50.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        }
        kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        cv_rmse = []
        for tr_idx, va_idx in kfold.split(X_full):
            X_cv_tr, X_cv_va = X_full[tr_idx], X_full[va_idx]
            y_cv_tr, y_cv_va = y_full[tr_idx], y_full[va_idx]
            m = CatBoostRegressor(random_seed=SEED, verbose=0, **params)
            m.fit(X_cv_tr, y_cv_tr, eval_set=(X_cv_va, y_cv_va),
                  early_stopping_rounds=100, verbose=False)
            pred = m.predict(X_cv_va)
            cv_rmse.append(np.sqrt(mean_squared_error(y_cv_va, pred)))
        return np.mean(cv_rmse)

    study = optuna.create_study(
        direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS)

    best_params = study.best_params.copy()
    print(f"\nBest CV RMSE: {study.best_value:.4f}")
    for k, v in best_params.items():
        print(f"  {k} = {v}")

    # ========== Final training ==========
    print(f"\n{'=' * 60}")
    print("Training final model on full train+val data ...")

    final_iter = max(best_params.get('iterations', 1000), 3000)
    best_params['iterations'] = final_iter

    model = CatBoostRegressor(random_seed=SEED, verbose=0, **best_params)
    model.fit(
        X_full, y_full,
        eval_set=(X_val, y_val),
        early_stopping_rounds=150,
        verbose=False,
    )

    # ========== Evaluate ==========
    def evaluate(y_true, y_pred, name):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"  [{name}] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        return rmse, mae, r2

    train_pred = model.predict(X_full)
    evaluate(y_full, train_pred, 'Train')

    test_pred = model.predict(X_test)
    evaluate(y_test, test_pred, 'Test')

    # ========== Save ==========
    save_path = os.path.join(WEIGHTS_DIR, 'catboost_all_features_model.pkl')
    joblib.dump(model, save_path)
    print(f"\n✅ Model saved to {save_path}")

    # ========== Feature importance ==========
    print(f"\n{'=' * 60}")
    print("Feature importance (Top 20):")
    feat_imp = model.feature_importances_
    top20 = np.argsort(feat_imp)[-20:][::-1]
    desc_names = get_all_descriptor_names()
    print(f"{'Rank':>4} {'Feature':<30} {'Importance':>10}")
    print("-" * 46)
    for rank, idx in enumerate(top20, 1):
        name = desc_names[idx] if idx < len(desc_names) else f'idx_{idx}'
        print(f"{rank:>4} {name:<30} {feat_imp[idx]:>10.2f}")


if __name__ == '__main__':
    main()
