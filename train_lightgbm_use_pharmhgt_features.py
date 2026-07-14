"""
train_lightgbm_use_pharmhgt_features.py — LightGBM with PharmHGT-style Featurization
=================================================================================

Uses shared featurization from smiles_to_features_pharmhgt.py (522-dim).
Features are cached under data/features/pharmhgt/ after first computation.

Usage:
  python train_lightgbm_use_pharmhgt_features.py

Data:
  ./data/surfpro_imputed.csv  (training, imputed)
  ./data/surfpro_test.csv     (test)
"""

import os, sys, math, random, warnings

import numpy as np
import pandas as pd

# Shared featurization
from smiles_to_features_pharmhgt import load_or_compute_features, FEATURE_NAMES

# LightGBM
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optuna
import optuna
from optuna.pruners import MedianPruner

warnings.filterwarnings('ignore')


# ===========================================================================
# 5. Main — Load Data, Featurize, Train LightGBM with Optuna
# ===========================================================================

def main():
    DATA_TRAIN = './data/surfpro_imputed.csv'
    DATA_TEST = './data/surfpro_test.csv'
    TARGET_COL = 'pCMC'
    SMILES_COL = 'SMILES'
    VAL_FRAC = 0.125
    SEED = 42
    N_OPTUNA_TRIALS = 50
    N_FOLDS = 5

    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("LightGBM + PharmHGT-style Featurization for LogCMC (pCMC) Prediction")
    print("=" * 60)

    # ---- Load / featurize (cached) ----
    X_full, y_full, X_test, y_test = load_or_compute_features(
        train_csv=DATA_TRAIN, test_csv=DATA_TEST,
        target_col=TARGET_COL, smiles_col=SMILES_COL,
    )
    print(f"  Train features: {X_full.shape}")
    print(f"  Test features:  {X_test.shape}")

    # ---- Train/Validation split ----
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=VAL_FRAC, random_state=SEED)
    print(f"\nSplit: Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}")

    # ======================================================================
    # Optuna Hyperparameter Optimization (K-Fold CV)
    # ======================================================================
    print("\n" + "=" * 60)
    print(f"Optuna Hyperparameter Tuning ({N_OPTUNA_TRIALS} trials, {N_FOLDS}-Fold CV)")
    print("=" * 60)

    FEATURE_NAME = 'pharmhgt_522'

    def objective(trial):
        params = {
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'num_leaves': trial.suggest_int('num_leaves', 15, 255),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 1e-1, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'cat_smooth': trial.suggest_float('cat_smooth', 0.0, 50.0),
            'cat_l2': trial.suggest_float('cat_l2', 0.0, 50.0),
            'verbose': -1,
        }

        # DART-specific
        if params['boosting_type'] == 'dart':
            params['drop_rate'] = trial.suggest_float('drop_rate', 0.01, 0.3)
            params['max_drop'] = trial.suggest_int('max_drop', 1, 50)
            params['skip_drop'] = trial.suggest_float('skip_drop', 0.01, 0.3)

        cv_scores = []
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        for fold, (train_idx_cv, val_idx_cv) in enumerate(kf.split(X_full)):
            X_tr_cv = X_full[train_idx_cv]
            y_tr_cv = y_full[train_idx_cv]
            X_val_cv = X_full[val_idx_cv]
            y_val_cv = y_full[val_idx_cv]

            model_cv = lgb.LGBMRegressor(**params, random_state=SEED, n_jobs=-1)
            model_cv.fit(
                X_tr_cv, y_tr_cv,
                eval_set=[(X_val_cv, y_val_cv)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
            )
            y_pred_cv = model_cv.predict(X_val_cv)
            rmse_cv = np.sqrt(mean_squared_error(y_val_cv, y_pred_cv))
            cv_scores.append(rmse_cv)

            mean_so_far = np.mean(cv_scores)
            trial.report(mean_so_far, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(cv_scores)

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1, n_min_trials=3)
    study = optuna.create_study(
        study_name=f'lightgbm_{FEATURE_NAME}',
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

    print(f"\n=== Best Trial ===")
    print(f"  CV RMSE: {study.best_value:.6f}")
    print(f"  Params:  {study.best_params}")

    # ======================================================================
    # Final Training with Best Params
    # ======================================================================
    print("\n" + "=" * 60)
    print("Training Final Model with Best Hyperparameters")
    print("=" * 60)

    best_params = study.best_params
    best_params['verbose'] = -1
    # Remove DART-specific params if final model is GBDT
    if best_params.get('boosting_type') != 'dart':
        for dart_key in ['drop_rate', 'max_drop', 'skip_drop']:
            best_params.pop(dart_key, None)

    final_model = lgb.LGBMRegressor(**best_params, random_state=SEED, n_jobs=-1)
    final_model.fit(
        X_full, y_full,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )

    # ======================================================================
    # Evaluation
    # ======================================================================
    print(f"\n{'='*60}")
    print("Test Evaluation")
    print(f"{'='*60}")

    y_pred = final_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    print(f"  Test MSE:  {test_mse:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test MAE:  {test_mae:.4f}")
    print(f"  Test R²:   {test_r2:.4f}")

    # ---- Feature Importance ----
    print(f"\n{'='*60}")
    print("Top 20 Feature Importances")
    print(f"{'='*60}")
    importances = final_model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]
    feature_names = [
        'atom_mean', 'atom_std', 'atom_min', 'atom_max',
        'bond_mean', 'bond_std', 'bond_min', 'bond_max',
    ] + [f'maccs_{i}' for i in range(194)] + [f'brics_{i}' for i in range(34)] + [
        'surf_anionic', 'surf_cationic', 'surf_nonionic', 'surf_zwitterionic',
        'head_ratio', 'tail_ratio',
        'MolWt', 'LogP', 'TPSA', 'RotBonds', 'HBA', 'HBD',
        'NumRings', 'AroRings', 'AliRings', 'FracSP3', 'HeavyAtoms', 'NAtoms',
    ]
    # The concatenated feature vector has a specific layout — generate names
    # atom agg (220) + bond agg (56) + maccs (194) + brics (34) + surf (4) + head/tail (2) + desc (12)
    names = FEATURE_NAMES  # from smiles_to_features_pharmhgt
    for rank, idx in enumerate(top_idx):
        print(f"  {rank+1:2d}. {names[idx]:25s}  {importances[idx]:.1f}")

    # ---- Save predictions plot ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('LightGBM + PharmHGT Features — pCMC Prediction', fontsize=14)

        # Pred vs True
        ax = axes[0]
        ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
        lims = [min(y_test.min(), y_pred.min()) - 0.5, max(y_test.max(), y_pred.max()) + 0.5]
        ax.plot(lims, lims, 'r--', alpha=0.8, linewidth=1)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('True pCMC'); ax.set_ylabel('Predicted pCMC')
        ax.set_title(f'Test R² = {test_r2:.4f}')
        ax.axis('square')

        # Residuals
        ax = axes[1]
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax.set_xlabel('Predicted pCMC'); ax.set_ylabel('Residuals')
        ax.set_title(f'MAE = {test_mae:.4f}')

        plt.tight_layout()
        plot_path = 'reports/lightgbm_pharmhgt_pred_vs_true.png'
        os.makedirs('reports', exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {plot_path}")
    except ImportError:
        print("\n(Matplotlib not available — skipping plot)")

    # ---- Save model ----
    import joblib
    model_path = 'models/predictor/weights/lightgbm_pharmhgt_model.pkl'
    joblib.dump(final_model, model_path)
    print(f"Model saved to {model_path}")

    print(f"\n{'='*60}")
    print("SUMMARY — LightGBM + PharmHGT Features")
    print(f"{'='*60}")
    print(f"  Features:  {X_full.shape[1]}-dim (atom_agg + bond_agg + MACCS + BRICS + surfactant + descriptors)")
    print(f"  Train:     {len(X_full)} (split {len(X_train)} train + {len(X_val)} val)")
    print(f"  Test:      {len(X_test)}")
    print(f"  Optuna:    {N_OPTUNA_TRIALS} trials, {N_FOLDS}-fold CV")
    print(f"  Best CV RMSE: {study.best_value:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test MAE:  {test_mae:.4f}")
    print(f"  Test R²:   {test_r2:.4f}")


if __name__ == '__main__':
    main()
