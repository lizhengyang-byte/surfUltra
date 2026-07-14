"""
train_xgboost_use_pharmhgt_features.py 鈥?XGBoost with PharmHGT-style Featurization
===================================================================================

Uses shared featurization from smiles_to_features_pharmhgt.py (522-dim).
Features are cached under data/features/pharmhgt/ after first computation.

Usage:
  python train_xgboost_use_pharmhgt_features.py

Data:
  ./data/surfpro_imputed.csv  (training, imputed)
  ./data/surfpro_test.csv     (test)
"""

import os, sys, math, random, warnings, zlib

import numpy as np
import pandas as pd

# Shared featurization
from smiles_to_features_pharmhgt import load_or_compute_features, FEATURE_NAMES

# XGBoost
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optuna
import optuna
from optuna.pruners import MedianPruner

warnings.filterwarnings('ignore')


# ===========================================================================
# Main - Load Data, Train XGBoost with Optuna
# ===========================================================================

def main():
    DATA_TRAIN = './data/surfpro_imputed.csv'
    DATA_TEST = './data/surfpro_test.csv'
    TARGET_COL = 'pCMC'
    SMILES_COL = 'SMILES'
    VAL_FRAC = 0.125
    HOLDOUT_FRAC = 0.10           # holdout 用于从 Top-K 候选参数中二次筛选
    SEED = 42
    N_OPTUNA_TRIALS = 200          # 增加搜索量以覆盖更多参数组合
    N_FOLDS = 5
    TOP_K_CANDIDATES = 5           # 从 Top-K 中选泛化最好的

    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("XGBoost + PharmHGT-style Featurization for LogCMC (pCMC) Prediction")
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

    # ---- Holdout split (从训练数据中划出部分，用于 Optuna 后 Top-K 筛选) ----
    X_cv, X_holdout, y_cv, y_holdout = train_test_split(
        X_full, y_full, test_size=HOLDOUT_FRAC, random_state=SEED + 1)
    print(f"  CV (Optuna): {len(X_cv)}, Holdout (Top-K filter): {len(X_holdout)}")

    # ======================================================================
    # Optuna Hyperparameter Optimization (K-Fold CV)
    # ======================================================================
    print("\n" + "=" * 60)
    print(f"Optuna Hyperparameter Tuning ({N_OPTUNA_TRIALS} trials, {N_FOLDS}-Fold CV)")
    print("=" * 60)

    FEATURE_NAME = 'pharmhgt_522'

    def objective(trial):
        params = {
            # Core parameters — 精炼范围
            "n_estimators": trial.suggest_int("n_estimators", 800, 3000),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            # Regularization
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.3, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 8.0),
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        }

        cv_scores = []
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        for fold, (train_idx_cv, val_idx_cv) in enumerate(kf.split(X_cv)):
            X_tr_cv = X_cv[train_idx_cv]
            y_tr_cv = y_cv[train_idx_cv]
            X_val_cv = X_cv[val_idx_cv]
            y_val_cv = y_cv[val_idx_cv]

            model_cv = xgb.XGBRegressor(
                random_state=SEED,
                verbosity=0,
                early_stopping_rounds=100,
                **params,
            )
            model_cv.fit(
                X_tr_cv, y_tr_cv,
                eval_set=[(X_val_cv, y_val_cv)],
                verbose=False,
            )
            y_pred_cv = model_cv.predict(X_val_cv)
            rmse_cv = np.sqrt(mean_squared_error(y_val_cv, y_pred_cv))

            # ---- 计算训练-验证差距，惩罚过拟合 ----
            y_train_pred_cv = model_cv.predict(X_tr_cv)
            rmse_train_cv = np.sqrt(mean_squared_error(y_tr_cv, y_train_pred_cv))
            gap = rmse_train_cv - rmse_cv
            # 如果 gap > 0.3 说明明显过拟合，轻微调高 score
            adjusted_rmse = rmse_cv * (1.0 + 0.05 * max(0.0, gap - 0.3))
            cv_scores.append(adjusted_rmse)

            # Report intermediate mean to pruner after each fold
            mean_so_far = np.mean(cv_scores)
            trial.report(mean_so_far, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(cv_scores)

    sampler = optuna.samplers.TPESampler(multivariate=True, seed=SEED, n_startup_trials=10)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1, n_min_trials=3)
    study = optuna.create_study(
        study_name=f'xgboost_{FEATURE_NAME}',
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

    # ---- Print all trial CV results ----
    print("\n" + "-" * 60)
    print("All Trial CV Results:")
    print(f"{'Trial':>5} {'Mean RMSE':>10} {'Std RMSE':>9}  {'CV Scores'}")
    print("-" * 60)
    for t in study.trials:
        if t.values is not None:
            print(f"{t.number:>5} {t.values[0]:>10.4f}")
    print("-" * 60)

    print(f"\n=== Best Trial ===")
    print(f"  CV RMSE: {study.best_value:.6f}")
    print(f"  Params:  {study.best_params}")

    # ---- Parameter importance ----
    try:
        importances = optuna.importance.get_param_importances(study)
        print(f"\nParameter Importance:")
        for param, imp in importances.items():
            print(f"  {param}: {imp:.4f}")
    except Exception:
        pass

    # ======================================================================
    # Top-K Holdout Validation — 从最佳参数中选泛化最好的
    # ======================================================================
    print("\n" + "=" * 60)
    print(f"Top-{TOP_K_CANDIDATES} Holdout Validation — selecting most generalizable params")
    print("=" * 60)

    # 按 CV RMSE 排序，取前 TOP_K_CANDIDATES 个 trial
    sorted_trials = sorted(
        [t for t in study.trials if t.values is not None],
        key=lambda t: t.values[0]
    )
    top_k = sorted_trials[:TOP_K_CANDIDATES]

    best_holdout_rmse = float('inf')
    best_holdout_params = None

    print(f"{'Rank':>5} {'CV RMSE':>10} {'Holdout RMSE':>14} {'Holdout R²':>12}")
    print("-" * 60)
    for rank, t in enumerate(top_k):
        params = t.params
        # 在 holdout 集上评估
        model_tmp = xgb.XGBRegressor(
            random_state=SEED, verbosity=0,
            early_stopping_rounds=150,
            **params,
        )
        model_tmp.fit(
            X_cv, y_cv,
            eval_set=[(X_cv, y_cv)],
            verbose=False,
        )
        y_pred_ho = model_tmp.predict(X_holdout)
        ho_rmse = np.sqrt(mean_squared_error(y_holdout, y_pred_ho))
        ho_r2 = r2_score(y_holdout, y_pred_ho)
        print(f"{rank+1:>5} {t.values[0]:>10.4f} {ho_rmse:>14.4f} {ho_r2:>12.4f}")
        if ho_rmse < best_holdout_rmse:
            best_holdout_rmse = ho_rmse
            best_holdout_params = params

    print(f"\n  → Selected params (holdout RMSE={best_holdout_rmse:.4f}):")
    for k, v in best_holdout_params.items():
        print(f"    {k}: {v}")

    # ======================================================================
    # Final Training with Best Params — 使用全部数据
    # ======================================================================
    print("\n" + "=" * 60)
    print(f"Training Final Model with Best Hyperparameters (X_full: {len(X_full)} samples)")
    print("=" * 60)

    best_params = best_holdout_params.copy()

    # 确保用于最终训练的 n_estimators 足够大 (>= 3000)
    # 这样模型可以充分收敛 (early_stopping_rounds 控制早停)
    best_params["n_estimators"] = max(best_params.get("n_estimators", 1000), 3000)

    final_model = xgb.XGBRegressor(
        random_state=SEED,
        verbosity=0,
        **best_params,
    )
    final_model.fit(
        X_full, y_full,
        verbose=False,
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
    print(f"  Test R虏:   {test_r2:.4f}")

    # ---- Feature Importance ----
    print(f"\n{'='*60}")
    print("Top 20 Feature Importances")
    print(f"{'='*60}")
    importances = final_model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]

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
        fig.suptitle('XGBoost + PharmHGT Features - pCMC Prediction', fontsize=14)

        # Pred vs True
        ax = axes[0]
        ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
        lims = [min(y_test.min(), y_pred.min()) - 0.5, max(y_test.max(), y_pred.max()) + 0.5]
        ax.plot(lims, lims, 'r--', alpha=0.8, linewidth=1)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('True pCMC'); ax.set_ylabel('Predicted pCMC')
        ax.set_title(f'Test R虏 = {test_r2:.4f}')
        ax.axis('square')

        # Residuals
        ax = axes[1]
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax.set_xlabel('Predicted pCMC'); ax.set_ylabel('Residuals')
        ax.set_title(f'MAE = {test_mae:.4f}')

        plt.tight_layout()
        plot_path = 'reports/xgboost_pharmhgt_pred_vs_true.png'
        os.makedirs('reports', exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {plot_path}")
    except ImportError:
        print("\n(Matplotlib not available - skipping plot)")

    # ---- Save model ----
    import joblib
    model_path = 'models/predictor/weights/xgboost_pharmhgt_model.pkl'
    # XGBoost has its own save/load, but we store the sklearn-compatible wrapper
    joblib.dump(final_model, model_path)
    print(f"Model saved to {model_path}")

    print(f"\n{'='*60}")
    print("SUMMARY - XGBoost + PharmHGT Features")
    print(f"{'='*60}")
    print(f"  Features:  {X_full.shape[1]}-dim (atom_agg + bond_agg + MACCS + BRICS + surfactant + descriptors)")
    print(f"  Train:     {len(X_full)} (CV {len(X_cv)} + Holdout {len(X_holdout)})")
    print(f"  Test:      {len(X_test)}")
    print(f"  Optuna:    {N_OPTUNA_TRIALS} trials, {N_FOLDS}-fold CV, multivariate TPE, gap penalty")
    print(f"  Top-K:     {TOP_K_CANDIDATES} candidates re-evaluated on holdout ({len(X_holdout)} samples)")
    print(f"  Best CV RMSE: {study.best_value:.4f}")
    print(f"  Holdout RMSE: {best_holdout_rmse:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test MAE:  {test_mae:.4f}")
    print(f"  Test R虏:   {test_r2:.4f}")


if __name__ == '__main__':
    main()

