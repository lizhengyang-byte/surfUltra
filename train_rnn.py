import pandas as pd
import numpy as np
from smiles_to_features import smiles_to_features

data_train_file_path = 'data/surfpro_train.csv'
data_test_file_path = 'data/surfpro_test.csv'

# 数据清洗
df_train = pd.read_csv(data_train_file_path)
df_train = df_train.dropna(subset=['pCMC'])

# 将SMILES逐条转换为特征
x_train = np.array([smiles_to_features(smi) for smi in df_train["SMILES"]])
y_train = df_train['pCMC'].values

# ==================== RNN 训练 ====================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# -------------------- 1. 使用已加载数据 --------------------
X = x_train
y = y_train
print(f"样本数: {X.shape[0]}, 特征维度: {X.shape[1]}")

# -------------------- 2. 划分数据集 (70% / 15% / 15%) --------------------
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")

# -------------------- 3. 特征标准化 --------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# -------------------- 4. Optuna 超参数搜索 --------------------
print("\n" + "=" * 60)
print("超参数搜索: Optuna (TPE) ...")

import optuna

N_TRIALS = 30


def build_and_train(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "dropout":       trial.suggest_float("dropout", 0.1, 0.4),
        "l2_reg":        trial.suggest_float("l2_reg", 1e-5, 1e-3, log=True),
        "units_1":       trial.suggest_categorical("units_1", [64, 128, 256]),
        "units_2":       trial.suggest_categorical("units_2", [32, 64, 128]),
        "batch_size":    trial.suggest_categorical("batch_size", [16, 32, 64]),
    }
    model = Sequential([
        Dense(params["units_1"], activation='relu',
              kernel_regularizer=l2(params["l2_reg"]),
              input_shape=(X_train_scaled.shape[1],)),
        BatchNormalization(),
        Dropout(params["dropout"]),
        Dense(params["units_2"], activation='relu',
              kernel_regularizer=l2(params["l2_reg"])),
        BatchNormalization(),
        Dropout(params["dropout"]),
        Dense(32, activation='relu', kernel_regularizer=l2(params["l2_reg"])),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=params["learning_rate"]),
                  loss='mse', metrics=['mae'])

    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=300, batch_size=params["batch_size"],
        callbacks=[es], verbose=0,
    )
    best_val_r2 = max(r2_score(y_val, model.predict(X_val_scaled, verbose=0).flatten()),
                      r2_score(y_val, model.predict(X_val_scaled, verbose=0).flatten()))
    return best_val_r2


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(build_and_train, n_trials=N_TRIALS, show_progress_bar=True)

best_params = study.best_params
print(f"\n最佳参数: {best_params}")
print(f"最佳验证 R²: {study.best_value:.4f}")

# -------------------- 5. 用最佳参数训练最终模型 --------------------
print("\n" + "=" * 60)
print("用最佳参数训练最终模型 ...")

model = Sequential([
    Dense(best_params["units_1"], activation='relu',
          kernel_regularizer=l2(best_params["l2_reg"]),
          input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(best_params["dropout"]),
    Dense(best_params["units_2"], activation='relu',
          kernel_regularizer=l2(best_params["l2_reg"])),
    BatchNormalization(),
    Dropout(best_params["dropout"]),
    Dense(32, activation='relu', kernel_regularizer=l2(best_params["l2_reg"])),
    Dense(1),
])
model.compile(optimizer=Adam(learning_rate=best_params["learning_rate"]),
              loss='mse', metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=500, batch_size=best_params["batch_size"],
    callbacks=[early_stop, lr_scheduler], verbose=1,
)

# -------------------- 6. 评估 --------------------
y_pred_train = model.predict(X_train_scaled).flatten()
y_pred_val   = model.predict(X_val_scaled).flatten()
y_pred_test  = model.predict(X_test_scaled).flatten()

print("\n===== 训练集 =====")
print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
print(f"MAE:  {mean_absolute_error(y_train, y_pred_train):.4f}")
print(f"R²:   {r2_score(y_train, y_pred_train):.4f}")

print("\n===== 验证集 =====")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val, y_pred_val)):.4f}")
print(f"MAE:  {mean_absolute_error(y_val, y_pred_val):.4f}")
print(f"R²:   {r2_score(y_val, y_pred_val):.4f}")

print("\n===== 测试集 =====")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, y_pred_test):.4f}")
print(f"R²:   {r2_score(y_test, y_pred_test):.4f}")