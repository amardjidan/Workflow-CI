# -*- coding: utf-8 -*-
# =========================
# IMPORT LIBRARY
# =========================
import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================
# LOAD DATASET
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "dataset_sample.csv")

df = pd.read_csv(dataset_path)
print("Dataset berhasil dimuat")
print("Shape awal:", df.shape)

# =========================
# HANDLE MISSING VALUE
# =========================
df.dropna(subset=['sales'], inplace=True)

# =========================
# SAMPLING DATA
# =========================
if len(df) > 100000:
    df = df.sample(n=100000, random_state=42)

print("Shape setelah sampling:", df.shape)
print(df.head())

# =========================
# FEATURE & TARGET
# =========================
X = df.drop("sales", axis=1)
y = df["sales"]

# =========================
# ONE HOT ENCODING
# =========================
categorical_cols = []
if 'store_id' in X.columns:
    categorical_cols.append('store_id')
if 'item_id' in X.columns:
    categorical_cols.append('item_id')
if len(categorical_cols) > 0:
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print("Shape feature:", X.shape)

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train size:", X_train.shape)
print("Test size :", X_test.shape)

input_example = X_train.iloc[:5]

# =========================
# TRAINING + MLFLOW
# =========================
# JANGAN set_experiment() di sini — mlflow run sudah handle run ID nya
# Cukup aktifkan run yang sudah ada dari mlflow run CLI

with mlflow.start_run() as run:

    # ✅ WAJIB: autolog sebelum model.fit()
    mlflow.autolog()

    n_estimators = 10
    max_depth    = 10

    mlflow.log_param("model_type",   "RandomForestRegressor")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth",    max_depth)

    # =========================
    # MODEL & TRAINING
    # =========================
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # =========================
    # PREDICTION & EVALUATION
    # =========================
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    mlflow.log_metric("MAE",      mae)
    mlflow.log_metric("MSE",      mse)
    mlflow.log_metric("R2_Score", r2)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )

    run_id = run.info.run_id

    # Simpan run_id ke 2 lokasi agar CI bisa baca
    for save_path in [
        os.path.join(BASE_DIR, "run_id.txt"),
        os.path.join(BASE_DIR, "..", "run_id.txt"),
    ]:
        with open(os.path.abspath(save_path), "w") as f:
            f.write(run_id)
        print(f"run_id disimpan ke: {os.path.abspath(save_path)}")

    print("\n" + "="*45)
    print(f"  RUN ID    : {run_id}")
    print(f"  MAE       : {mae:.4f}")
    print(f"  MSE       : {mse:.4f}")
    print(f"  R2 Score  : {r2:.4f}")
    print("="*45)

print("\nJalankan 'mlflow ui' untuk melihat dashboard.")