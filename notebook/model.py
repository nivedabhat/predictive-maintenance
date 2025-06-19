import os
import json
import warnings

import pandas as pd
import numpy as np
import joblib


from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor


warnings.filterwarnings("ignore")

# === CONFIG ===
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))   # 

ROOT_DIR = os.path.dirname(PROJECT_DIR)                    # 

DATA_PATH = os.path.join(ROOT_DIR, "pdf_spec_parser/output/synthetic_timeseries_realistic_wide.csv")
THRESHOLD_PATH = os.path.join(ROOT_DIR, "pdf_spec_parser/output/final_clean_parameters.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "fastapi_app/models")
os.makedirs(MODEL_DIR, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(DATA_PATH)
thresh_df = pd.read_csv(THRESHOLD_PATH)

# Normalize threshold df columns
thresh_df.columns = thresh_df.columns.str.lower()
thresh_df['parameter'] = thresh_df['parameter'].str.strip().str.lower()

# Handle duplicate parameters in threshold file by averaging their bounds
if thresh_df['parameter'].duplicated().any():
    print("Warning: duplicate parameters found. Aggregating bounds by mean.")
    thresh_df = thresh_df.groupby('parameter')[['lower', 'upper']].mean().reset_index()

# Lowercase column names in main df for consistency
df.columns = [c.lower() for c in df.columns]

# === FEATURE ENGINEERING ===
# List of parameters to normalize - exclude non-parameter cols
exclude_cols = ['timestamp', 'equipment_id', 'model_id', 'anomaly_type', 'status', 'failure_mode', 'deviation', 'rul']
parameter_cols = [c for c in df.columns if c not in exclude_cols]

# Create lookup dict for parameter bounds
param_bounds = thresh_df.set_index('parameter')[['lower', 'upper']].to_dict('index')

# Normalize and create additional features
for param in parameter_cols:
    bounds = param_bounds.get(param, None)
    if bounds and not np.isnan(bounds['lower']) and not np.isnan(bounds['upper']) and bounds['upper'] > bounds['lower']:
        lower = bounds['lower']
        upper = bounds['upper']
        df[f'{param}_norm'] = (df[param] - lower) / (upper - lower)
        df[f'{param}_dev_mid'] = df[param] - ((lower + upper) / 2)
        df[f'{param}_dist_lower'] = df[param] - lower
        df[f'{param}_dist_upper'] = upper - df[param]
    else:
        # If no bounds, fill with NaN for now (will fill later)
        df[f'{param}_norm'] = np.nan
        df[f'{param}_dev_mid'] = np.nan
        df[f'{param}_dist_lower'] = np.nan
        df[f'{param}_dist_upper'] = np.nan

# Rolling stats: mean and std over last 5 and 10 timestamps grouped by equipment_id
df = df.sort_values(by=['equipment_id', 'timestamp'])

for param in parameter_cols:
    df[f'{param}_mean_5'] = df.groupby('equipment_id')[param].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df[f'{param}_std_5'] = df.groupby('equipment_id')[param].transform(lambda x: x.rolling(5, min_periods=1).std().fillna(0))
    df[f'{param}_mean_10'] = df.groupby('equipment_id')[param].transform(lambda x: x.rolling(10, min_periods=1).mean())
    df[f'{param}_std_10'] = df.groupby('equipment_id')[param].transform(lambda x: x.rolling(10, min_periods=1).std().fillna(0))

# Encode 'model_id' categorical feature
le_model = LabelEncoder()
df['model_enc'] = le_model.fit_transform(df['model_id'])
joblib.dump(le_model, os.path.join(MODEL_DIR, "le_model.pkl"))

# Prepare feature list: encoded model + all engineered features
feature_cols = ['model_enc']
for param in parameter_cols:
    feature_cols.extend([
        f'{param}_norm',
        f'{param}_dev_mid',
        f'{param}_dist_lower',
        f'{param}_dist_upper',
        f'{param}_mean_5',
        f'{param}_std_5',
        f'{param}_mean_10',
        f'{param}_std_10',
    ])

# --- Handle missing values in features by filling with 0 ---
df[feature_cols] = df[feature_cols].fillna(0)

# --- Verify data shapes and label distribution ---
print(f"Dataset shape after feature engineering: {df.shape}")
print(f"Label 'status' value counts:\n{df['status'].value_counts()}")

# --- Classification: Failure Status (binary) ---
X_cls = df[feature_cols]
y_cls = df['status'].apply(lambda x: 1 if str(x).lower() == 'failure' else 0)

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

scaler_cls = StandardScaler()
X_train_cls = scaler_cls.fit_transform(X_train_cls)
X_test_cls = scaler_cls.transform(X_test_cls)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_cls, y_train_cls)
y_pred_cls = clf.predict(X_test_cls)

print("\nClassification Report:\n", classification_report(y_test_cls, y_pred_cls))

joblib.dump(clf, os.path.join(MODEL_DIR, "failure_classifier.pkl"))
joblib.dump(scaler_cls, os.path.join(MODEL_DIR, "scaler_cls.pkl"))

# --- RUL Regression ---
y_rul = df['rul'].values
X_rul = df[feature_cols]

# Log-transform target to stabilize variance
df['rul'] = df['rul'].clip(upper=120)
y_rul_log = np.log1p(y_rul)
# Clip RUL before extracting targets



# Scale target between 0 and 1
rul_scaler = MinMaxScaler()
y_rul_scaled = rul_scaler.fit_transform(y_rul_log.reshape(-1, 1)).ravel()

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_rul, y_rul_scaled, test_size=0.2, random_state=42
)

scaler_r = StandardScaler()
X_train_r = scaler_r.fit_transform(X_train_r)
X_test_r = scaler_r.transform(X_test_r)

# Hyperparameter tuning with RandomizedSearchCV for XGBRegressor
param_grid = {
    'n_estimators': [200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0]
}

print("\nStarting XGBoost hyperparameter tuning for RUL regression...")
reg_base = XGBRegressor(objective='reg:squarederror', random_state=42, verbosity=0)

search = RandomizedSearchCV(
    reg_base,
    param_grid,
    n_iter=10,
    scoring='neg_mean_absolute_error',
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

search.fit(X_train_r, y_train_r)
reg = search.best_estimator_
print("\nBest XGBoost parameters:", search.best_params_)

joblib.dump(reg, os.path.join(MODEL_DIR, "rul_regressor.pkl"))
joblib.dump(scaler_r, os.path.join(MODEL_DIR, "scaler_reg.pkl"))
joblib.dump(rul_scaler, os.path.join(MODEL_DIR, "rul_scaler.pkl"))

# --- Evaluate RUL Regression ---
y_pred_scaled = reg.predict(X_test_r)
print(" Scaled RUL predictions (first 20):", y_pred_scaled[:20])
print(" Inverse (log) RULs:", rul_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))[:20])
print(" Final RULs:", np.expm1(rul_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()[:20]))

# Inverse transformations to get RUL in original scale
y_pred_log = rul_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_pred = np.expm1(y_pred_log)
y_pred = np.maximum(y_pred, 0)

y_true_log = rul_scaler.inverse_transform(y_test_r.reshape(-1, 1)).ravel()
y_true = np.expm1(y_true_log)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae_pct = 100 * mae / y_true.max()

print(f"\nRUL Regression MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAE%: {mae_pct:.2f}%")

# --- Save feature columns ---
with open(os.path.join(MODEL_DIR, "feature_cols.json"), "w", encoding="utf-8") as f:
    json.dump(feature_cols, f)
