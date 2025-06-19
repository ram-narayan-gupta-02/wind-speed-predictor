# Best Accuracy Wind Speed Model (XGBoost)
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

# --- Load and Preprocess Data ---
print("📥 Loading data...")
df = pd.read_csv(r"D:\Python\ADRDE_Project\wind_data_combined_2016_2025.csv", usecols=["time", "uwnd", "vwnd"])

# --- Feature Engineering ---
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time")
df["wind_speed"] = np.sqrt(df["uwnd"]**2 + df["vwnd"]**2)
df["wind_speed_lag_1"] = df["wind_speed"].shift(1)
df["wind_speed_lag_2"] = df["wind_speed"].shift(2)
df["wind_speed_lag_3"] = df["wind_speed"].shift(3)
df["wind_speed_avg3"] = df[["wind_speed_lag_1", "wind_speed_lag_2", "wind_speed_lag_3"]].mean(axis=1)
df["day_of_year"] = df["time"].dt.dayofyear
df["month"] = df["time"].dt.month

# Drop NaNs
features = ["uwnd", "vwnd", "wind_speed_lag_1", "wind_speed_lag_2", "wind_speed_lag_3", "wind_speed_avg3", "day_of_year", "month"]
df = df.dropna(subset=features + ["wind_speed"])

# --- Feature/Target Split ---
X = df[features]
y = df["wind_speed"]

# --- Train-Test Split (no shuffle) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# --- Model Definition ---
model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# --- Model Training ---
print("🚀 Training XGBoost model...")
model.fit(X_train, y_train)

# --- Evaluation ---
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("✅ Model Trained Successfully.")
print(f"📊 MSE: {mse:.4f} | R²: {r2:.4f} | MAE: {mae:.4f}")

# --- Save Model ---
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/wind_speed_model.pkl")

# --- Save Metrics ---
with open("model/metrics.txt", "w") as f:
    f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
    f.write(f"R² Score: {r2:.4f}\n")
    f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")

print("💾 Model and metrics saved.")
