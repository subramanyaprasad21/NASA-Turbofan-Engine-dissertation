
# train_xgboost.py
import pandas as pd, joblib, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

TRAIN_CSV = "PM_train.csv"
OUT = "model_xgboost.pkl"

df = pd.read_csv(TRAIN_CSV)
rul = df.groupby("id")["cycle"].max().reset_index().rename(columns={"cycle":"max_cycle"})
df = df.merge(rul, on="id", how="left")
df["RUL"] = df["max_cycle"] - df["cycle"]

X = df.drop(columns=["id","cycle","max_cycle","RUL"])
y = df["RUL"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=1
)

print("Training XGBoost...")
model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], early_stopping_rounds=30, verbose=True)

y_pred = model.predict(X_val_s)
mae = mean_absolute_error(y_val, y_pred)
rmse = mean_squared_error(y_val, y_pred, squared=False)
r2 = r2_score(y_val, y_pred)

print(f"Validation: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.4f}")
joblib.dump({{"model": model, "scaler": scaler}}, OUT)
print(f"Saved model to {OUT}")
