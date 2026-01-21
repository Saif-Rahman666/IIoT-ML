# merge_health.py
import pandas as pd

# Load NASA + RUL
base = pd.read_csv("data/train.csv")

# Load autoencoder output
health = pd.read_csv("data/anomaly_output.csv")

# Safety check
assert {"unit", "time", "health_index"}.issubset(health.columns), \
    "health_index missing from anomaly_output.csv"

# Merge
df = base.merge(
    health[["unit", "time", "health_index"]],
    on=["unit", "time"],
    how="left"
)

# Fill any missing values (early cycles sometimes)
df["health_index"] = df["health_index"].fillna(1.0)

# -----------------------------
# RUL capping
# -----------------------------
RUL_CAP = 80  # standard for FD001

df["RUL"] = df["RUL"].clip(upper=RUL_CAP)

# -----------------------------
# Optional: smooth health index
# -----------------------------
df["health_index"] = (
    df.groupby("unit")["health_index"]
      .rolling(window=5, min_periods=1)
      .mean()
      .reset_index(level=0, drop=True)
)


df.to_csv("data/train_lstm.csv", index=False)
print("✅ data/train_lstm.csv created")
print(df.head())
