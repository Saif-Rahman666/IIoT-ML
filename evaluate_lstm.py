import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Constants (MUST match training)
# -----------------------------
SEQ_LEN = 30
RUL_CAP = 80

# -----------------------------
# Load data + model
# -----------------------------
df = pd.read_csv("data/train_lstm.csv")
model = load_model("lstm_rul_model.h5")

features = [
    "health_index",
    "s2", "s3", "s4", "s7", "s11", "s12", "s15"
]

# -----------------------------
# Scale features (same as training)
# -----------------------------
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# -----------------------------
# Make sequences (SAME logic)
# -----------------------------
def make_sequences(df):
    X, y = [], []

    for unit in df["unit"].unique():
        sub = df[df["unit"] == unit].sort_values("time")

        for i in range(len(sub) - SEQ_LEN):
            X.append(sub[features].iloc[i:i+SEQ_LEN].values)
            y.append(sub["RUL"].iloc[i+SEQ_LEN] / RUL_CAP)

    return np.array(X), np.array(y)

X, y_true = make_sequences(df)

# -----------------------------
# Predict
# -----------------------------
y_pred = model.predict(X).flatten()

# Undo normalization
y_pred = y_pred * RUL_CAP
y_true = y_true * RUL_CAP

# -----------------------------
# Metrics
# -----------------------------
mae = np.mean(np.abs(y_true - y_pred))
print("MAE:", mae)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, s=3, alpha=0.3)
plt.plot([0, RUL_CAP], [0, RUL_CAP], "r--")
plt.xlabel("True RUL")
plt.ylabel("Predicted RUL")
plt.title("LSTM RUL Prediction")
plt.grid(True)
plt.show()
