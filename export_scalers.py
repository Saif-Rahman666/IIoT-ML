import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# CONFIG
# -----------------------------
AE_SENSORS = ["s2", "s3", "s4", "s7", "s11", "s12", "s15"]
FEATURES = ["health_index"] + AE_SENSORS

# -----------------------------
# LOAD TRAINING DATA
# -----------------------------
df = pd.read_csv("data/train_lstm.csv")

# -----------------------------
# AE SCALER
# -----------------------------
ae_scaler = MinMaxScaler()
ae_scaler.fit(df[AE_SENSORS])

joblib.dump(ae_scaler, "models/ae_scaler.pkl")

# -----------------------------
# LSTM SCALER
# -----------------------------
lstm_scaler = MinMaxScaler()
lstm_scaler.fit(df[FEATURES])

joblib.dump(lstm_scaler, "models/lstm_scaler.pkl")

print("✅ Scalers saved successfully:")
print(" - models/ae_scaler.pkl")
print(" - models/lstm_scaler.pkl")
