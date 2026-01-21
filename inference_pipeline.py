import numpy as np
import pandas as pd
from collections import deque
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
SEQ_LEN = 30
RUL_CAP = 80

AE_SENSORS = ["s2", "s3", "s4", "s7", "s11", "s12", "s15"]
FEATURES = ["health_index"] + AE_SENSORS

# -----------------------------
# LOAD MODELS
# -----------------------------
autoencoder = load_model("models/autoencoder.h5")
lstm_model = load_model("models/lstm_rul_model.h5")

# -----------------------------
# LOAD SCALERS
# -----------------------------
ae_scaler = joblib.load("models/ae_scaler.pkl")
lstm_scaler = joblib.load("models/lstm_scaler.pkl")

# -----------------------------
# ENGINE STATE
# -----------------------------
engine_buffers = {}
last_sensor_state = {}

def get_risk_level(rul):
    if rul < 20:
        return "CRITICAL"
    elif rul < 40:
        return "HIGH"
    elif rul < 80:
        return "MEDIUM"
    return "LOW"

# -----------------------------
# INFERENCE STEP
# -----------------------------
def predict_step(unit_id: int, sensor_row: dict):
    if unit_id not in engine_buffers:
        engine_buffers[unit_id] = deque(maxlen=SEQ_LEN)
        last_sensor_state[unit_id] = sensor_row.copy()

    # --- 1. AUTOENCODER (Health Index) ---
    # Convert incoming raw dict to DataFrame for the scaler
    ae_input = pd.DataFrame([[sensor_row[s] for s in AE_SENSORS]], columns=AE_SENSORS)
    ae_scaled = ae_scaler.transform(ae_input) 

    recon = autoencoder.predict(ae_scaled, verbose=0)
    mse = np.mean((ae_scaled - recon) ** 2)

    # Increase threshold to 0.5 to allow for more noise before failing
    threshold = 0.5 
    health_index = 1.0 - (mse / threshold)
    # Increase minimum clip to 0.05 so it doesn't look like a dead line
    health_index = float(np.clip(health_index, 0.05, 1.0)) 

    # DEBUG: Print this so you can see if MSE is actually smaller now
    print(f"MSE: {mse:.4f} | Health: {health_index:.2f}")

    # --- 2. LSTM (RUL Prediction) ---
    row = {"health_index": health_index}
    for s in AE_SENSORS:
        row[s] = sensor_row[s]

    row_df = pd.DataFrame([row], columns=FEATURES)
    row_scaled = lstm_scaler.transform(row_df)

    engine_buffers[unit_id].append(row_scaled[0])

    if len(engine_buffers[unit_id]) < SEQ_LEN:
        return {
            "unit": unit_id,
            "health_index": round(health_index, 3),
            "predicted_rul": None,
            "risk": "WARMING_UP",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    X = np.array(engine_buffers[unit_id]).reshape(1, SEQ_LEN, len(FEATURES))
    rul_norm = lstm_model.predict(X, verbose=0)[0][0]
    predicted_rul = int(np.clip(rul_norm * RUL_CAP, 1, RUL_CAP))

    return {
        "unit": unit_id,
        "health_index": round(health_index, 3),
        "predicted_rul": predicted_rul,
        "risk": get_risk_level(predicted_rul),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

# -----------------------------
# LOCAL TEST
# -----------------------------
if __name__ == "__main__":
    df = pd.read_csv("data/train_lstm.csv")
    sample = df[df["unit"] == 1].iloc[0]
    sensors = {s: sample[s] for s in AE_SENSORS}

    for _ in range(40):
        print(predict_step(1, sensors))
