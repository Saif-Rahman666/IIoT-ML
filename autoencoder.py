import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load data - Ensure this points to your RAW train.csv
df = pd.read_csv("data/train.csv")
sensor_cols = ["s2", "s3", "s4", "s7", "s11", "s12", "s15"]

# 1. Fit scaler on RAW values
scaler = MinMaxScaler()
scaler.fit(df[sensor_cols])
X_scaled = scaler.transform(df[sensor_cols])

# 2. Train only on Healthy data (first 50 cycles)
X_healthy = X_scaled[df['time'] < 50]

autoencoder = Sequential([
    Dense(32, activation="relu", input_shape=(len(sensor_cols),)),
    Dense(16, activation="relu"),
    Dense(32, activation="relu"),
    Dense(len(sensor_cols), activation="sigmoid")
])

autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(X_healthy, X_healthy, epochs=50, batch_size=32, verbose=0)

# 3. Save
autoencoder.save("models/autoencoder.h5")
joblib.dump(scaler, "models/ae_scaler.pkl")
print("✅ Retraining complete. Scaler and Model are now aligned with raw data.")