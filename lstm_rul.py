import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

SEQ_LEN = 30

df = pd.read_csv("data/train_lstm.csv")

RUL_CAP = 80
df["RUL"] = df["RUL"].clip(upper=RUL_CAP)

features = [
    "health_index",
    "s2", "s3", "s4", "s7", "s11", "s12", "s15"
]


scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

def make_sequences(df, seq_len=30):
    X, y = [], []

    for unit in df["unit"].unique():
        sub = df[df["unit"] == unit]

        for i in range(len(sub) - seq_len):
            X.append(sub[features].iloc[i:i+seq_len].values)
            y.append(sub["RUL"].iloc[i+seq_len] / RUL_CAP)

    return np.array(X), np.array(y)

X, y = make_sequences(df, SEQ_LEN)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, X.shape[2])),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

model.fit(
    X, y,
    epochs=40,
    batch_size=64,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=5)],
    shuffle=False
)

model.save("lstm_rul_model.h5")
print("LSTM RUL model saved")
