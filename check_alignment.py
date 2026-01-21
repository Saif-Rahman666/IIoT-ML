import joblib
import pandas as pd
import numpy as np

# Load the scaler we just saved
scaler = joblib.load("models/ae_scaler.pkl")
sensors = ["s2", "s3", "s4", "s7", "s11", "s12", "s15"]

# Test case: A typical raw sensor row (NASA FD001)
raw_data = {
    "s2": 642.0, "s3": 1585.0, "s4": 1400.0, 
    "s7": 553.0, "s11": 47.0, "s12": 521.0, "s15": 8.4
}

# Convert to DataFrame
df_test = pd.DataFrame([raw_data], columns=sensors)

# Transform
scaled_data = scaler.transform(df_test)

print("--- SCALER SANITY CHECK ---")
print(f"Raw Input: {raw_data['s3']} (sensor s3)")
print(f"Scaled Output: {scaled_data[0][1]:.4f}") # Should be between 0 and 1

if 0 <= scaled_data[0][1] <= 1:
    print("✅ SUCCESS: The scaler is correctly normalizing raw values!")
else:
    print("❌ ERROR: The output is still too large. Check your training data source.")