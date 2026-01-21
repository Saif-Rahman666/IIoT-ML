from fastapi import FastAPI
from pydantic import BaseModel
from inference_pipeline import predict_step

app = FastAPI(title="Predictive Maintenance API")

# -----------------------------
# Input schema
# -----------------------------
class SensorInput(BaseModel):
    unit: int
    s2: float
    s3: float
    s4: float
    s7: float
    s11: float
    s12: float
    s15: float

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def root():
    return {"status": "API running"}

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(data: SensorInput):
    sensors = {
        "s2": data.s2,
        "s3": data.s3,
        "s4": data.s4,
        "s7": data.s7,
        "s11": data.s11,
        "s12": data.s12,
        "s15": data.s15,
    }

    result = predict_step(data.unit, sensors)
    return result
