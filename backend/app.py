from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

app = FastAPI(title="Oil & Gas DSS API", version="1.0")
def explain_decision(data):
    reasons = []

    if data.vibration > 6:
        reasons.append("High vibration indicates mechanical instability")
    if data.runtime_hours > 2000:
        reasons.append("Extended runtime increases wear and fatigue")
    if data.maintenance_gap > 300:
        reasons.append("Long maintenance gap elevates failure risk")
    if data.temperature > 110:
        reasons.append("Elevated temperature stresses equipment")

    return reasons

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # allow dashboard to call API
    allow_credentials=True,
    allow_methods=["*"],      # allow OPTIONS, POST, GET, etc.
    allow_headers=["*"],
)


# Paths (relative to backend/)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "dnn_failure_model.keras"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

# Load model + scaler
model = tf.keras.models.load_model(str(MODEL_PATH))
scaler = joblib.load(str(SCALER_PATH))

class DSSInput(BaseModel):
    pressure: float
    temperature: float
    vibration: float
    flow_rate: float
    runtime_hours: float
    maintenance_gap: float

def decision_rule(prob: float):
    if prob >= 0.70:
        return "HIGH", "Immediate maintenance required"
    elif prob >= 0.40:
        return "MEDIUM", "Monitor closely and schedule inspection"
    else:
        return "LOW", "Normal operation"

@app.get("/")
def home():
    return {"message": "DSS API is running. Go to /docs to test the API."}

@app.post("/predict")
def predict(data: DSSInput):
    X = np.array([[data.pressure,
                   data.temperature,
                   data.vibration,
                   data.flow_rate,
                   data.runtime_hours,
                   data.maintenance_gap]])

    X_scaled = scaler.transform(X)
    prob = float(model.predict(X_scaled, verbose=0)[0][0])

    risk, recommendation = decision_rule(prob)
    reasons = explain_decision(data)

    return {
        "failure_probability": round(prob, 4),
        "risk_level": risk,
        "recommendation": recommendation,
        "explanation": reasons
    }
