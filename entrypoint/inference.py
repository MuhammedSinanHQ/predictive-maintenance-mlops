# entrypoint/inference.py
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional

# -----------------------
# Config / model loading
# -----------------------
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "rf_failure_joblib.pkl"
IMPUTER_PATH = ROOT / "models" / "imputer_joblib.pkl"
SCALER_PATH = ROOT / "models" / "scaler_joblib.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run the training script first.")

model = joblib.load(MODEL_PATH)

# Imputer & scaler maybe optional; load if present
imputer = joblib.load(IMPUTER_PATH) if IMPUTER_PATH.exists() else None
scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None

# load feature names used at training (saved by training as feature_importances.csv maybe)
FI_PATH = ROOT / "data" / "04-predictions" / "feature_importances.csv"
if FI_PATH.exists():
    FEATURE_ORDER = list(pd.read_csv(FI_PATH)["feature"].values)
else:
    # fallback: accept arbitrary ordered features (user must provide full set)
    FEATURE_ORDER = None

app = FastAPI(title="Predictive Maintenance (Failure Risk) API",
              description="Simple API to predict failure risk using the trained RandomForest model.",
              version="0.1")

# -----------------------
# Request models
# -----------------------
class FeatureInput(BaseModel):
    """
    Provide a mapping of feature_name -> value.
    Must include the same features that the model was trained on.
    Example:
    {
      "features": {"sensor_1_r5_mean": 518.67, "sensor_1_r5_std": 0.003, ...}
    }
    """
    features: Dict[str, float]


class SimpleSensorInput(BaseModel):
    """
    Provide only base sensors and settings for a single unit (no history).
    This will create a minimal feature vector for demo: it will set rolling means to
    the supplied sensor values and trend features to zero. This is only for quick demos.
    Prefer FeatureInput for proper predictions.
    Example:
    {
      "sensors": {"sensor_1": 518.67, "sensor_2": 642.37, ...},
      "settings": {"setting1": 100.0, "setting2": 518.67, "setting3": 642.37}
    }
    """
    sensors: Dict[str, float]
    settings: Optional[Dict[str, float]] = None

# -----------------------
# Helpers
# -----------------------
def _prepare_from_features_dict(features_dict: Dict[str, float]):
    # If training used FEATURE_ORDER, ensure same order
    if FEATURE_ORDER is not None:
        missing = [f for f in FEATURE_ORDER if f not in features_dict]
        if missing:
            raise ValueError(f"Missing required features: {missing[:10]} (showing up to 10).")
        # create 2D array in right order
        arr = np.array([features_dict[f] for f in FEATURE_ORDER], dtype=float).reshape(1, -1)
        X = arr
    else:
        # if no order file, use the keys sorted to create deterministic order (user must be consistent)
        keys = sorted(features_dict.keys())
        arr = np.array([features_dict[k] for k in keys], dtype=float).reshape(1, -1)
        X = arr
    # impute/scale if available
    if imputer is not None:
        X = imputer.transform(X)
    if scaler is not None:
        X = scaler.transform(X)
    return X

def _prepare_from_simple_sensors(sensors: Dict[str, float], settings: Optional[Dict[str, float]] = None):
    """
    Quick demo builder: take base sensors and set rolling features = same sensor value,
    trend features = 0. This is not as powerful as using history but usable for a quick demo.
    """
    # build candidate feature dict
    features = {}
    # add sensors as-is
    for k, v in sensors.items():
        features[k] = float(v)
        # create a few rolling-like features commonly used
        features[f"{k}_r5_mean"] = float(v)
        features[f"{k}_r10_mean"] = float(v)
        features[f"{k}_r20_mean"] = float(v)
        features[f"{k}_trend"] = 0.0

    # add settings if provided
    if settings:
        for k, v in settings.items():
            features[k] = float(v)

    return _prepare_from_features_dict(features)

# -----------------------
# Endpoints
# -----------------------
@app.post("/predict_from_features")
def predict_from_features(payload: FeatureInput):
    try:
        X = _prepare_from_features_dict(payload.features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # prediction
    pred = model.predict(X)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # robust handling
        if proba.ndim == 2 and proba.shape[1] >= 2:
            score = float(proba[0, 1])
        else:
            score = float(proba[0, 0])
    else:
        score = float(pred)

    return {"prediction": int(pred), "failure_risk": score}


@app.post("/predict_simple")
def predict_simple(payload: SimpleSensorInput):
    try:
        X = _prepare_from_simple_sensors(payload.sensors, payload.settings)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    pred = model.predict(X)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        score = float(proba[0, 1]) if proba.shape[1] >= 2 else float(proba[0, 0])
    else:
        score = float(pred)

    return {"prediction": int(pred), "failure_risk": score}


@app.get("/")
def root():
    return {"status": "ok", "service": "predictive-maintenance-inference"}
