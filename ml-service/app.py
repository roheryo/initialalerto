"""
FastAPI service for Dengue Outbreak Prediction
Uses Bi-LSTM + Attention model for forecasting
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import os
from pathlib import Path

app = FastAPI(title="Dengue Outbreak Prediction API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
MODEL_DIR = Path(__file__).parent.parent / "version-2 DL"
# Try .keras first, then .h5
MODEL_PATH_KERAS = MODEL_DIR / "davao_bilstm_attention_percentile_outbreak.keras"
MODEL_PATH_H5 = MODEL_DIR / "davao_bilstm_attention_percentile_outbreak.h5"
SCALER_PATH = MODEL_DIR / "case_scaler.pkl"

# Determine which model file exists
if MODEL_PATH_KERAS.exists():
    MODEL_PATH = MODEL_PATH_KERAS
elif MODEL_PATH_H5.exists():
    MODEL_PATH = MODEL_PATH_H5
else:
    MODEL_PATH = MODEL_PATH_KERAS  # Default to .keras

TIME_STEPS = 4
FEATURE_COLS = [
    "confirmed_cases",
    "lag_1", "lag_2", "lag_3",
    "rolling_mean_4", "rolling_std_4",
    "week"
]

# Global variables for model and scaler
model = None
scaler = None

# Custom Attention Layer (must match the one used during training)
class Attention(Layer):
    """Custom attention layer used in the Bi-LSTM model"""
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="normal",
            name="attention_W"
        )
        self.b = self.add_weight(
            shape=(input_shape[1], 1),
            initializer="zeros",
            name="attention_b"
        )
        super().build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        return K.sum(x * a, axis=1)
    
    def get_config(self):
        config = super().get_config()
        return config

def load_ml_models():
    """Load model and scaler once at startup"""
    global model, scaler
    
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
        
        print(f"Loading model from {MODEL_PATH}")
        # Provide custom_objects to load the custom Attention layer
        custom_objects = {'Attention': Attention}
        model = load_model(str(MODEL_PATH), compile=False, custom_objects=custom_objects)
        print(f"Loading scaler from {SCALER_PATH}")
        scaler = joblib.load(str(SCALER_PATH))
        print("Model and scaler loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load models when the server starts"""
    load_ml_models()

# Request/Response models
class PredictionRequest(BaseModel):
    weekly_data: list  # List of 4 sequences, each with 7 features
    municipality: str
    barangay: str

class PredictionResponse(BaseModel):
    municipality: str
    barangay: str
    predicted_cases: int
    outbreak_probability: float
    outbreak_predicted: int
    risk_level: str

def get_risk_level(probability: float, outbreak_predicted: int) -> str:
    """Determine risk level based on probability"""
    if outbreak_predicted == 1:
        return "HIGH"
    elif probability >= 0.2:
        return "MEDIUM"
    else:
        return "LOW"

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict dengue outbreak for a given location
    
    Expected input:
    {
      "weekly_data": [
        [confirmed_cases, lag_1, lag_2, lag_3, rolling_mean_4, rolling_std_4, week],
        ... (4 sequences total)
      ],
      "municipality": "Nabunturan",
      "barangay": "Poblacion"
    }
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate input
        if len(request.weekly_data) != TIME_STEPS:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {TIME_STEPS} sequences, got {len(request.weekly_data)}"
            )
        
        for i, seq in enumerate(request.weekly_data):
            if len(seq) != len(FEATURE_COLS):
                raise HTTPException(
                    status_code=400,
                    detail=f"Sequence {i} has {len(seq)} features, expected {len(FEATURE_COLS)}"
                )
        
        # Convert to numpy array and reshape
        sequence = np.array(request.weekly_data, dtype=np.float32)
        sequence = sequence.reshape(1, TIME_STEPS, len(FEATURE_COLS))
        
        # Make prediction
        pred_cases_scaled, pred_outbreak = model.predict(sequence, verbose=0)
        
        # Inverse scale case count
        # Create dummy array with same shape as scaler expects
        dummy = np.zeros((1, len(FEATURE_COLS)))
        dummy[:, 0] = pred_cases_scaled.flatten()
        predicted_cases = scaler.inverse_transform(dummy)[0][0]
        predicted_cases = max(0, int(round(predicted_cases)))  # Ensure non-negative
        
        # Get outbreak probability
        outbreak_prob = float(pred_outbreak[0][0])
        outbreak_predicted = 1 if outbreak_prob >= 0.3 else 0
        
        risk_level = get_risk_level(outbreak_prob, outbreak_predicted)
        
        return PredictionResponse(
            municipality=request.municipality,
            barangay=request.barangay,
            predicted_cases=predicted_cases,
            outbreak_probability=round(outbreak_prob, 4),
            outbreak_predicted=outbreak_predicted,
            risk_level=risk_level
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-batch")
def predict_batch(requests: list[PredictionRequest]):
    """
    Predict for multiple locations at once
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for request in requests:
        try:
            result = predict(request)
            results.append(result.dict())
        except Exception as e:
            results.append({
                "municipality": request.municipality,
                "barangay": request.barangay,
                "error": str(e)
            })
    
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
