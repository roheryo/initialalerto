# ML Prediction Service

FastAPI service for ILI outbreak predictions using Bi-LSTM + Attention model.

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure model files are in the correct location:
   - `../version-2 DL/davao_bilstm_attention_percentile_outbreak.keras`
   - `../version-2 DL/case_scaler.pkl`

3. Run the service:
```bash
python app.py
```

Or with uvicorn:
```bash
uvicorn app:app --reload --port 8000
```

The service will be available at `http://localhost:8000`

## API Endpoints

### Health Check
- `GET /` - Check if service is running and models are loaded

### Single Prediction
- `POST /predict` - Predict outbreak for a single location

Request body:
```json
{
  "weekly_data": [
    [0.1, 0.1, 0.2, 0.3, 0.2, 0.05, 12],
    [0.2, 0.1, 0.1, 0.2, 0.25, 0.06, 13],
    [0.25, 0.2, 0.1, 0.1, 0.3, 0.07, 14],
    [0.3, 0.25, 0.2, 0.1, 0.35, 0.08, 15]
  ],
  "municipality": "Nabunturan",
  "barangay": "Poblacion"
}
```

Response:
```json
{
  "municipality": "Nabunturan",
  "barangay": "Poblacion",
  "predicted_cases": 3,
  "outbreak_probability": 0.60,
  "outbreak_predicted": 1,
  "risk_level": "HIGH"
}
```

### Batch Prediction
- `POST /predict-batch` - Predict for multiple locations at once

## Testing

Visit `http://localhost:8000/docs` for interactive API documentation.
