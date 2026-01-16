# ML Model Integration Guide

This guide explains how to integrate the trained Bi-LSTM + Attention model into the web application.

## 📋 Overview

The integration consists of three main components:
1. **Python FastAPI Service** - Handles ML model predictions
2. **Express.js Backend** - Proxies requests to Python service
3. **React Dashboard** - Displays predictions with visual indicators

## 🚀 Setup Instructions

### Step 1: Install Python Dependencies

1. Navigate to the `ml-service` directory:
```bash
cd ml-service
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **Linux/Mac:**
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

### Step 2: Verify Model Files

Ensure the following files exist in `version-2 DL/`:
- `davao_bilstm_attention_percentile_outbreak.keras`
- `case_scaler.pkl`

### Step 3: Install Node.js Dependencies

From the project root:
```bash
npm install
```

This will install `axios` which is needed for the Express backend to communicate with the Python service.

### Step 4: Start the Services

You need to run **three services** simultaneously:

#### Terminal 1: Python ML Service
```bash
cd ml-service
python app.py
```
Or with uvicorn:
```bash
uvicorn app:app --reload --port 8000
```

The service will be available at `http://localhost:8000`

#### Terminal 2: Express Backend
```bash
npm run server
```

The backend will run on `http://localhost:5000`

#### Terminal 3: React Frontend
```bash
npm run client
```

The frontend will run on `http://localhost:3000`

**Or use the convenience script:**
```bash
npm run dev
```

This runs both the server and client concurrently (you'll still need to start the Python service separately).

## 🧪 Testing the Integration

### 1. Test Python Service Directly

Visit `http://localhost:8000/docs` for interactive API documentation.

Test with a sample request:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "weekly_data": [
      [0.1, 0.1, 0.2, 0.3, 0.2, 0.05, 12],
      [0.2, 0.1, 0.1, 0.2, 0.25, 0.06, 13],
      [0.25, 0.2, 0.1, 0.1, 0.3, 0.07, 14],
      [0.3, 0.25, 0.2, 0.1, 0.35, 0.08, 15]
    ],
    "municipality": "Nabunturan",
    "barangay": "Poblacion"
  }'
```

### 2. Test from Dashboard

1. Log in to the web application
2. Navigate to the Dashboard
3. Check that "ML Service Online" status is green
4. Click "Get Predictions" button
5. View the outbreak predictions table and alerts

## 📊 Understanding the Predictions

### Input Format

The model expects **4 sequences** of **7 features** each:

1. `confirmed_cases` - Number of confirmed cases (normalized)
2. `lag_1` - Cases from 1 week ago (normalized)
3. `lag_2` - Cases from 2 weeks ago (normalized)
4. `lag_3` - Cases from 3 weeks ago (normalized)
5. `rolling_mean_4` - 4-week rolling mean (normalized)
6. `rolling_std_4` - 4-week rolling standard deviation (normalized)
7. `week` - Week number (0-51)

### Output Format

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

### Risk Levels

- **HIGH** - `outbreak_predicted = 1` (probability ≥ 0.3)
- **MEDIUM** - `0.2 ≤ probability < 0.3`
- **LOW** - `probability < 0.2`

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the `server/` directory:

```env
PORT=5000
ML_SERVICE_URL=http://localhost:8000
JWT_SECRET=your-secret-key
```

### Model Path

If your model files are in a different location, update `MODEL_DIR` in `ml-service/app.py`:

```python
MODEL_DIR = Path(__file__).parent.parent / "version-2 DL"
```

## 🐛 Troubleshooting

### ML Service Not Available

**Error:** "ML Prediction Service is not available"

**Solutions:**
1. Ensure Python service is running on port 8000
2. Check that model files exist in the correct location
3. Verify Python dependencies are installed
4. Check Python service logs for errors

### Model Loading Errors

**Error:** "Model file not found"

**Solutions:**
1. Verify file path in `ml-service/app.py`
2. Check file permissions
3. Ensure you're using `.keras` format (not `.h5`)

### CORS Errors

If you see CORS errors, ensure the Python service has CORS enabled (already configured in `app.py`).

### Port Conflicts

If port 8000 is already in use:
1. Change port in `ml-service/app.py`: `uvicorn.run(app, host="0.0.0.0", port=8001)`
2. Update `ML_SERVICE_URL` in Express backend

## 📈 Production Considerations

### Performance

- Model is loaded once at startup (not per request)
- Consider using a model server like TensorFlow Serving for high traffic
- Implement caching for repeated predictions

### Security

- Add authentication to Python service
- Validate and sanitize all inputs
- Rate limit prediction endpoints
- Use HTTPS in production

### Data Pipeline

Currently, the Dashboard generates sample weekly data. In production:

1. **Extract historical data** from patient database
2. **Calculate features** (lags, rolling statistics)
3. **Normalize** using the saved scaler
4. **Format** as sequences for the model

Consider creating a data preprocessing service that:
- Aggregates patient data by week and location
- Calculates all required features
- Maintains historical sequences

## 🔄 Next Steps

1. **Automate Predictions**: Set up a cron job to run predictions weekly
2. **Database Storage**: Save predictions to database for trend analysis
3. **Historical Data Integration**: Connect to actual patient data
4. **GIS Mapping**: Visualize predictions on a map
5. **Alert System**: Send notifications for high-risk areas

## 📚 API Documentation

### Express Backend Endpoints

- `POST /api/predictions/predict` - Single prediction
- `POST /api/predictions/predict-batch` - Batch predictions
- `GET /api/predictions/health` - Service health check

### Python Service Endpoints

- `GET /` - Health check
- `POST /predict` - Single prediction
- `POST /predict-batch` - Batch predictions

See `ml-service/README.md` for detailed Python API documentation.

## 🆘 Support

If you encounter issues:
1. Check service logs
2. Verify all dependencies are installed
3. Ensure model files are accessible
4. Test each service independently
