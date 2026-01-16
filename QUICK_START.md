# Quick Start Guide - ML Model Integration

## 🚀 Quick Setup (3 Steps)

### 1. Install Python Dependencies
```bash
cd ml-service
pip install -r requirements.txt
```

### 2. Install Node.js Dependencies
```bash
npm install
```

### 3. Start All Services

**Terminal 1 - Python ML Service:**
```bash
cd ml-service
python app.py
```

**Terminal 2 - Backend & Frontend:**
```bash
npm run dev
```

## ✅ Verify It Works

1. Open browser to `http://localhost:3000`
2. Log in to the dashboard
3. Check that "ML Service Online" shows green
4. Click "Get Predictions" button
5. View outbreak predictions!

## 📁 File Structure

```
├── ml-service/              # Python FastAPI service
│   ├── app.py              # Main prediction API
│   ├── requirements.txt   # Python dependencies
│   └── README.md          # Python service docs
├── server/
│   ├── routes/
│   │   └── predictions.js  # Express route for predictions
│   └── index.js           # Updated with predictions route
├── client/src/pages/
│   ├── Dashboard.js       # Updated with prediction UI
│   └── Dashboard.css      # Prediction styles
└── INTEGRATION_GUIDE.md    # Full documentation
```

## 🔍 Troubleshooting

**ML Service Offline?**
- Check Python service is running on port 8000
- Verify model files exist: `version-2 DL/davao_bilstm_attention_percentile_outbreak.keras`
- Check Python service terminal for errors

**Can't see predictions?**
- Ensure you're logged in
- Check browser console for errors
- Verify all three services are running

## 📚 More Help

See `INTEGRATION_GUIDE.md` for detailed documentation.
