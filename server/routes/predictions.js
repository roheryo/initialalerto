const express = require('express');
const axios = require('axios');
const { authenticateToken } = require('../middleware/auth');

const router = express.Router();

// ML Service URL - adjust if running on different port
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

// Root route for debugging
router.get('/', (req, res) => {
  res.json({ 
    message: 'Predictions API is working',
    endpoints: [
      'GET /api/predictions/health - Health check',
      'POST /api/predictions/predict - Single prediction',
      'POST /api/predictions/predict-batch - Batch predictions'
    ]
  });
});

/**
 * Proxy prediction request to Python ML service
 * POST /api/predictions/predict
 */
router.post('/predict', authenticateToken, async (req, res) => {
  try {
    const { weekly_data, municipality, barangay } = req.body;

    // Validate input
    if (!weekly_data || !municipality || !barangay) {
      return res.status(400).json({
        error: 'Missing required fields: weekly_data, municipality, barangay'
      });
    }

    if (!Array.isArray(weekly_data) || weekly_data.length !== 4) {
      return res.status(400).json({
        error: 'weekly_data must be an array of 4 sequences'
      });
    }

    // Forward request to Python ML service
    const response = await axios.post(
      `${ML_SERVICE_URL}/predict`,
      {
        weekly_data,
        municipality,
        barangay
      },
      {
        timeout: 30000 // 30 second timeout
      }
    );

    res.json(response.data);
  } catch (error) {
    console.error('Prediction error:', error.message);
    
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({
        error: 'ML prediction service is not available. Please ensure the Python service is running on port 8000.'
      });
    }

    if (error.response) {
      // Python service returned an error
      return res.status(error.response.status).json({
        error: error.response.data.detail || 'Prediction failed'
      });
    }

    res.status(500).json({
      error: 'Internal server error during prediction'
    });
  }
});

/**
 * Get prediction for multiple locations
 * POST /api/predictions/predict-batch
 */
router.post('/predict-batch', authenticateToken, async (req, res) => {
  try {
    const { requests } = req.body;

    if (!Array.isArray(requests)) {
      return res.status(400).json({
        error: 'requests must be an array'
      });
    }

    const response = await axios.post(
      `${ML_SERVICE_URL}/predict-batch`,
      requests,
      {
        timeout: 60000 // 60 second timeout for batch
      }
    );

    res.json(response.data);
  } catch (error) {
    console.error('Batch prediction error:', error.message);
    
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({
        error: 'ML prediction service is not available'
      });
    }

    if (error.response) {
      return res.status(error.response.status).json({
        error: error.response.data.detail || 'Batch prediction failed'
      });
    }

    res.status(500).json({
      error: 'Internal server error during batch prediction'
    });
  }
});

/**
 * Health check for ML service
 * GET /api/predictions/health
 */
router.get('/health', authenticateToken, async (req, res) => {
  try {
    const response = await axios.get(`${ML_SERVICE_URL}/`, {
      timeout: 5000
    });
    // Ensure status is 'online' if model is loaded
    const healthData = response.data;
    if (healthData.model_loaded && healthData.scaler_loaded) {
      healthData.status = 'online';
    }
    res.json(healthData);
  } catch (error) {
    console.error('Health check error:', error.message);
    res.status(503).json({
      status: 'offline',
      error: 'ML service is not available',
      details: error.code === 'ECONNREFUSED' 
        ? 'Connection refused. Is the Python service running on port 8000?'
        : error.message
    });
  }
});

module.exports = router;
