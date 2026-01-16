import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import Navbar from '../components/Navbar';
import './Dashboard.css';

// Helper function to get week number
const getWeekNumber = (date) => {
  const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
  const dayNum = d.getUTCDay() || 7;
  d.setUTCDate(d.getUTCDate() + 4 - dayNum);
  const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
  return Math.ceil((((d - yearStart) / 86400000) + 1) / 7);
};

const Dashboard = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedDisease, setSelectedDisease] = useState('DENGUE');
  const [municipalityData, setMunicipalityData] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [predictionsLoading, setPredictionsLoading] = useState(false);
  const [mlServiceStatus, setMlServiceStatus] = useState(null);

  useEffect(() => {
    fetchStats();
    checkMlServiceStatus();
  }, []);

  useEffect(() => {
    if (selectedDisease) {
      fetchMunicipalityData(selectedDisease);
    }
  }, [selectedDisease]);

  const fetchStats = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get('http://localhost:5000/api/dashboard/stats', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setStats(response.data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchMunicipalityData = async (diseaseType) => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get(`http://localhost:5000/api/dashboard/municipalities/${diseaseType}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setMunicipalityData(response.data);
    } catch (error) {
      console.error('Error fetching municipality data:', error);
      setMunicipalityData([]);
    }
  };

  const checkMlServiceStatus = async () => {
    try {
      const token = localStorage.getItem('token');
      if (!token) {
        setMlServiceStatus({ status: 'offline', error: 'Not authenticated' });
        return;
      }
      
      const response = await axios.get('http://localhost:5000/api/predictions/health', {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      // Ensure status is set correctly
      const data = response.data;
      if (data.model_loaded && data.scaler_loaded) {
        data.status = 'online';
      }
      setMlServiceStatus(data);
    } catch (error) {
      console.error('ML service not available:', error);
      // Check if it's an auth error
      if (error.response?.status === 401 || error.response?.status === 403) {
        setMlServiceStatus({ 
          status: 'offline', 
          error: 'Authentication required. Please log in again.' 
        });
      } else if (error.response?.status === 503) {
        setMlServiceStatus({ 
          status: 'offline', 
          error: error.response.data?.details || 'ML service is not available' 
        });
      } else {
        setMlServiceStatus({ 
          status: 'offline', 
          error: 'Cannot connect to ML service. Ensure Python service is running on port 8000.' 
        });
      }
    }
  };

  // Generate sample weekly data for demonstration
  // In production, this should come from historical patient data
  const generateSampleWeeklyData = (baseCases = 2) => {
    const currentWeek = getWeekNumber(new Date());
    const sequences = [];
    
    // Build sequences in forward order (week -3, week -2, week -1, current week)
    for (let i = 0; i < 4; i++) {
      const week = currentWeek - 3 + i;
      const cases = baseCases + Math.random() * 2;
      
      // Calculate lags from previous sequences (if they exist)
      const lag1 = sequences.length > 0 ? sequences[sequences.length - 1][0] * 10 : cases * 0.9;
      const lag2 = sequences.length > 1 ? sequences[sequences.length - 2][0] * 10 : cases * 0.8;
      const lag3 = sequences.length > 2 ? sequences[sequences.length - 3][0] * 10 : cases * 0.7;
      
      // Calculate rolling statistics
      const allCases = [cases, lag1, lag2, lag3].filter(c => c > 0);
      const rollingMean = allCases.reduce((sum, c) => sum + c, 0) / allCases.length;
      const rollingStd = Math.sqrt(
        allCases.reduce((sum, c) => sum + Math.pow(c - rollingMean, 2), 0) / allCases.length
      ) || 0.1;
      
      // Normalize values (simplified - in production use actual scaler)
      sequences.push([
        cases / 10,      // confirmed_cases (normalized)
        lag1 / 10,       // lag_1
        lag2 / 10,       // lag_2
        lag3 / 10,       // lag_3
        rollingMean / 10, // rolling_mean_4
        rollingStd / 10,  // rolling_std_4
        week % 52        // week
      ]);
    }
    
    return sequences;
  };

  const fetchPredictions = async () => {
    setPredictionsLoading(true);
    try {
      const token = localStorage.getItem('token');
      
      // Get top municipalities for predictions
      const topMunicipalities = municipalityData.slice(0, 3);
      
      if (topMunicipalities.length === 0) {
        // Use sample data if no municipality data available
        const sampleMunicipalities = [
          { municipality: 'Nabunturan', barangay: 'Poblacion' },
          { municipality: 'Monkayo', barangay: 'Poblacion' },
          { municipality: 'Compostela', barangay: 'Poblacion' }
        ];
        
        const predictionRequests = sampleMunicipalities.map(loc => ({
          weekly_data: generateSampleWeeklyData(),
          municipality: loc.municipality,
          barangay: loc.barangay
        }));

        const response = await axios.post(
          'http://localhost:5000/api/predictions/predict-batch',
          { requests: predictionRequests },
          { headers: { Authorization: `Bearer ${token}` } }
        );
        
        setPredictions(response.data.results || []);
      } else {
        // Use actual municipality data
        const predictionRequests = topMunicipalities.map(item => ({
          weekly_data: generateSampleWeeklyData(item.count),
          municipality: item.municipality,
          barangay: 'Poblacion' // Default barangay - can be enhanced
        }));

        const response = await axios.post(
          'http://localhost:5000/api/predictions/predict-batch',
          { requests: predictionRequests },
          { headers: { Authorization: `Bearer ${token}` } }
        );
        
        setPredictions(response.data.results || []);
      }
    } catch (error) {
      console.error('Error fetching predictions:', error);
      console.error('Error details:', {
        message: error.message,
        status: error.response?.status,
        statusText: error.response?.statusText,
        data: error.response?.data,
        url: error.config?.url
      });
      
      // Show user-friendly error
      if (error.response?.status === 404) {
        console.error('Route not found. Make sure the server is running and routes are registered.');
      } else if (error.response?.status === 401 || error.response?.status === 403) {
        console.error('Authentication failed. Please log in again.');
      } else if (error.code === 'ECONNREFUSED') {
        console.error('Cannot connect to server. Is the Express backend running on port 5000?');
      }
      
      setPredictions([]);
    } finally {
      setPredictionsLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="App">
        <Navbar />
        <div className="dashboard-container">
          <div className="loading">Loading...</div>
        </div>
      </div>
    );
  }

  const chartData = municipalityData.map((item, index) => ({
    name: item.municipality,
    cases: item.count,
    color: index === 0 ? '#e74c3c' : '#f39c12' // First (highest) is red, others are orange
  }));

  return (
    <div className="App">
      <Navbar />
      <div className="dashboard-container">
        <h1>Disease Surveillance Dashboard - Davao de Oro</h1>
        
        <div className="dashboard-sections">
          <Link to="/patients" className="dashboard-card">
            <h3>Manage Patient Information</h3>
          </Link>
          <Link to="/patients" className="dashboard-card">
            <h3>Patient Logs</h3>
          </Link>
        </div>

        <div className="disease-categories">
          <div className="disease-card awd">
            <h2>AWD</h2>
            <p className="disease-count">{stats?.diseaseCounts?.awd || 0}</p>
          </div>
          <div className="disease-card dengue">
            <h2>DENGUE</h2>
            <p className="disease-count">{stats?.diseaseCounts?.dengue || 0}</p>
          </div>
          <div className="disease-card ili">
            <h2>ILI</h2>
            <p className="disease-count">{stats?.diseaseCounts?.ili || 0}</p>
          </div>
        </div>

        <div className="chart-section">
          <div className="chart-header">
            <h2>Top 3 Municipalities with Highest {selectedDisease} Cases</h2>
            <div className="disease-filters">
              <button 
                className={`filter-btn ${selectedDisease === 'AWD' ? 'active' : ''}`}
                onClick={() => setSelectedDisease('AWD')}
              >
                AWD
              </button>
              <button 
                className={`filter-btn ${selectedDisease === 'DENGUE' ? 'active' : ''}`}
                onClick={() => setSelectedDisease('DENGUE')}
              >
                DENGUE
              </button>
              <button 
                className={`filter-btn ${selectedDisease === 'ILI' ? 'active' : ''}`}
                onClick={() => setSelectedDisease('ILI')}
              >
                ILI
              </button>
            </div>
          </div>
          {chartData.length > 0 ? (
            <div className="chart-wrapper">
              <ResponsiveContainer width="100%" height={250}>
                <BarChart 
                  data={chartData} 
                  layout="vertical"
                  margin={{ top: 5, right: 80, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" hide />
                  <YAxis 
                    type="category" 
                    dataKey="name" 
                    width={150}
                    tick={{ fontSize: 14, fill: '#2c3e50' }}
                    axisLine={false}
                  />
                  <Tooltip />
                  <Bar 
                    dataKey="cases" 
                    radius={[0, 4, 4, 0]}
                    label={{ 
                      position: 'right', 
                      formatter: (value) => value,
                      fill: '#2c3e50',
                      fontSize: 14,
                      fontWeight: 'bold'
                    }}
                  >
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className="no-data">No data available for {selectedDisease}</p>
          )}
        </div>

        {/* Outbreak Prediction Section */}
        <div className="prediction-section">
          <div className="prediction-header">
            <h2>🔮 ILI Outbreak Predictions</h2>
            <div className="prediction-controls">
              {mlServiceStatus?.status === 'online' ? (
                <span className="service-status online">Deep Learning Service Online</span>
              ) : (
                <span className="service-status offline">Deep Learning Service Offline</span>
              )}
              <button 
                className="predict-btn"
                onClick={fetchPredictions}
                disabled={predictionsLoading || mlServiceStatus?.status !== 'online'}
              >
                {predictionsLoading ? 'Loading...' : 'Get Predictions'}
              </button>
            </div>
          </div>

          {mlServiceStatus?.status !== 'online' && (
            <div className="service-warning">
              <p>⚠️ ML Prediction Service is not available.</p>
              {mlServiceStatus?.error && (
                <p style={{ marginTop: '0.5rem', fontSize: '0.9rem' }}>
                  {mlServiceStatus.error}
                </p>
              )}
              <p style={{ marginTop: '0.5rem', fontSize: '0.9rem' }}>
                To start the service, run: <code>cd ml-service && python app.py</code>
              </p>
            </div>
          )}

          {predictions.length > 0 && (
            <>
              {/* Outbreak Alerts */}
              {predictions.filter(p => p.outbreak_predicted === 1 && !p.error).length > 0 && (
                <div className="outbreak-alerts">
                  <h3>🚨 High-Risk Areas (Outbreak Predicted)</h3>
                  <div className="alert-grid">
                    {predictions
                      .filter(p => p.outbreak_predicted === 1 && !p.error)
                      .map((pred, index) => (
                        <div key={index} className="alert-card high-risk">
                          <div className="alert-header">
                            <span className="alert-icon">🔴</span>
                            <h4>{pred.municipality}</h4>
                          </div>
                          <div className="alert-body">
                            <p className="alert-location">{pred.barangay}</p>
                            <p className="alert-cases">Predicted Cases: <strong>{pred.predicted_cases}</strong></p>
                            <p className="alert-probability">
                              Outbreak Probability: <strong>{(pred.outbreak_probability * 100).toFixed(1)}%</strong>
                            </p>
                            <span className="risk-badge high">HIGH RISK</span>
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}

              {/* All Predictions Table */}
              <div className="predictions-table-section">
                <h3>All Location Predictions</h3>
                <div className="predictions-table-wrapper">
                  <table className="predictions-table">
                    <thead>
                      <tr>
                        <th>Municipality</th>
                        <th>Barangay</th>
                        <th>Predicted Cases</th>
                        <th>Outbreak Probability</th>
                        <th>Risk Level</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {predictions.map((pred, index) => {
                        if (pred.error) {
                          return (
                            <tr key={index} className="error-row">
                              <td colSpan="6" className="error-message">
                                {pred.municipality} - {pred.barangay}: {pred.error}
                              </td>
                            </tr>
                          );
                        }
                        const riskClass = pred.risk_level?.toLowerCase() || 'low';
                        return (
                          <tr key={index} className={`risk-${riskClass}`}>
                            <td>{pred.municipality}</td>
                            <td>{pred.barangay}</td>
                            <td><strong>{pred.predicted_cases}</strong></td>
                            <td>{(pred.outbreak_probability * 100).toFixed(1)}%</td>
                            <td>
                              <span className={`risk-badge ${riskClass}`}>
                                {pred.risk_level || 'LOW'}
                              </span>
                            </td>
                            <td>
                              {pred.outbreak_predicted === 1 ? (
                                <span className="outbreak-indicator">🔴 Outbreak</span>
                              ) : (
                                <span className="normal-indicator">🟢 Normal</span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}

          {predictions.length === 0 && !predictionsLoading && mlServiceStatus?.status === 'online' && (
            <div className="no-predictions">
              <p>Click "Get Predictions" to see outbreak forecasts for top municipalities.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
