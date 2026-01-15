import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import Navbar from '../components/Navbar';
import './Dashboard.css';

const Dashboard = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedDisease, setSelectedDisease] = useState('DENGUE');
  const [municipalityData, setMunicipalityData] = useState([]);

  useEffect(() => {
    fetchStats();
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
      </div>
    </div>
  );
};

export default Dashboard;
