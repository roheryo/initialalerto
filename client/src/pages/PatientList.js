import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import Navbar from '../components/Navbar';
import './PatientList.css';

const PatientList = () => {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchPatients();
  }, []);

  const fetchPatients = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get('http://localhost:5000/api/patients', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setPatients(response.data);
    } catch (error) {
      console.error('Error fetching patients:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Are you sure you want to delete this patient record?')) {
      return;
    }

    try {
      const token = localStorage.getItem('token');
      await axios.delete(`http://localhost:5000/api/patients/${id}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      fetchPatients();
      alert('Patient record deleted successfully');
    } catch (error) {
      console.error('Error deleting patient:', error);
      alert('Failed to delete patient record');
    }
  };

  if (loading) {
    return (
      <div className="App">
        <Navbar />
        <div className="patient-list-container">
          <div className="loading">Loading...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="App">
      <Navbar />
      <div className="patient-list-container">
        <div className="patient-list-header">
          <h1>Patient Records</h1>
          <Link to="/patients/new" className="add-button">
            + Add New Patient
          </Link>
        </div>

        {patients.length === 0 ? (
          <div className="no-patients">
            <p>No patient records found.</p>
            <Link to="/patients/new" className="add-button">
              Add Your First Patient
            </Link>
          </div>
        ) : (
          <div className="patient-table-container">
            <table className="patient-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Age</th>
                  <th>Sex</th>
                  <th>Municipality</th>
                  <th>Disease Type</th>
                  <th>Date of Entry</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {patients.map((patient) => (
                  <tr key={patient.id}>
                    <td>{patient.patient_name}</td>
                    <td>{patient.patient_age || '-'}</td>
                    <td>{patient.sex || '-'}</td>
                    <td>{patient.municipality || '-'}</td>
                    <td>{patient.disease_type || '-'}</td>
                    <td>{patient.date_of_entry || '-'}</td>
                    <td>
                      <Link
                        to={`/patients/edit/${patient.id}`}
                        className="action-button edit"
                      >
                        Edit
                      </Link>
                      <button
                        onClick={() => handleDelete(patient.id)}
                        className="action-button delete"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default PatientList;
