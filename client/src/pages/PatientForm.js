import React, { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import axios from 'axios';
import Navbar from '../components/Navbar';
import './PatientForm.css';

const PatientForm = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    patient_name: '',
    patient_age: '',
    sex: '',
    date_of_birth: '',
    municipality: '',
    street_purok: '',
    barangay: '',
    disease_type: '',
    admitted_times: '',
    date_of_admit: '',
    date_of_onset: '',
    laboratory_result: '',
    date_of_entry: ''
  });

  useEffect(() => {
    if (id) {
      fetchPatient();
    }
  }, [id]);

  const fetchPatient = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get(`http://localhost:5000/api/patients/${id}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setFormData(response.data);
    } catch (error) {
      console.error('Error fetching patient:', error);
      alert('Failed to load patient data');
    }
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const formatDateForInput = (dateString) => {
    if (!dateString) return '';
    // Convert DD/MM/YYYY to YYYY-MM-DD for input
    const parts = dateString.split('/');
    if (parts.length === 3) {
      return `${parts[2]}-${parts[1]}-${parts[0]}`;
    }
    return dateString;
  };

  const formatDateForStorage = (dateString) => {
    if (!dateString) return '';
    // Convert YYYY-MM-DD to DD/MM/YYYY
    const parts = dateString.split('-');
    if (parts.length === 3) {
      return `${parts[2]}/${parts[1]}/${parts[0]}`;
    }
    return dateString;
  };

  const handleDateChange = (e) => {
    const formatted = formatDateForStorage(e.target.value);
    setFormData({
      ...formData,
      [e.target.name]: formatted
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const token = localStorage.getItem('token');
      const url = id 
        ? `http://localhost:5000/api/patients/${id}`
        : 'http://localhost:5000/api/patients';
      const method = id ? 'put' : 'post';

      await axios[method](url, formData, {
        headers: { Authorization: `Bearer ${token}` }
      });

      alert(id ? 'Patient record updated successfully!' : 'Patient record saved successfully!');
      navigate('/patients');
    } catch (error) {
      console.error('Error saving patient:', error);
      alert('Failed to save patient record');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <Navbar />
      <div className="patient-form-container">
        <h1>{id ? 'Edit Patient Record' : 'Patient Details'}</h1>
        <form onSubmit={handleSubmit} className="patient-form">
          <div className="form-section">
            <h2>Patient Details</h2>
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="patient_name">Patient Name</label>
                <input
                  type="text"
                  id="patient_name"
                  name="patient_name"
                  value={formData.patient_name}
                  onChange={handleChange}
                  required
                />
              </div>
              <div className="form-group">
                <label htmlFor="patient_age">Patient Age</label>
                <input
                  type="number"
                  id="patient_age"
                  name="patient_age"
                  value={formData.patient_age}
                  onChange={handleChange}
                />
              </div>
            </div>
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="sex">Sex</label>
                <select
                  id="sex"
                  name="sex"
                  value={formData.sex}
                  onChange={handleChange}
                >
                  <option value="">Select</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>
              <div className="form-group">
                <label htmlFor="date_of_birth">Date of Birth</label>
                <input
                  type="date"
                  id="date_of_birth"
                  name="date_of_birth"
                  value={formatDateForInput(formData.date_of_birth)}
                  onChange={handleDateChange}
                />
              </div>
            </div>
          </div>

          <div className="form-section">
            <h2>Address Information</h2>
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="municipality">Municipality</label>
                <input
                  type="text"
                  id="municipality"
                  name="municipality"
                  value={formData.municipality}
                  onChange={handleChange}
                />
              </div>
              <div className="form-group">
                <label htmlFor="street_purok">Street / Purok</label>
                <input
                  type="text"
                  id="street_purok"
                  name="street_purok"
                  value={formData.street_purok}
                  onChange={handleChange}
                />
              </div>
            </div>
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="barangay">Barangay</label>
                <input
                  type="text"
                  id="barangay"
                  name="barangay"
                  value={formData.barangay}
                  onChange={handleChange}
                />
              </div>
            </div>
          </div>

          <div className="form-section">
            <h2>Admission & Disease Information</h2>
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="disease_type">Disease Type</label>
                <select
                  id="disease_type"
                  name="disease_type"
                  value={formData.disease_type}
                  onChange={handleChange}
                >
                  <option value="">Select Disease</option>
                  <option value="AWD">AWD</option>
                  <option value="DENGUE">DENGUE</option>
                  <option value="ILI">ILI</option>
                </select>
              </div>
              <div className="form-group">
                <label htmlFor="admitted_times">Admitted Times</label>
                <input
                  type="number"
                  id="admitted_times"
                  name="admitted_times"
                  value={formData.admitted_times}
                  onChange={handleChange}
                />
              </div>
            </div>
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="date_of_admit">Date of Admit</label>
                <input
                  type="date"
                  id="date_of_admit"
                  name="date_of_admit"
                  value={formatDateForInput(formData.date_of_admit)}
                  onChange={handleDateChange}
                />
              </div>
              <div className="form-group">
                <label htmlFor="date_of_onset">Date of Onset</label>
                <input
                  type="date"
                  id="date_of_onset"
                  name="date_of_onset"
                  value={formatDateForInput(formData.date_of_onset)}
                  onChange={handleDateChange}
                />
              </div>
            </div>
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="laboratory_result">Laboratory Result</label>
                <input
                  type="text"
                  id="laboratory_result"
                  name="laboratory_result"
                  value={formData.laboratory_result}
                  onChange={handleChange}
                  placeholder="Positive / Negative / Pending"
                />
              </div>
              <div className="form-group">
                <label htmlFor="date_of_entry">Date of Entry</label>
                <input
                  type="date"
                  id="date_of_entry"
                  name="date_of_entry"
                  value={formatDateForInput(formData.date_of_entry)}
                  onChange={handleDateChange}
                />
              </div>
            </div>
          </div>

          <button type="submit" disabled={loading} className="submit-button">
            {loading ? 'Saving...' : 'Save Patient Record'}
          </button>
        </form>
      </div>
    </div>
  );
};

export default PatientForm;
