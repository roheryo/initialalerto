import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import PatientForm from './pages/PatientForm';
import PatientList from './pages/PatientList';
import FileUpload from './pages/FileUpload';
import PrivateRoute from './components/PrivateRoute';
import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route
          path="/dashboard"
          element={
            <PrivateRoute>
              <Dashboard />
            </PrivateRoute>
          }
        />
        <Route
          path="/patients"
          element={
            <PrivateRoute>
              <PatientList />
            </PrivateRoute>
          }
        />
        <Route
          path="/patients/new"
          element={
            <PrivateRoute>
              <PatientForm />
            </PrivateRoute>
          }
        />
        <Route
          path="/patients/edit/:id"
          element={
            <PrivateRoute>
              <PatientForm />
            </PrivateRoute>
          }
        />
        <Route
          path="/files"
          element={
            <PrivateRoute>
              <FileUpload />
            </PrivateRoute>
          }
        />
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </Router>
  );
}

export default App;
