const express = require('express');
const { getDb } = require('../database/db');
const { authenticateToken } = require('../middleware/auth');

const router = express.Router();
const db = getDb();

// Get all patients
router.get('/', authenticateToken, (req, res) => {
  db.all('SELECT * FROM patients ORDER BY created_at DESC', [], (err, rows) => {
    if (err) {
      return res.status(500).json({ error: 'Database error' });
    }
    res.json(rows);
  });
});

// Get patient by ID
router.get('/:id', authenticateToken, (req, res) => {
  const { id } = req.params;
  db.get('SELECT * FROM patients WHERE id = ?', [id], (err, row) => {
    if (err) {
      return res.status(500).json({ error: 'Database error' });
    }
    if (!row) {
      return res.status(404).json({ error: 'Patient not found' });
    }
    res.json(row);
  });
});

// Create new patient
router.post('/', authenticateToken, (req, res) => {
  const {
    patient_name,
    patient_age,
    sex,
    date_of_birth,
    municipality,
    street_purok,
    barangay,
    disease_type,
    admitted_times,
    date_of_admit,
    date_of_onset,
    laboratory_result,
    date_of_entry
  } = req.body;

  db.run(
    `INSERT INTO patients (
      patient_name, patient_age, sex, date_of_birth,
      municipality, street_purok, barangay,
      disease_type, admitted_times, date_of_admit,
      date_of_onset, laboratory_result, date_of_entry
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      patient_name, patient_age, sex, date_of_birth,
      municipality, street_purok, barangay,
      disease_type, admitted_times, date_of_admit,
      date_of_onset, laboratory_result, date_of_entry
    ],
    function(err) {
      if (err) {
        return res.status(500).json({ error: 'Database error' });
      }
      res.json({ id: this.lastID, message: 'Patient record created successfully' });
    }
  );
});

// Update patient
router.put('/:id', authenticateToken, (req, res) => {
  const { id } = req.params;
  const {
    patient_name,
    patient_age,
    sex,
    date_of_birth,
    municipality,
    street_purok,
    barangay,
    disease_type,
    admitted_times,
    date_of_admit,
    date_of_onset,
    laboratory_result,
    date_of_entry
  } = req.body;

  db.run(
    `UPDATE patients SET
      patient_name = ?, patient_age = ?, sex = ?, date_of_birth = ?,
      municipality = ?, street_purok = ?, barangay = ?,
      disease_type = ?, admitted_times = ?, date_of_admit = ?,
      date_of_onset = ?, laboratory_result = ?, date_of_entry = ?
    WHERE id = ?`,
    [
      patient_name, patient_age, sex, date_of_birth,
      municipality, street_purok, barangay,
      disease_type, admitted_times, date_of_admit,
      date_of_onset, laboratory_result, date_of_entry,
      id
    ],
    function(err) {
      if (err) {
        return res.status(500).json({ error: 'Database error' });
      }
      if (this.changes === 0) {
        return res.status(404).json({ error: 'Patient not found' });
      }
      res.json({ message: 'Patient record updated successfully' });
    }
  );
});

// Delete patient
router.delete('/:id', authenticateToken, (req, res) => {
  const { id } = req.params;
  db.run('DELETE FROM patients WHERE id = ?', [id], function(err) {
    if (err) {
      return res.status(500).json({ error: 'Database error' });
    }
    if (this.changes === 0) {
      return res.status(404).json({ error: 'Patient not found' });
    }
    res.json({ message: 'Patient record deleted successfully' });
  });
});

module.exports = router;
