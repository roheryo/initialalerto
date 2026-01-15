const express = require('express');
const { getDb } = require('../database/db');
const { authenticateToken } = require('../middleware/auth');

const router = express.Router();
const db = getDb();

// Get dashboard statistics
router.get('/stats', authenticateToken, (req, res) => {
  const stats = {};

  // Get total patients
  db.get('SELECT COUNT(*) as total FROM patients', [], (err, row) => {
    if (err) {
      return res.status(500).json({ error: 'Database error' });
    }
    stats.totalPatients = row.total;

    // Get disease type counts
    db.all(
      `SELECT disease_type, COUNT(*) as count 
       FROM patients 
       WHERE disease_type IS NOT NULL AND disease_type != ''
       GROUP BY disease_type`,
      [],
      (err, rows) => {
        if (err) {
          return res.status(500).json({ error: 'Database error' });
        }
        stats.diseaseTypes = rows;

        // Get counts for AWD, DENGUE, ILI
        db.all(
          `SELECT 
            SUM(CASE WHEN disease_type = 'AWD' THEN 1 ELSE 0 END) as awd,
            SUM(CASE WHEN disease_type = 'DENGUE' THEN 1 ELSE 0 END) as dengue,
            SUM(CASE WHEN disease_type = 'ILI' THEN 1 ELSE 0 END) as ili
           FROM patients`,
          [],
          (err, countRow) => {
            if (err) {
              return res.status(500).json({ error: 'Database error' });
            }
            stats.diseaseCounts = countRow[0] || { awd: 0, dengue: 0, ili: 0 };
            res.json(stats);
          }
        );
      }
    );
  });
});

// Get top municipalities by disease type
router.get('/municipalities/:diseaseType', authenticateToken, (req, res) => {
  const { diseaseType } = req.params;
  
  // Validate disease type
  const validTypes = ['AWD', 'DENGUE', 'ILI'];
  if (!validTypes.includes(diseaseType)) {
    return res.status(400).json({ error: 'Invalid disease type' });
  }

  db.all(
    `SELECT municipality, COUNT(*) as count 
     FROM patients 
     WHERE disease_type = ? 
     AND municipality IS NOT NULL AND municipality != ''
     GROUP BY municipality 
     ORDER BY count DESC 
     LIMIT 3`,
    [diseaseType],
    (err, rows) => {
      if (err) {
        return res.status(500).json({ error: 'Database error' });
      }
      res.json(rows);
    }
  );
});

module.exports = router;
