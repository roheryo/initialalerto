const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { getDb } = require('../database/db');
const { authenticateToken } = require('../middleware/auth');

const router = express.Router();
const db = getDb();

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, '../uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadsDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({
  storage: storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit
  fileFilter: (req, file, cb) => {
    cb(null, true);
  }
});

// Upload file
router.post('/upload', authenticateToken, upload.single('file'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  const { filename, originalname, mimetype, size, path: filePath } = req.file;

  db.run(
    'INSERT INTO files (filename, originalname, mimetype, size, path) VALUES (?, ?, ?, ?, ?)',
    [filename, originalname, mimetype, size, filePath],
    function(err) {
      if (err) {
        // Delete uploaded file if database insert fails
        fs.unlinkSync(filePath);
        return res.status(500).json({ error: 'Database error' });
      }
      res.json({
        id: this.lastID,
        filename,
        originalname,
        mimetype,
        size,
        path: `/uploads/${filename}`,
        message: 'File uploaded successfully'
      });
    }
  );
});

// Get all files
router.get('/', authenticateToken, (req, res) => {
  db.all('SELECT * FROM files ORDER BY uploaded_at DESC', [], (err, rows) => {
    if (err) {
      return res.status(500).json({ error: 'Database error' });
    }
    // Update paths to be relative URLs
    const files = rows.map(file => ({
      ...file,
      path: `/uploads/${file.filename}`
    }));
    res.json(files);
  });
});

// Get file by ID
router.get('/:id', authenticateToken, (req, res) => {
  const { id } = req.params;
  db.get('SELECT * FROM files WHERE id = ?', [id], (err, row) => {
    if (err) {
      return res.status(500).json({ error: 'Database error' });
    }
    if (!row) {
      return res.status(404).json({ error: 'File not found' });
    }
    row.path = `/uploads/${row.filename}`;
    res.json(row);
  });
});

// Delete file
router.delete('/:id', authenticateToken, (req, res) => {
  const { id } = req.params;
  
  // Get file info first
  db.get('SELECT * FROM files WHERE id = ?', [id], (err, file) => {
    if (err) {
      return res.status(500).json({ error: 'Database error' });
    }
    if (!file) {
      return res.status(404).json({ error: 'File not found' });
    }

    // Delete from database
    db.run('DELETE FROM files WHERE id = ?', [id], (err) => {
      if (err) {
        return res.status(500).json({ error: 'Database error' });
      }

      // Delete physical file
      const filePath = path.join(__dirname, '../uploads', file.filename);
      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
      }

      res.json({ message: 'File deleted successfully' });
    });
  });
});

module.exports = router;
