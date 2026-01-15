const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const bcrypt = require('bcryptjs');

const dbPath = path.join(__dirname, 'database.sqlite');
const db = new sqlite3.Database(dbPath);

const init = () => {
  return new Promise((resolve, reject) => {
    db.serialize(() => {
      // Users table
      db.run(`CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )`, (err) => {
        if (err) reject(err);
      });

      // Patients table
      db.run(`CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_name TEXT NOT NULL,
        patient_age INTEGER,
        sex TEXT,
        date_of_birth TEXT,
        municipality TEXT,
        street_purok TEXT,
        barangay TEXT,
        disease_type TEXT,
        admitted_times INTEGER,
        date_of_admit TEXT,
        date_of_onset TEXT,
        laboratory_result TEXT,
        date_of_entry TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )`, (err) => {
        if (err) reject(err);
      });

      // Files table
      db.run(`CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        originalname TEXT NOT NULL,
        mimetype TEXT,
        size INTEGER,
        path TEXT NOT NULL,
        uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )`, (err) => {
        if (err) reject(err);
      });

      // Create default admin user
      const defaultPassword = bcrypt.hashSync('admin123', 10);
      db.run(`INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)`, 
        ['admin', defaultPassword], (err) => {
        if (err) console.error('Error creating default user:', err);
        else console.log('Database initialized successfully');
        resolve();
      });
    });
  });
};

const getDb = () => db;

module.exports = { init, getDb };
