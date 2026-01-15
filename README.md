# Disease Surveillance Dashboard

A full-stack web application for managing disease surveillance data with patient information management and file upload capabilities.

## Features

- **User Authentication**: Secure login system with JWT tokens
- **Dashboard**: Visual dashboard with disease statistics and charts
- **Patient Management**: Create, read, update, and delete patient records
- **File Upload**: Upload and manage files with preview capabilities
- **SQLite Database**: Lightweight database for data storage

## Tech Stack

- **Frontend**: React.js, React Router, Axios, Recharts
- **Backend**: Node.js, Express.js
- **Database**: SQLite3
- **Authentication**: JWT (JSON Web Tokens)
- **File Upload**: Multer

## Installation

1. Install all dependencies:
```bash
npm run install-all
```

2. Start the development server:
```bash
npm run dev
```

This will start both the backend server (port 5000) and the React frontend (port 3000).

## Default Login Credentials

- **Username**: admin
- **Password**: admin123

## Project Structure

```
.
├── server/
│   ├── database/
│   │   └── db.js          # Database initialization
│   ├── middleware/
│   │   └── auth.js        # JWT authentication middleware
│   ├── routes/
│   │   ├── auth.js        # Authentication routes
│   │   ├── patients.js    # Patient CRUD routes
│   │   ├── dashboard.js   # Dashboard statistics
│   │   └── files.js       # File upload routes
│   ├── uploads/           # Uploaded files storage
│   └── index.js           # Express server
├── client/
│   ├── public/
│   ├── src/
│   │   ├── components/    # Reusable components
│   │   ├── pages/         # Page components
│   │   ├── App.js         # Main app component
│   │   └── index.js       # Entry point
│   └── package.json
└── package.json
```

## API Endpoints

### Authentication
- `POST /api/auth/login` - User login

### Patients
- `GET /api/patients` - Get all patients
- `GET /api/patients/:id` - Get patient by ID
- `POST /api/patients` - Create new patient
- `PUT /api/patients/:id` - Update patient
- `DELETE /api/patients/:id` - Delete patient

### Dashboard
- `GET /api/dashboard/stats` - Get dashboard statistics

### Files
- `GET /api/files` - Get all files
- `GET /api/files/:id` - Get file by ID
- `POST /api/files/upload` - Upload file
- `DELETE /api/files/:id` - Delete file

## Notes

- The SQLite database file will be created automatically in `server/database/database.sqlite`
- Uploaded files are stored in `server/uploads/`
- Make sure to change the JWT_SECRET in production
