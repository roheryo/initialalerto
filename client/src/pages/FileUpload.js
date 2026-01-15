import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Navbar from '../components/Navbar';
import './FileUpload.css';

const FileUpload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchFiles();
  }, []);

  const fetchFiles = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get('http://localhost:5000/api/files', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setFiles(response.data);
    } catch (error) {
      console.error('Error fetching files:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      alert('Please select a file');
      return;
    }

    setUploading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const token = localStorage.getItem('token');
      await axios.post('http://localhost:5000/api/files/upload', formData, {
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'multipart/form-data'
        }
      });

      alert('File uploaded successfully!');
      setSelectedFile(null);
      document.getElementById('file-input').value = '';
      fetchFiles();
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Failed to upload file');
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Are you sure you want to delete this file?')) {
      return;
    }

    try {
      const token = localStorage.getItem('token');
      await axios.delete(`http://localhost:5000/api/files/${id}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      fetchFiles();
      alert('File deleted successfully');
    } catch (error) {
      console.error('Error deleting file:', error);
      alert('Failed to delete file');
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const isImage = (mimetype) => {
    return mimetype && mimetype.startsWith('image/');
  };

  if (loading) {
    return (
      <div className="App">
        <Navbar />
        <div className="file-upload-container">
          <div className="loading">Loading...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="App">
      <Navbar />
      <div className="file-upload-container">
        <h1>File Upload</h1>

        <div className="upload-section">
          <h2>Upload New File</h2>
          <form onSubmit={handleUpload} className="upload-form">
            <div className="file-input-wrapper">
              <input
                type="file"
                id="file-input"
                onChange={handleFileChange}
                className="file-input"
              />
              {selectedFile && (
                <div className="selected-file">
                  Selected: {selectedFile.name} ({formatFileSize(selectedFile.size)})
                </div>
              )}
            </div>
            <button type="submit" disabled={uploading || !selectedFile} className="upload-button">
              {uploading ? 'Uploading...' : 'Upload File'}
            </button>
          </form>
        </div>

        <div className="files-section">
          <h2>Uploaded Files</h2>
          {files.length === 0 ? (
            <div className="no-files">
              <p>No files uploaded yet.</p>
            </div>
          ) : (
            <div className="files-grid">
              {files.map((file) => (
                <div key={file.id} className="file-card">
                  {isImage(file.mimetype) ? (
                    <div className="file-preview">
                      <img
                        src={`http://localhost:5000${file.path}`}
                        alt={file.originalname}
                        className="preview-image"
                      />
                    </div>
                  ) : (
                    <div className="file-icon">
                      <svg
                        width="64"
                        height="64"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                      >
                        <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" />
                        <polyline points="13 2 13 9 20 9" />
                      </svg>
                    </div>
                  )}
                  <div className="file-info">
                    <h3 className="file-name" title={file.originalname}>
                      {file.originalname}
                    </h3>
                    <p className="file-meta">
                      {formatFileSize(file.size)} • {formatDate(file.uploaded_at)}
                    </p>
                    <div className="file-actions">
                      <a
                        href={`http://localhost:5000${file.path}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="action-link"
                      >
                        View
                      </a>
                      <button
                        onClick={() => handleDelete(file.id)}
                        className="action-link delete"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default FileUpload;
