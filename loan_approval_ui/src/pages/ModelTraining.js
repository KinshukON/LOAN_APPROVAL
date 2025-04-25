import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import {
  Box,
  Typography,
  Button,
  Paper,
  LinearProgress,
  Alert,
  Card,
  CardContent,
  Grid,
} from '@mui/material';
import {
  CloudUpload as CloudUploadIcon,
  PlayArrow as PlayArrowIcon,
} from '@mui/icons-material';

function ModelTraining() {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile && selectedFile.type === 'text/csv') {
      setFile(selectedFile);
      setError(null);
    } else {
      setError('Please select a valid CSV file');
      setFile(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setUploading(true);
    setError(null);
    setSuccess(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      await axios.post('http://localhost:5001/api/upload', formData);
      setSuccess('File uploaded successfully');
    } catch (err) {
      setError(err.response?.data?.error || 'Error uploading file');
    } finally {
      setUploading(false);
    }
  };

  const handleTrain = async () => {
    setTraining(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await axios.post('http://localhost:5001/api/train');
      setSuccess('Model trained successfully');
      setTimeout(() => navigate('/results'), 2000);
    } catch (err) {
      setError(err.response?.data?.error || 'Error training model');
    } finally {
      setTraining(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Train Model
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Step 1: Upload Data
              </Typography>
              <Box sx={{ mb: 2 }}>
                <input
                  accept=".csv"
                  style={{ display: 'none' }}
                  id="file-upload"
                  type="file"
                  onChange={handleFileChange}
                />
                <label htmlFor="file-upload">
                  <Button
                    variant="contained"
                    component="span"
                    startIcon={<CloudUploadIcon />}
                  >
                    Select CSV File
                  </Button>
                </label>
                {file && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    Selected file: {file.name}
                  </Typography>
                )}
              </Box>
              <Button
                variant="contained"
                color="primary"
                onClick={handleUpload}
                disabled={!file || uploading}
                fullWidth
              >
                {uploading ? 'Uploading...' : 'Upload'}
              </Button>
              {uploading && <LinearProgress sx={{ mt: 1 }} />}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Step 2: Train Model
              </Typography>
              <Button
                variant="contained"
                color="primary"
                onClick={handleTrain}
                disabled={training}
                startIcon={<PlayArrowIcon />}
                fullWidth
              >
                {training ? 'Training...' : 'Start Training'}
              </Button>
              {training && <LinearProgress sx={{ mt: 1 }} />}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}
          {success && (
            <Alert severity="success" sx={{ mt: 2 }}>
              {success}
            </Alert>
          )}
        </Grid>
      </Grid>
    </Box>
  );
}

export default ModelTraining; 