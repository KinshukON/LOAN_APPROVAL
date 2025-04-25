import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Tabs,
  Tab,
  Paper,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

function TabPanel({ children, value, index }) {
  return (
    <div hidden={value !== index} style={{ padding: '20px 0' }}>
      {value === index && children}
    </div>
  );
}

function Results() {
  const [activeTab, setActiveTab] = useState(0);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [featureImportance, setFeatureImportance] = useState([]);

  useEffect(() => {
    const fetchResults = async () => {
      try {
        const response = await axios.get('http://localhost:5001/api/results');
        setResults(response.data);
        
        // If feature importance data is available, fetch and parse it
        if (response.data.featureImportance) {
          try {
            const featureResponse = await axios.get(`http://localhost:5001${response.data.featureImportance}`);
            const features = featureResponse.data
              .split('\n')
              .filter(line => line.trim())
              .map(line => {
                const [name, value] = line.split(',').map(item => item.trim());
                return {
                  name: name.replace(/['"]/g, ''), // Remove any quotes
                  value: parseFloat(value)
                };
              })
              .filter(item => !isNaN(item.value)) // Filter out invalid numbers
              .sort((a, b) => b.value - a.value); // Sort by importance
            
            console.log('Parsed feature importance:', features);
            setFeatureImportance(features);
          } catch (err) {
            console.error('Error loading feature importance:', err);
            console.error('Error details:', {
              url: `http://localhost:5001${response.data.featureImportance}`,
              error: err.message
            });
          }
        }
      } catch (err) {
        setError('Error loading results: ' + err.message);
        console.error('Error details:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchResults();
  }, []);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  const NoDataMessage = ({ message }) => (
    <Box sx={{ p: 3, textAlign: 'center' }}>
      <Typography variant="body1" color="text.secondary">
        {message || 'No data available. Please train the model first.'}
      </Typography>
    </Box>
  );

  const ImageDisplay = ({ src, alt, errorMessage }) => {
    const [imgError, setImgError] = useState(false);

    if (imgError) {
      return <NoDataMessage message={errorMessage} />;
    }

    return (
      <Box
        component="img"
        src={src}
        alt={alt}
        sx={{ width: '100%', maxWidth: 800, display: 'block', margin: '0 auto' }}
        onError={(e) => {
          console.error(`Error loading ${alt}`);
          setImgError(true);
        }}
      />
    );
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Model Results
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={activeTab}
          onChange={(e, newValue) => setActiveTab(newValue)}
          indicatorColor="primary"
          textColor="primary"
        >
          <Tab label="Confusion Matrix" />
          <Tab label="Feature Importance" />
          <Tab label="LIME Explanations" />
          <Tab label="SHAP Explanations" />
        </Tabs>
      </Paper>

      <TabPanel value={activeTab} index={0}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Confusion Matrix
                </Typography>
                {results?.confusionMatrix ? (
                  <ImageDisplay
                    src={`http://localhost:5001${results.confusionMatrix}`}
                    alt="Confusion Matrix"
                    errorMessage="Confusion matrix not available. Please train the model first."
                  />
                ) : (
                  <NoDataMessage />
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={activeTab} index={1}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Feature Importance
                </Typography>
                {featureImportance.length > 0 ? (
                  <Box sx={{ height: 400 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={featureImportance} layout="vertical" margin={{ left: 150 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="number" />
                        <YAxis type="category" dataKey="name" width={150} />
                        <Tooltip formatter={(value) => value.toFixed(4)} />
                        <Legend />
                        <Bar dataKey="value" fill="#1976d2" name="Importance Score" />
                      </BarChart>
                    </ResponsiveContainer>
                  </Box>
                ) : (
                  <NoDataMessage message="Feature importance data not available. Please train the model first." />
                )}
                {process.env.NODE_ENV === 'development' && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="caption" display="block">
                      Debug: Feature Importance Data
                    </Typography>
                    <pre style={{ overflow: 'auto', fontSize: '0.8em' }}>
                      {JSON.stringify(featureImportance, null, 2)}
                    </pre>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={activeTab} index={2}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  LIME Explanations
                </Typography>
                {results?.limeExplanations ? (
                  <ImageDisplay
                    src={`http://localhost:5001${results.limeExplanations}/lime_explanation_1.png`}
                    alt="LIME Explanation"
                    errorMessage="LIME explanations not available. Please train the model first."
                  />
                ) : (
                  <NoDataMessage />
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={activeTab} index={3}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  SHAP Explanations
                </Typography>
                {results?.shapExplanations ? (
                  <ImageDisplay
                    src={`http://localhost:5001${results.shapExplanations}/shap_summary.png`}
                    alt="SHAP Summary"
                    errorMessage="SHAP explanations not available. Please train the model first."
                  />
                ) : (
                  <NoDataMessage />
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {results?.debug && process.env.NODE_ENV === 'development' && (
        <Card sx={{ mt: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Debug Information
            </Typography>
            <pre style={{ overflow: 'auto' }}>
              {JSON.stringify(results.debug, null, 2)}
            </pre>
          </CardContent>
        </Card>
      )}
    </Box>
  );
}

export default Results; 