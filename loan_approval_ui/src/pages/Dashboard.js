import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Paper,
} from '@mui/material';
import {
  Assessment as AssessmentIcon,
  PlayArrow as PlayArrowIcon,
  BarChart as BarChartIcon,
} from '@mui/icons-material';

function Dashboard() {
  const navigate = useNavigate();

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <AssessmentIcon sx={{ mr: 1 }} />
                <Typography variant="h6">Model Status</Typography>
              </Box>
              <Typography variant="body1" color="text.secondary">
                Current model: XGBoost Classifier
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Last trained: Not available
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <PlayArrowIcon sx={{ mr: 1 }} />
                <Typography variant="h6">Quick Actions</Typography>
              </Box>
              <Button
                variant="contained"
                color="primary"
                fullWidth
                onClick={() => navigate('/train')}
                sx={{ mb: 1 }}
              >
                Train New Model
              </Button>
              <Button
                variant="outlined"
                color="primary"
                fullWidth
                onClick={() => navigate('/results')}
              >
                View Results
              </Button>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <BarChartIcon sx={{ mr: 1 }} />
                <Typography variant="h6">Model Performance</Typography>
              </Box>
              <Typography variant="body1" color="text.secondary">
                Accuracy: Not available
              </Typography>
              <Typography variant="body2" color="text.secondary">
                F1 Score: Not available
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              About the Model
            </Typography>
            <Typography variant="body1" paragraph>
              This loan approval model uses machine learning to predict whether a loan application should be approved or not.
              The model takes into account various factors such as credit score, income, debt-to-income ratio, and other financial metrics.
            </Typography>
            <Typography variant="body1">
              The model is trained using XGBoost and includes explainable AI features such as LIME and SHAP explanations
              to help understand the decision-making process.
            </Typography>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Dashboard; 