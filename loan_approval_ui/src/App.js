import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Box, Container } from '@mui/material';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import ModelTraining from './pages/ModelTraining';
import Results from './pages/Results';

function App() {
  return (
    <Router>
      <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        <Navbar />
        <Container component="main" sx={{ mt: 4, mb: 4, flex: 1 }}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/train" element={<ModelTraining />} />
            <Route path="/results" element={<Results />} />
          </Routes>
        </Container>
      </Box>
    </Router>
  );
}

export default App; 