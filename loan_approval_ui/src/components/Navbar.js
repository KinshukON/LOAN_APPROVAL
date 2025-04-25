import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
} from '@mui/material';

function Navbar() {
  return (
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          Loan Approval Model
        </Typography>
        <Box>
          <Button
            color="inherit"
            component={RouterLink}
            to="/"
          >
            Dashboard
          </Button>
          <Button
            color="inherit"
            component={RouterLink}
            to="/train"
          >
            Train Model
          </Button>
          <Button
            color="inherit"
            component={RouterLink}
            to="/results"
          >
            Results
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  );
}

export default Navbar; 