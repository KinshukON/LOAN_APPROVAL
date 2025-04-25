# Loan Approval Prediction System

A comprehensive machine learning system for predicting loan approval outcomes using advanced ML techniques and explainable AI.

## Overview

This project implements a machine learning pipeline for loan approval prediction. It uses various features such as credit score, income, debt-to-income ratio, and other financial metrics to predict whether a loan application should be approved or not.

This project is part of a comprehensive blog series on Explainable AI (XAI) at [Data-Nizant](https://datanizant.com/unlocking-ai-transparency-a-practical-guide-to-getting-started-with-explainable-ai-xai/). The blog series provides detailed insights into implementing XAI techniques and understanding AI model interpretability.

## Features

- Data preprocessing and feature engineering
- Advanced machine learning model training with XGBoost
- Model hyperparameter optimization using Optuna
- Explainable AI using LIME and SHAP
- Comprehensive evaluation metrics and visualizations
- REST API for model inference
- Modern web interface for predictions

## Project Structure

```
loan_approval_project/
├── data/               # Dataset directory
├── notebooks/         # Jupyter notebooks for analysis
├── reports/          # Generated reports and visualizations
├── src/              # Source code
│   ├── features/     # Feature engineering
│   ├── models/       # Model training and evaluation
│   └── visualization/# Data visualization
├── predict.py        # Prediction script
├── run_pipeline.py   # Main pipeline execution
└── requirements.txt  # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/KinshukON/LOAN_APPROVAL.git
cd LOAN_APPROVAL
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Pipeline

To run the complete ML pipeline:
```bash
python run_pipeline.py
```

### Making Predictions

To make predictions using the trained model:
```bash
python predict.py
```

## Model Features

The model uses the following features for prediction:
- Credit Policy
- Purpose
- Interest Rate
- Installment
- Log Annual Income
- Debt-to-Income Ratio (DTI)
- FICO Score
- Days with Credit Line
- Revolving Balance
- Revolving Utilization
- Inquiries in Last 6 Months
- Delinquencies in Last 2 Years
- Public Records
- Not Fully Paid (Target Variable)

## Dependencies

- numpy>=1.26.0
- pandas>=2.2.0
- scikit-learn>=1.4.0
- matplotlib>=3.8.0
- seaborn>=0.13.0
- lime>=0.2.0.1
- shap>=0.44.0
- jupyter>=1.0.0
- xgboost>=2.0.0
- imbalanced-learn>=0.11.0
- optuna>=3.5.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support and Contact

If you have any questions or face any issues while using this project, please feel free to reach out through:
- Blog: [Data-Nizant](https://datanizant.com)
- Author's Contact: Available through the blog's contact form

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Kinshuk Dutta

## Acknowledgments

- This project is part of the Explainable AI blog series at [Data-Nizant](https://datanizant.com/unlocking-ai-transparency-a-practical-guide-to-getting-started-with-explainable-ai-xai/)
- The dataset used in this project is from [source]
- Special thanks to all contributors and maintainers of the libraries used in this project 