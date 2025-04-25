import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import os

class LoanDataPreprocessor:
    def __init__(self):
        self.numeric_features = []
        self.categorical_features = []
        self.preprocessor = None
        
    def load_data(self, file_path):
        """Load the loan dataset"""
        return pd.read_csv(file_path)
    
    def identify_features(self, df):
        """Identify numeric and categorical features"""
        self.numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target variable from numeric features if present
        if 'loan_approved' in self.numeric_features:
            self.numeric_features.remove('loan_approved')
    
    def create_preprocessing_pipeline(self):
        """Create preprocessing pipeline for numeric and categorical features"""
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
    
    def preprocess_data(self, df, is_training=True):
        """Preprocess the data using the pipeline"""
        if is_training:
            self.identify_features(df)
            self.create_preprocessing_pipeline()
            processed_data = self.preprocessor.fit_transform(df)
            # Save the preprocessor
            joblib.dump(self.preprocessor, 'data/preprocessor.joblib')
        else:
            processed_data = self.preprocessor.transform(df)
        
        return processed_data
    
    def save_processed_data(self, data, filename):
        """Save processed data to CSV"""
        data.to_csv(filename, index=False)

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = LoanDataPreprocessor()
    
    # Load data
    df = preprocessor.load_data('data/loan_data.csv')
    
    # Preprocess data
    processed_data = preprocessor.preprocess_data(df, is_training=True)
    
    # Save processed data
    preprocessor.save_processed_data(pd.DataFrame(processed_data), 'data/processed_data.csv')
    
    print("Data preprocessing completed successfully!")

if __name__ == "__main__":
    main() 