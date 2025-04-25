import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import optuna
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

class LoanModelTrainer:
    def __init__(self):
        self.model = None
        self.best_params = None
        
    def load_data(self, file_path):
        """Load the preprocessed data"""
        return pd.read_csv(file_path)
    
    def prepare_data(self, df):
        """Split data into features and target"""
        X = df.drop('loan_approved', axis=1)
        y = df['loan_approved']
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def objective(self, trial, X_train, y_train, X_val, y_val):
        """Optuna objective function for hyperparameter optimization"""
        param = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5)
        }
        
        model = xgb.XGBClassifier(**param, random_state=42)
        model.fit(X_train, y_train)
        return model.score(X_val, y_val)
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val):
        """Optimize hyperparameters using Optuna"""
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
                      n_trials=50)
        self.best_params = study.best_params
        return study.best_params
    
    def train_model(self, X_train, y_train):
        """Train the model with best parameters"""
        # Apply SMOTE for handling class imbalance
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        self.model = xgb.XGBClassifier(**self.best_params, random_state=42)
        self.model.fit(X_train_balanced, y_train_balanced)
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model and generate reports"""
        y_pred = self.model.predict(X_test)
        
        # Generate classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('reports/confusion_matrix.png')
        plt.close()
        
        # Save feature importance
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        feature_importance.to_csv('reports/feature_importance.csv', index=False)
    
    def save_model(self):
        """Save the trained model"""
        joblib.dump(self.model, 'data/model.joblib')

def main():
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Initialize trainer
    trainer = LoanModelTrainer()
    
    # Load and prepare data
    df = trainer.load_data('data/processed_data.csv')
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    
    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                     test_size=0.2, random_state=42)
    
    # Optimize hyperparameters
    print("Optimizing hyperparameters...")
    best_params = trainer.optimize_hyperparameters(X_train, y_train, X_val, y_val)
    print("Best parameters:", best_params)
    
    # Train model
    print("\nTraining model...")
    trainer.train_model(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    trainer.evaluate_model(X_test, y_test)
    
    # Save model
    trainer.save_model()
    print("\nModel training completed successfully!")

if __name__ == "__main__":
    main() 