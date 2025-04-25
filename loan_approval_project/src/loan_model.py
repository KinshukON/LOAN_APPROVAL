import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Step 1: Load the dataset
print("Loading dataset...")
data = pd.read_csv('data/loan_data.csv')
print("\nFirst 5 rows of the dataset:")
print(data.head())

# Step 2: Inspect the dataset
print("\nDataset information:")
print(data.info())
print("\nDataset description:")
print(data.describe())

# Step 3: Handle missing values
print("\nHandling missing values...")
data.fillna(method='ffill', inplace=True)
print("Missing values after filling:")
print(data.isnull().sum())

# Step 4: Encode categorical variables
print("\nEncoding categorical variables...")
encoder = LabelEncoder()
categorical_cols = ['purpose']
for col in categorical_cols:
    if col in data.columns:
        data[col] = encoder.fit_transform(data[col])
        print(f"Encoded {col}")

# Step 5: Normalize numerical features
print("\nNormalizing numerical features...")
scaler = StandardScaler()
numerical_cols = ['int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 
                 'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 
                 'delinq.2yrs', 'pub.rec']
for col in numerical_cols:
    if col in data.columns:
        data[col] = scaler.fit_transform(data[[col]])
        print(f"Normalized {col}")

# Step 6: Save the processed dataset
print("\nSaving processed dataset...")
data.to_csv('data/processed_data.csv', index=False)
print("Processed data saved to 'data/processed_data.csv'")

# Step 7: Feature selection
print("\nSelecting features for model training...")
feature_cols = ['credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc', 
                'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 
                'inq.last.6mths', 'delinq.2yrs', 'pub.rec']
X = data[feature_cols]
y = data['not.fully.paid']

# Step 8: Train-test split
print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Step 9: Train the model
print("\nTraining Logistic Regression model...")
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
print("Model training completed")

# Step 10: Evaluate the model
print("\nEvaluating model performance...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Generate confusion matrix
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=['No', 'Yes'], 
            yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('reports/confusion_matrix.png')
plt.close()
print("Confusion matrix saved to 'reports/confusion_matrix.png'")

# Feature importance
print("\nCalculating feature importance...")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': np.abs(model.coef_[0])
}).sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)
feature_importance.to_csv('reports/feature_importance.csv', index=False)
print("Feature importance saved to 'reports/feature_importance.csv'")

print("\nLoan approval model training and evaluation completed successfully!") 