import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lime
import lime.lime_tabular
import shap
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Create necessary directories
os.makedirs('reports/lime_explanations', exist_ok=True)
os.makedirs('reports/shap_explanations', exist_ok=True)

# Load the processed data
print("Loading processed data...")
data = pd.read_csv('data/processed_data.csv')

# Prepare features and target
feature_cols = ['credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc', 
                'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 
                'inq.last.6mths', 'delinq.2yrs', 'pub.rec']
X = data[feature_cols]
y = data['not.fully.paid']

# Train a model (or load if already trained)
try:
    model = joblib.load('data/model.joblib')
    print("Loaded existing model")
except:
    print("Training new model...")
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    joblib.dump(model, 'data/model.joblib')
    print("Model trained and saved")

# Generate LIME explanations
print("\nGenerating LIME explanations...")
explainer = lime.lime_tabular.LimeTabularExplainer(
    X.values,
    feature_names=feature_cols,
    class_names=['No', 'Yes'],
    mode='classification'
)

# Generate explanations for 5 sample instances
for i in range(min(5, len(X))):
    exp = explainer.explain_instance(
        X.iloc[i].values,
        model.predict_proba
    )
    
    # Save explanation plot
    plt.figure(figsize=(10, 6))
    exp.as_pyplot_figure()
    plt.title(f'LIME Explanation for Instance {i+1}')
    plt.tight_layout()
    plt.savefig(f'reports/lime_explanations/lime_explanation_{i+1}.png')
    plt.close()
    print(f"Generated LIME explanation for instance {i+1}")

# Generate SHAP explanations
print("\nGenerating SHAP explanations...")

# Convert categorical variables to numeric
X_shap = X.copy()
for col in X_shap.select_dtypes(['object']).columns:
    X_shap[col] = pd.Categorical(X_shap[col]).codes

# For logistic regression, we'll use LinearExplainer instead of KernelExplainer
explainer = shap.LinearExplainer(model, X_shap)
shap_values = explainer.shap_values(X_shap.iloc[:100])

# Summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_shap.iloc[:100], feature_names=feature_cols, show=False)
plt.title('SHAP Summary Plot')
plt.tight_layout()
plt.savefig('reports/shap_explanations/shap_summary.png')
plt.close()
print("Generated SHAP summary plot")

# Feature importance plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_shap.iloc[:100], feature_names=feature_cols, 
                 plot_type='bar', show=False)
plt.title('SHAP Feature Importance')
plt.tight_layout()
plt.savefig('reports/shap_explanations/shap_importance.png')
plt.close()
print("Generated SHAP feature importance plot")

# Dependence plots for each feature
for i, feature in enumerate(feature_cols):
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        i,
        shap_values,
        X_shap.iloc[:100],
        feature_names=feature_cols,
        show=False
    )
    plt.title(f'SHAP Dependence Plot for {feature}')
    plt.tight_layout()
    plt.savefig(f'reports/shap_explanations/shap_dependence_{feature}.png')
    plt.close()
    print(f"Generated SHAP dependence plot for {feature}")

print("\nModel explanation completed successfully!")
print("LIME explanations saved to 'reports/lime_explanations/'")
print("SHAP explanations saved to 'reports/shap_explanations/'") 