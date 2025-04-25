import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Create necessary directories
os.makedirs('reports/feature_analysis', exist_ok=True)

# Load the model and data
print("Loading model and data...")
model = joblib.load('data/model.joblib')
data = pd.read_csv('data/processed_data.csv')

# Get feature names
feature_cols = ['credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc', 
                'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 
                'inq.last.6mths', 'delinq.2yrs', 'pub.rec']

# Prepare data for statistical analysis
X = data[feature_cols]
y = data['not.fully.paid']

# Standardize features for proper coefficient interpretation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

# Fit model with standardized features
model_scaled = LogisticRegression(random_state=42)
model_scaled.fit(X_scaled, y)

# Extract coefficients and calculate standard errors
coefficients = model_scaled.coef_[0]
n = len(X)
p = len(feature_cols)
X_with_intercept = np.column_stack([np.ones(n), X_scaled])
var_covar_matrix = np.linalg.inv(np.dot(X_with_intercept.T, X_with_intercept))
standard_errors = np.sqrt(np.diagonal(var_covar_matrix)[1:])

# Calculate z-scores and p-values
z_scores = coefficients / standard_errors
p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

# Calculate confidence intervals
confidence_level = 0.95
z_score = stats.norm.ppf((1 + confidence_level) / 2)
ci_lower = coefficients - z_score * standard_errors
ci_upper = coefficients + z_score * standard_errors

# Calculate odds ratios and their confidence intervals
odds_ratios = np.exp(coefficients)
odds_ratio_ci_lower = np.exp(ci_lower)
odds_ratio_ci_upper = np.exp(ci_upper)

# Create feature importance DataFrame
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': coefficients,
    'Std_Error': standard_errors,
    'Z_Score': z_scores,
    'P_Value': p_values,
    'CI_Lower': ci_lower,
    'CI_Upper': ci_upper,
    'Odds_Ratio': odds_ratios,
    'OR_CI_Lower': odds_ratio_ci_lower,
    'OR_CI_Upper': odds_ratio_ci_upper
})

# Sort by absolute coefficient value
feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
feature_importance = feature_importance.drop('Abs_Coefficient', axis=1)

# Print feature importance table
print("\nFeature Contribution Table:")
print("=" * 100)
print(f"{'Feature':<20} {'Coefficient':>10} {'Std Error':>10} {'Z-Score':>10} {'P-Value':>10} {'Odds Ratio':>12} {'95% CI':>25}")
print("-" * 100)
for _, row in feature_importance.iterrows():
    print(f"{row['Feature']:<20} {row['Coefficient']:>10.4f} {row['Std_Error']:>10.4f} {row['Z_Score']:>10.4f} "
          f"{row['P_Value']:>10.4f} {row['Odds_Ratio']:>12.4f} [{row['OR_CI_Lower']:.4f}, {row['OR_CI_Upper']:.4f}]")
print("=" * 100)

# Save to CSV
feature_importance.to_csv('reports/feature_analysis/feature_importance_detailed.csv', index=False)
print("\nDetailed feature importance saved to 'reports/feature_analysis/feature_importance_detailed.csv'")

# Visualize coefficients with confidence intervals
plt.figure(figsize=(12, 8))
y_pos = np.arange(len(feature_cols))
plt.errorbar(feature_importance['Coefficient'], y_pos, 
             xerr=[feature_importance['Coefficient'] - feature_importance['CI_Lower'],
                   feature_importance['CI_Upper'] - feature_importance['Coefficient']],
             fmt='o', capsize=5)
plt.yticks(y_pos, feature_importance['Feature'])
plt.axvline(x=0, color='black', linestyle='--', alpha=0.3)
plt.title('Feature Coefficients with 95% Confidence Intervals')
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.savefig('reports/feature_analysis/coefficients_with_ci.png')
plt.close()
print("Coefficients with confidence intervals plot saved to 'reports/feature_analysis/coefficients_with_ci.png'")

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Plot 1: Coefficients
y_pos = np.arange(len(feature_cols))
ax1.barh(y_pos, feature_importance['Coefficient'], 
         xerr=[feature_importance['Coefficient'] - feature_importance['CI_Lower'],
               feature_importance['CI_Upper'] - feature_importance['Coefficient']],
         color='skyblue', capsize=5)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(feature_importance['Feature'])
ax1.axvline(x=0, color='black', linestyle='--', alpha=0.3)
ax1.set_title('Feature Coefficients with 95% CI')
ax1.set_xlabel('Coefficient Value')

# Add p-value annotations
for i, p_val in enumerate(feature_importance['P_Value']):
    if p_val < 0.001:
        sig = '***'
    elif p_val < 0.01:
        sig = '**'
    elif p_val < 0.05:
        sig = '*'
    else:
        sig = ''
    ax1.text(feature_importance['Coefficient'].iloc[i], i, sig, 
             ha='left', va='center')

# Plot 2: Odds Ratios
ax2.barh(y_pos, feature_importance['Odds_Ratio'], 
         xerr=[feature_importance['Odds_Ratio'] - feature_importance['OR_CI_Lower'],
               feature_importance['OR_CI_Upper'] - feature_importance['Odds_Ratio']],
         color='lightgreen', capsize=5)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(feature_importance['Feature'])
ax2.axvline(x=1, color='black', linestyle='--', alpha=0.3)
ax2.set_title('Feature Odds Ratios with 95% CI')
ax2.set_xlabel('Odds Ratio')

# Add odds ratio annotations
for i, (odds, ci_lower, ci_upper) in enumerate(zip(feature_importance['Odds_Ratio'],
                                                  feature_importance['OR_CI_Lower'],
                                                  feature_importance['OR_CI_Upper'])):
    ax2.text(odds, i, f'{odds:.3f}\n[{ci_lower:.3f}, {ci_upper:.3f}]', 
             ha='left', va='center')

plt.tight_layout()
plt.savefig('reports/feature_analysis/feature_importance_comparison.png')
plt.close()
print("Feature importance comparison plot saved to 'reports/feature_analysis/feature_importance_comparison.png'")

# Create a heatmap of feature correlations
plt.figure(figsize=(12, 10))
correlation_matrix = data[feature_cols + ['not.fully.paid']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('reports/feature_analysis/correlation_heatmap.png')
plt.close()
print("Correlation heatmap saved to 'reports/feature_analysis/correlation_heatmap.png'")

# Create a summary plot of feature importance
plt.figure(figsize=(12, 8))
importance_data = pd.DataFrame({
    'Feature': feature_importance['Feature'],
    'Abs_Coefficient': np.abs(feature_importance['Coefficient']),
    'Direction': np.sign(feature_importance['Coefficient'])
})

# Sort by absolute coefficient value
importance_data = importance_data.sort_values('Abs_Coefficient', ascending=True)

# Create horizontal bar plot
bars = plt.barh(importance_data['Feature'], importance_data['Abs_Coefficient'],
                color=['red' if d < 0 else 'green' for d in importance_data['Direction']])

# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width, i, f'{width:.3f}', ha='left', va='center')

plt.title('Feature Importance Summary')
plt.xlabel('Absolute Coefficient Value')
plt.tight_layout()
plt.savefig('reports/feature_analysis/feature_importance_summary.png')
plt.close()
print("Feature importance summary plot saved to 'reports/feature_analysis/feature_importance_summary.png'")

# Create a bar chart showing odds ratios for all features
plt.figure(figsize=(12, 10))
y_pos = np.arange(len(feature_importance))

# Sort by odds ratio for better visualization
feature_importance_sorted = feature_importance.sort_values('Odds_Ratio', ascending=True)

# Create horizontal bar chart
bars = plt.barh(y_pos, feature_importance_sorted['Odds_Ratio'], 
                color=['#3498db' if i < 3 else '#95a5a6' for i in range(len(feature_importance_sorted))])

# Add error bars for confidence intervals
plt.errorbar(feature_importance_sorted['Odds_Ratio'], y_pos,
             xerr=[feature_importance_sorted['Odds_Ratio'] - feature_importance_sorted['OR_CI_Lower'],
                   feature_importance_sorted['OR_CI_Upper'] - feature_importance_sorted['Odds_Ratio']],
             fmt='none', color='black', capsize=5)

# Add a vertical line at odds ratio = 1 (no effect)
plt.axvline(x=1, color='red', linestyle='--', alpha=0.7)

# Add value labels
for i, (odds, ci_lower, ci_upper) in enumerate(zip(feature_importance_sorted['Odds_Ratio'],
                                                  feature_importance_sorted['OR_CI_Lower'],
                                                  feature_importance_sorted['OR_CI_Upper'])):
    plt.text(odds, i, f'{odds:.3f}', ha='left', va='center')

# Highlight the most important features
for i, feature in enumerate(feature_importance_sorted['Feature']):
    if feature in ['fico', 'log.annual.inc', 'installment', 'inq.last.6mths']:
        plt.text(0, i, '★', color='gold', fontsize=16, ha='right', va='center')

plt.yticks(y_pos, feature_importance_sorted['Feature'])
plt.xlabel('Odds Ratio')
plt.title('Feature Importance: Odds Ratios with 95% Confidence Intervals')
plt.tight_layout()
plt.savefig('reports/feature_analysis/odds_ratios_bar_chart.png')
plt.close()
print("Odds ratios bar chart saved to 'reports/feature_analysis/odds_ratios_bar_chart.png'")

# Print interpretation
print("\nFeature Importance Interpretation:")
for _, row in feature_importance.iterrows():
    feature = row['Feature']
    coef = row['Coefficient']
    odds = row['Odds_Ratio']
    p_val = row['P_Value']
    ci_lower = row['OR_CI_Lower']
    ci_upper = row['OR_CI_Upper']
    
    if coef > 0:
        direction = "increases"
        impact = "positive"
    else:
        direction = "decreases"
        impact = "negative"
    
    significance = "highly significant" if p_val < 0.001 else \
                  "significant" if p_val < 0.05 else \
                  "not significant"
    
    print(f"\n{feature}:")
    print(f"- Coefficient: {coef:.4f} (p-value: {p_val:.4f})")
    print(f"- Odds Ratio: {odds:.4f} [95% CI: {ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"- Statistical Significance: {significance}")
    print(f"- Interpretation: A one standard deviation increase in {feature} {direction} the odds of loan default by a factor of {odds:.4f}")
    print(f"- Impact: {impact} impact on loan approval")

print("\nFeature analysis completed successfully!") 