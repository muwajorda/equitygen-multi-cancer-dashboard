# Demographic and Race-Aware Precision Medicine Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from lifelines import KaplanMeierFitter

# Load data with demographic information
# Ensure dataset contains 'gene_expression', 'survival_time', 'survival_status', 'gene', 'race'
data = pd.read_csv("your_data.csv")

# Genes of interest - replace with actual genes
genes_of_interest = ['gene1', 'gene2', 'gene3', 'gene4']

# Boxplot to visualize gene expression differences by race
plt.figure(figsize=(10, 6))
sns.boxplot(x='race', y='gene_expression', hue='gene',
            data=data[data['gene'].isin(genes_of_interest)])
plt.title('Gene Expression by Race')
plt.xlabel('Race')
plt.ylabel('Gene Expression')
plt.legend(title='Gene')
plt.tight_layout()
plt.show()

# Survival analysis stratified by race
kmf = KaplanMeierFitter()
plt.figure(figsize=(10, 6))
for race in data['race'].unique():
    race_data = data[data['race'] == race]
    kmf.fit(race_data['survival_time'], event_observed=race_data['survival_status'], label=race)
    kmf.plot()

plt.title('Survival Curves by Race')
plt.xlabel('Survival Time (Days)')
plt.ylabel('Survival Probability')
plt.legend(title='Race')
plt.tight_layout()
plt.show()

# Feature preparation: genes and race encoding
features = data[genes_of_interest + ['race']]
target = data['survival_status']
features_encoded = pd.get_dummies(features, columns=['race'])

# Random Forest risk prediction
rf_model = RandomForestClassifier()
rf_model.fit(features_encoded, target)
rf_predictions = rf_model.predict(features_encoded)
rf_probs = rf_model.predict_proba(features_encoded)[:, 1]
rf_accuracy = accuracy_score(target, rf_predictions)
rf_auc = roc_auc_score(target, rf_probs)

print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print(f"Random Forest AUC: {rf_auc:.2f}")

# Logistic Regression for comparison
lr_model = LogisticRegression()
lr_model.fit(features_encoded, target)
lr_predictions = lr_model.predict(features_encoded)
lr_probs = lr_model.predict_proba(features_encoded)[:, 1]
lr_accuracy = accuracy_score(target, lr_predictions)
lr_auc = roc_auc_score(target, lr_probs)

print(f"Logistic Regression Accuracy: {lr_accuracy:.2f}")
print(f"Logistic Regression AUC: {lr_auc:.2f}")

# Assign risk scores and risk categories
risk_scores = rf_probs
risk_categories = pd.cut(risk_scores, bins=[0, 0.33, 0.66, 1.0], labels=['Low', 'Medium', 'High'])
data['risk_score'] = risk_scores
data['risk_group'] = risk_categories

# Count of risk groups by race
plt.figure(figsize=(10, 6))
sns.countplot(x='race', hue='risk_group', data=data)
plt.title('Patient Risk Categories by Race')
plt.xlabel('Race')
plt.ylabel('Number of Patients')
plt.legend(title='Risk Group')
plt.tight_layout()
plt.show()

# Gene expression across races and risk groups
plt.figure(figsize=(12, 6))
sns.boxplot(x='race', y='gene_expression', hue='risk_group',
            data=data[data['gene'].isin(genes_of_interest)])
plt.title('Gene Expression Across Race and Risk Groups')
plt.xlabel('Race')
plt.ylabel('Gene Expression')
plt.legend(title='Risk Group')
plt.tight_layout()
plt.show()

# Feature importance for treatment insights
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': features_encoded.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("Top Genes and Demographic Features Influencing Survival Prediction:")
print(feature_importance_df.head(10))

# Clinical relevance explanation (in lay terms)
print("\nInterpretation for Users:")
print("Your genetic information and race can influence how your body responds to certain diseases and treatments. This tool uses a model trained on real patient data to estimate your survival risk and identify the most important genes linked to that risk.")
print("If your score is 'High Risk', it means that your profile closely matches patients who had more severe outcomes. This doesn’t mean a poor outcome is guaranteed — it highlights the need for closer monitoring or targeted treatment.")
print("Genes listed in the 'Top Important Features' are the ones most associated with your outcome prediction. These may be used in future treatment decisions.")

# TODO: Link gene information to specific known treatments based on race
# This can be implemented using a lookup table of gene-drug associations filtered by population response data

