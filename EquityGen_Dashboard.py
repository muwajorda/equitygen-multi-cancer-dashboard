# Demographic and Race-Aware Precision Medicine Analysis (Modular Version)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from lifelines import KaplanMeierFitter

# Global variable
GENES_OF_INTEREST = ['gene1', 'gene2', 'gene3', 'gene4']

# Load Data
def load_data(path):
    return pd.read_csv(path)

# Visualization of Gene Expression by Race
def plot_gene_expression_by_race(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='race', y='gene_expression', hue='gene',
                data=data[data['gene'].isin(GENES_OF_INTEREST)])
    plt.title('Gene Expression by Race')
    plt.xlabel('Race')
    plt.ylabel('Gene Expression')
    plt.legend(title='Gene')
    plt.tight_layout()
    plt.show()

# Kaplan-Meier Survival Curves by Race
def plot_survival_by_race(data):
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

# Feature Preparation
def prepare_features(data):
    features = data[GENES_OF_INTEREST + ['race']]
    features_encoded = pd.get_dummies(features, columns=['race'])
    target = data['survival_status']
    return features_encoded, target

# Model Training and Evaluation
def train_models(features_encoded, target):
    rf_model = RandomForestClassifier()
    rf_model.fit(features_encoded, target)
    rf_probs = rf_model.predict_proba(features_encoded)[:, 1]
    rf_accuracy = accuracy_score(target, rf_model.predict(features_encoded))
    rf_auc = roc_auc_score(target, rf_probs)

    lr_model = LogisticRegression()
    lr_model.fit(features_encoded, target)
    lr_probs = lr_model.predict_proba(features_encoded)[:, 1]
    lr_accuracy = accuracy_score(target, lr_model.predict(features_encoded))
    lr_auc = roc_auc_score(target, lr_probs)

    print(f"🎯 Random Forest Accuracy: {rf_accuracy:.2f}, AUC: {rf_auc:.2f}")
    print(f"📊 Logistic Regression Accuracy: {lr_accuracy:.2f}, AUC: {lr_auc:.2f}")

    return rf_model, rf_probs

# Assign Risk Scores and Risk Groups
def assign_risk_groups(data, risk_scores):
    data['risk_score'] = risk_scores
    data['risk_group'] = pd.cut(risk_scores, bins=[0, 0.33, 0.66, 1.0], labels=['Low', 'Medium', 'High'])
    return data

# Plot Risk Categories by Race
def plot_risk_distribution_by_race(data):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='race', hue='risk_group', data=data)
    plt.title('Distribution of Risk Categories by Race')
    plt.xlabel('Race')
    plt.ylabel('Number of Patients')
    plt.legend(title='Risk Group')
    plt.tight_layout()
    plt.show()

# Plot Gene Expression by Risk Group and Race
def plot_expression_by_risk_group(data):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='race', y='gene_expression', hue='risk_group',
                data=data[data['gene'].isin(GENES_OF_INTEREST)])
    plt.title('Gene Expression by Risk Group and Race')
    plt.xlabel('Race')
    plt.ylabel('Gene Expression')
    plt.legend(title='Risk Group')
    plt.tight_layout()
    plt.show()

# Feature Importance Summary
def summarize_feature_importance(model, features_encoded):
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features_encoded.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    print("🔬 Top Genes and Demographic Features Influencing Survival Prediction:")
    print(feature_importance_df.head(10))

# Patient-Level Summary Output
def patient_summary(data):
    summary = data[['patient_id', 'race', 'risk_score', 'risk_group']].drop_duplicates()
    print("\n📋 Example Patient Risk Summary:")
    print(summary.head())

# Clinical Explanation Output
def print_clinical_explanation():
    print("\n🧬 What this means for your health:")
    print("- This tool uses your genetic expression and racial background to estimate your health risk.")
    print("- 'High Risk' means similarity to patients with shorter survival times — this may call for proactive screening or targeted care.")
    print("- Genes most strongly influencing your prediction may be relevant for personalized treatments in the future.")
    print("- This app aims to bridge gaps in care by considering racial disparities in biology and access.")

# Main Function
def main():
    data = load_data("your_data.csv")
    plot_gene_expression_by_race(data)
    plot_survival_by_race(data)
    features_encoded, target = prepare_features(data)
    rf_model, rf_probs = train_models(features_encoded, target)
    data = assign_risk_groups(data, rf_probs)
    plot_risk_distribution_by_race(data)
    plot_expression_by_risk_group(data)
    summarize_feature_importance(rf_model, features_encoded)
    patient_summary(data)
    print_clinical_explanation()

if __name__ == "__main__":
    main()

