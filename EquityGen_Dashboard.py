# EquityGen: Streamlit App for Demographic-Aware Precision Medicine

import streamlit as st
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

# Cache uploaded data
@st.cache_data

def load_data(clinical_file, expression_file, mutation_file):
    clinical_data = pd.read_csv(clinical_file)
    expression_data = pd.read_csv(expression_file)
    mutation_data = pd.read_csv(mutation_file)

    if 'patient_id' not in expression_data.columns:
        expression_data.insert(0, 'patient_id', range(1, len(expression_data) + 1))

    if 'patient_id' not in mutation_data.columns:
        raise ValueError("Mutation file must contain 'patient_id' column")

    long_expr = expression_data.melt(id_vars=['patient_id'], var_name='gene', value_name='gene_expression')

    # Compute mutation burden
    mutation_burden = mutation_data.groupby('patient_id').size().reset_index(name='mutation_burden')

    # Merge all data
    merged = pd.merge(clinical_data, mutation_burden, on="patient_id", how="left")
    merged['mutation_burden'] = merged['mutation_burden'].fillna(0)
    long_expr = pd.merge(long_expr, merged, on='patient_id', how='left')

    return long_expr

def plot_gene_expression_by_race(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='race', y='gene_expression', hue='gene',
                data=data[data['gene'].isin(GENES_OF_INTEREST)], ax=ax)
    ax.set_title('Gene Expression by Race')
    ax.set_xlabel('Race')
    ax.set_ylabel('Gene Expression')
    st.pyplot(fig)

def plot_survival_by_race(data):
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(10, 6))
    for race in data['race'].dropna().unique():
        race_data = data[data['race'] == race]
        kmf.fit(race_data['survival_time'], event_observed=race_data['survival_status'], label=race)
        kmf.plot(ax=ax)
    ax.set_title('Survival Curves by Race')
    ax.set_xlabel('Survival Time (Days)')
    ax.set_ylabel('Survival Probability')
    st.pyplot(fig)

def prepare_features(data):
    features = data[GENES_OF_INTEREST + ['race', 'mutation_burden']]
    features_encoded = pd.get_dummies(features, columns=['race'])
    target = data['survival_status']
    return features_encoded, target

def train_models(features_encoded, target):
    rf_model = RandomForestClassifier()
    rf_model.fit(features_encoded, target)
    rf_probs = rf_model.predict_proba(features_encoded)[:, 1]
    rf_accuracy = accuracy_score(target, rf_model.predict(features_encoded))
    rf_auc = roc_auc_score(target, rf_probs)

    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(features_encoded, target)
    lr_probs = lr_model.predict_proba(features_encoded)[:, 1]
    lr_accuracy = accuracy_score(target, lr_model.predict(features_encoded))
    lr_auc = roc_auc_score(target, lr_probs)

    st.write(f"🏏 **Random Forest** - Accuracy: `{rf_accuracy:.2f}`, AUC: `{rf_auc:.2f}`")
    st.write(f"📊 **Logistic Regression** - Accuracy: `{lr_accuracy:.2f}`, AUC: `{lr_auc:.2f}`")

    return rf_model, rf_probs

def assign_risk_groups(data, risk_scores):
    data = data.copy()
    data['risk_score'] = risk_scores
    data['risk_group'] = pd.cut(risk_scores, bins=[0, 0.33, 0.66, 1.0], labels=['Low', 'Medium', 'High'])
    return data

def plot_risk_distribution_by_race(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='race', hue='risk_group', data=data, ax=ax)
    ax.set_title('Distribution of Risk Categories by Race')
    ax.set_xlabel('Race')
    ax.set_ylabel('Number of Patients')
    st.pyplot(fig)

def plot_expression_by_risk_group(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='race', y='gene_expression', hue='risk_group',
                data=data[data['gene'].isin(GENES_OF_INTEREST)], ax=ax)
    ax.set_title('Gene Expression by Risk Group and Race')
    ax.set_xlabel('Race')
    ax.set_ylabel('Gene Expression')
    st.pyplot(fig)

def plot_mutation_burden_by_race(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='race', y='mutation_burden', data=data.drop_duplicates('patient_id'), ax=ax)
    ax.set_title('Mutation Burden by Race')
    ax.set_xlabel('Race')
    ax.set_ylabel('Mutation Burden')
    st.pyplot(fig)

def summarize_feature_importance(model, features_encoded):
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features_encoded.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    st.write("🔬 **Top Genes and Demographic Features Influencing Survival Prediction:**")
    st.dataframe(feature_importance_df.head(10))

def patient_summary(data):
    summary = data[['patient_id', 'race', 'mutation_burden', 'risk_score', 'risk_group']].drop_duplicates()
    st.write("📋 **Example Patient Risk Summary:**")
    st.dataframe(summary.head())

def print_clinical_explanation():
    st.markdown("""
    ### 🧜‍♀️ What this means for your health:
    - This tool uses your genetic expression, mutation burden, and racial background to estimate your health risk.
    - 'High Risk' means similarity to patients with shorter survival times — this may call for proactive screening or targeted care.
    - Genes and mutation burden most strongly influencing your prediction may be relevant for personalized treatments in the future.
    - This app aims to bridge gaps in care by considering racial disparities in biology and access.
    """)

def show_about_section():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About EquityGen")
    st.sidebar.info("EquityGen helps visualize how genetic and demographic factors impact cancer risk prediction using machine learning.")

def main():
    st.set_page_config(page_title="EquityGen Dashboard", layout="wide")
    show_about_section()
    st.title("EquityGen: Demographic-Aware Precision Medicine")

    st.markdown("### Upload Your Data Files")
    clinical_file = st.file_uploader("Step 1: Upload Clinical Data", type="csv", key="clinical")
    expression_file = st.file_uploader("Step 2: Upload Expression Data", type="csv", key="expression")
    mutation_file = st.file_uploader("Step 3: Upload Mutation Data", type="csv", key="mutation")

    if clinical_file is not None and expression_file is not None and mutation_file is not None:
        try:
            data = load_data(clinical_file, expression_file, mutation_file)
            st.success("✅ Data successfully uploaded!")
            plot_gene_expression_by_race(data)
            plot_survival_by_race(data)
            plot_mutation_burden_by_race(data)
            features_encoded, target = prepare_features(data)
            rf_model, rf_probs = train_models(features_encoded, target)
            data = assign_risk_groups(data, rf_probs)
            plot_risk_distribution_by_race(data)
            plot_expression_by_risk_group(data)
            summarize_feature_importance(rf_model, features_encoded)
            patient_summary(data)
            print_clinical_explanation()
        except Exception as e:
            st.error(f"❌ Error reading files: {e}")
    else:
        st.warning("👆 Please upload **all three** files: Clinical, Expression, and Mutation.")

if __name__ == "__main__":
    main()

