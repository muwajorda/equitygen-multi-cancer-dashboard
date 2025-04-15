import streamlit as st
import pandas as pd
import plotly.express as px
from lifelines import KaplanMeierFitter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page config first
st.set_page_config(page_title="EquityGen Multi-Cancer Dashboard", layout="wide")
st.title("🧬 EquityGen: Multi-Cancer Equity Dashboard")

# Upload data
st.sidebar.header("Upload Raw Files")
clinical_file = st.sidebar.file_uploader("Upload Clinical Data (.csv)", type=["csv"])
expression_file = st.sidebar.file_uploader("Upload Expression Data (.csv)", type=["csv"])

# Optional toggle to compare with second dataset
compare_toggle = st.sidebar.checkbox("Compare with a second indication?")
second_clinical_file = None
second_expression_file = None
if compare_toggle:
    st.sidebar.markdown("### Secondary Dataset")
    second_clinical_file = st.sidebar.file_uploader("Upload 2nd Clinical Data", type=["csv"], key="clinical2")
    second_expression_file = st.sidebar.file_uploader("Upload 2nd Expression Data", type=["csv"], key="expression2")

# Load primary dataset
if clinical_file and expression_file:
    # Detect indication from filename
    cancer_type = os.path.basename(clinical_file.name).split("_")[1] if "_" in clinical_file.name else "Unknown"

    # Load files
    clinical_df = pd.read_csv(clinical_file)
    expression_df = pd.read_csv(expression_file)

    # Preprocess expression data
    expression_df_grouped = expression_df.groupby(['Gene', 'Sample_ID'])['FPKM'].mean().reset_index()
    expression_pivot = expression_df_grouped.pivot(index='Gene', columns='Sample_ID', values='FPKM')
    gene_variance = expression_pivot.var(axis=1)
    top_genes = gene_variance.sort_values(ascending=False).head(500).index
    top_expression_df = expression_pivot.loc[top_genes].T.reset_index()
    top_expression_df.rename(columns={'index': 'Sample_ID'}, inplace=True)
    top_expression_df['Patient_ID'] = top_expression_df['Sample_ID'].str[:12]

    # Merge with clinical
    merged_df = pd.merge(clinical_df, top_expression_df, on='Patient_ID', how='inner')
    df = merged_df.dropna(subset=['Race', 'Survival_Time', 'Event'])

    st.success(f"✅ {cancer_type} files loaded successfully")
    st.header(f"📊 Exploring {cancer_type} Dataset")

    # Sidebar filters
    st.sidebar.header("Filters")
    gene = st.sidebar.selectbox("Select a Gene", options=df.columns[12:])
    races = st.sidebar.multiselect("Select Race Groups", options=df["Race"].dropna().unique(), default=list(df["Race"].dropna().unique()))
    filtered_df = df[df["Race"].isin(races)]

    st.subheader("📊 Gene Expression by Race")
    fig = px.box(filtered_df, x="Race", y=gene, color="Race", title=f"Expression of {gene} in {cancer_type}", points="all")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Top Genes by Variance")
    variance_df = df.iloc[:, 12:].var().sort_values(ascending=False).reset_index()
    variance_df.columns = ["Gene", "Variance"]
    st.dataframe(variance_df.head(20))

    st.subheader("⏳ Kaplan-Meier Survival Curve")
    kmf = KaplanMeierFitter()
    fig_surv = px.line()
    for race in races:
        race_df = filtered_df[filtered_df["Race"] == race]
        kmf.fit(race_df["Survival_Time"], event_observed=race_df["Event"], label=race)
        kmf_df = kmf.survival_function_.reset_index()
        kmf_df.columns = ["Timeline", "Survival Probability"]
        fig_surv.add_scatter(x=kmf_df["Timeline"], y=kmf_df["Survival Probability"], mode='lines', name=race)
    fig_surv.update_layout(title=f"Survival by Race - {cancer_type}", xaxis_title="Time (days)", yaxis_title="Survival Probability")
    st.plotly_chart(fig_surv, use_container_width=True)

    st.subheader("🧠 Risk Prediction (ML Model)")
    ml_df = df.dropna(subset=['Age'])
    X = ml_df.iloc[:, 12:]
    X['Age'] = ml_df['Age']
    y = ml_df['Event']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    st.write(f"**Random Forest Accuracy:** {rf_acc:.2f}")
    st.write(f"**Random Forest AUC:** {rf_auc:.2f}")

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test))
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    st.write(f"**Logistic Regression Accuracy:** {lr_acc:.2f}")
    st.write(f"**Logistic Regression AUC:** {lr_auc:.2f}")

    st.download_button("Download Filtered Data", data=filtered_df.to_csv(index=False), file_name=f"filtered_equitygen_data_{cancer_type}.csv")

# Optional: second dataset processing block could go here in future
if compare_toggle and second_clinical_file and second_expression_file:
    st.warning("🧪 Compare feature is coming next — currently supports only one dataset loaded at a time.")

else:
    st.info("📂 Please upload your clinical and expression datasets to get started.")

