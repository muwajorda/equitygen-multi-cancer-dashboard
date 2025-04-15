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

# Sidebar input: Select or enter cancer type
cancer_type = st.sidebar.text_input("Enter Cancer Type (e.g., BRCA, LUAD, COAD)", value="BRCA")
st.set_page_config(page_title=f"EquityGen Dashboard - {cancer_type}", layout="wide")
st.title(f"🧬 EquityGen: {cancer_type} Equity Dashboard")

# Upload data
st.sidebar.header("Upload Raw Files")
clinical_file = st.sidebar.file_uploader("Upload Clinical Data (.csv)", type=["csv"])
expression_file = st.sidebar.file_uploader("Upload Expression Data (.csv)", type=["csv"])

if clinical_file and expression_file:
    # Load files
    clinical_df = pd.read_csv(clinical_file)
    expression_df = pd.read_csv(expression_file)

    # Preprocessing expression data
    expression_df_grouped = expression_df.groupby(['Gene', 'Sample_ID'])['FPKM'].mean().reset_index()
    expression_pivot = expression_df_grouped.pivot(index='Gene', columns='Sample_ID', values='FPKM')

    # Get top 500 most variable genes
    gene_variance = expression_pivot.var(axis=1)
    top_genes = gene_variance.sort_values(ascending=False).head(500).index
    top_expression_df = expression_pivot.loc[top_genes].T.reset_index()
    top_expression_df.rename(columns={'index': 'Sample_ID'}, inplace=True)
    top_expression_df['Patient_ID'] = top_expression_df['Sample_ID'].str[:12]

    # Merge with clinical data
    merged_df = pd.merge(clinical_df, top_expression_df, on='Patient_ID', how='inner')
    df = merged_df.dropna(subset=['Race', 'Survival_Time', 'Event'])
    st.success("✅ Files loaded and merged successfully")

    # Sidebar filters
    st.sidebar.header("Filters")
    gene = st.sidebar.selectbox("Select a Gene", options=df.columns[12:])
    races = st.sidebar.multiselect("Select Race Groups", options=df["Race"].dropna().unique(), default=list(df["Race"].dropna().unique()))

    filtered_df = df[df["Race"].isin(races)]

    # Gene Expression by Race
    st.subheader("📊 Gene Expression by Race")
    fig = px.box(filtered_df, x="Race", y=gene, color="Race",
                 title=f"Expression of {gene} Across Racial Groups",
                 points="all")
    st.plotly_chart(fig, use_container_width=True)

    # Differential Expression Table (Mock)
    st.subheader("📈 Top Genes by Variance (Mock Table)")
    variance_df = df.iloc[:, 12:].var().sort_values(ascending=False).reset_index()
    variance_df.columns = ["Gene", "Variance"]
    st.dataframe(variance_df.head(20))

    # Survival Analysis
    st.subheader("⏳ Kaplan-Meier Survival Curve")
    kmf = KaplanMeierFitter()
    fig_surv = px.line()

    for race in races:
        race_df = filtered_df[filtered_df["Race"] == race]
        kmf.fit(durations=race_df["Survival_Time"], event_observed=race_df["Event"], label=race)
        kmf_survival = kmf.survival_function_.reset_index()
        kmf_survival.columns = ["Timeline", "Survival Probability"]
        fig_surv.add_scatter(x=kmf_survival["Timeline"], y=kmf_survival["Survival Probability"], mode='lines', name=race)

    fig_surv.update_layout(title="Survival Curve by Race",
                           xaxis_title="Time (days)", yaxis_title="Survival Probability")
    st.plotly_chart(fig_surv, use_container_width=True)

    # ML Risk Prediction
    st.subheader("🧠 Risk Prediction Model")
    st.markdown("Predicting mortality risk (Event = 1) using gene expression and age")

    # Prepare data
    ml_df = df.dropna(subset=['Age'])
    X = ml_df.iloc[:, 12:]  # Gene expression
    X['Age'] = ml_df['Age']  # Add Age
    y = ml_df['Event']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

    st.write(f"**Random Forest Accuracy:** {rf_acc:.2f}")
    st.write(f"**Random Forest AUC:** {rf_auc:.2f}")

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])

    st.write(f"**Logistic Regression Accuracy:** {lr_acc:.2f}")
    st.write(f"**Logistic Regression AUC:** {lr_auc:.2f}")

    # File download
    st.download_button("Download Filtered Data", data=filtered_df.to_csv(index=False), file_name=f"filtered_equitygen_data_{cancer_type}.csv")

else:
    st.info("📂 Please upload your clinical and expression datasets to get started.")

