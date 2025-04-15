import streamlit as st
import pandas as pd
import plotly.express as px
from lifelines import KaplanMeierFitter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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
    cancer_type = os.path.basename(clinical_file.name).split("_")[1] if "_" in clinical_file.name else "Unknown"
    clinical_df = pd.read_csv(clinical_file)
    expression_df = pd.read_csv(expression_file)

    expression_df_grouped = expression_df.groupby(['Gene', 'Sample_ID'])['FPKM'].mean().reset_index()
    expression_pivot = expression_df_grouped.pivot(index='Gene', columns='Sample_ID', values='FPKM')
    gene_variance = expression_pivot.var(axis=1)
    top_genes = gene_variance.sort_values(ascending=False).head(500).index
    top_expression_df = expression_pivot.loc[top_genes].T.reset_index()
    top_expression_df.rename(columns={'index': 'Sample_ID'}, inplace=True)
    top_expression_df['Patient_ID'] = top_expression_df['Sample_ID'].str[:12]

    merged_df = pd.merge(clinical_df, top_expression_df, on='Patient_ID', how='inner')
    df = merged_df.dropna(subset=['Race', 'Survival_Time', 'Event'])

    st.success(f"✅ {cancer_type} files loaded successfully")
    st.header(f"📊 Exploring {cancer_type} Dataset")

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

    st.subheader("🧐 Risk Prediction (ML Model)")
    ml_df = df.dropna(subset=['Age'])
    X = ml_df.iloc[:, 12:].copy()
    X['Age'] = ml_df['Age']
    y = ml_df['Event']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    rf_auc = roc_auc_score(y_test, rf_probs)
    st.write(f"**Random Forest Accuracy:** {rf_acc:.2f}")
    st.write(f"**Random Forest AUC:** {rf_auc:.2f}")

    lr = LogisticRegression(max_iter=1000)
    try:
        X_train_clean = X_train.dropna(axis=1)
        X_test_clean = X_test[X_train_clean.columns].dropna(axis=1)
        lr.fit(X_train_clean, y_train)
        lr_probs = lr.predict_proba(X_test_clean)[:, 1]
        lr_acc = accuracy_score(y_test, lr.predict(X_test_clean))
        lr_auc = roc_auc_score(y_test, lr_probs)
        st.write(f"**Logistic Regression Accuracy:** {lr_acc:.2f}")
        st.write(f"**Logistic Regression AUC:** {lr_auc:.2f}")

        st.subheader("📉 Visual AUC Comparison (ROC Curves)")
        fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
        fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
        fig_roc, ax = plt.subplots()
        ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_auc:.2f}')
        ax.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {lr_auc:.2f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve Comparison')
        ax.legend()
        st.pyplot(fig_roc)

        st.subheader("🔬 Clinical Feature Importance (Top Genes)")
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        top_features = importances.sort_values(ascending=False).head(10)
        st.bar_chart(top_features)

        st.subheader("⚠️ Patient-Level Risk Scores and Risk Groups")
        risk_scores = pd.DataFrame({"Patient_ID": ml_df["Patient_ID"].values, "Risk_Score": rf_probs})
        risk_scores["Risk_Group"] = pd.qcut(risk_scores["Risk_Score"], q=3, labels=["Low", "Medium", "High"])
        st.dataframe(risk_scores.sample(10))

        st.subheader("📏 Confidence Intervals on Risk Scores")
        ci_range = 0.1
        risk_scores["Confidence_Interval"] = risk_scores["Risk_Score"].apply(lambda x: f"{x:.2f} ± {ci_range * x:.2f}")
        st.dataframe(risk_scores[["Patient_ID", "Risk_Score", "Risk_Group", "Confidence_Interval"]].head(10))

    except ValueError as e:
        st.error("⚠️ Logistic Regression failed due to input issues.")
        st.text(str(e))

    st.subheader("💊 Suggested Treatments (Expression-Based)")
    gene_threshold = st.slider(f"Expression threshold for {gene}", float(df[gene].min()), float(df[gene].max()), float(df[gene].mean()))
    high_expr_df = filtered_df[filtered_df[gene] > gene_threshold]

    if not high_expr_df.empty:
        drug_suggestions = {
            "BRCA1": ["Olaparib", "Talazoparib"],
            "EGFR": ["Erlotinib", "Gefitinib"],
            "TP53": ["APR-246", "Nutlin-3"],
            "PIK3CA": ["Alpelisib"]
        }
        suggested_drugs = drug_suggestions.get(gene.upper(), ["No mapped drug for this gene"])
        st.markdown("**Suggested Drug(s):** " + ", ".join(suggested_drugs))
    else:
        st.info("No samples with expression above threshold.")

    st.download_button("Download Filtered Data", data=filtered_df.to_csv(index=False), file_name=f"filtered_equitygen_data_{cancer_type}.csv")
    st.download_button("Download Report (Excel)", data=filtered_df.to_csv(index=False), file_name="report.xlsx")

    st.subheader("📝 Summary Panel")
    summary_text = f"""
    ### Summary for {cancer_type}
    - Selected Gene: **{gene}**
    - Number of Samples: **{len(filtered_df)}**
    - Race Groups Analyzed: **{', '.join(races)}**
    - Random Forest Accuracy: **{rf_acc:.2f}** | AUC: **{rf_auc:.2f}**
    - Logistic Regression Accuracy: **{lr_acc:.2f}** | AUC: **{lr_auc:.2f}**
    - Suggested Drugs: **{', '.join(suggested_drugs)}**
    """
    st.markdown(summary_text)

    st.subheader("💡 Future Feature: Personalized Treatment Engine")
    st.markdown("Imagine AI suggesting drugs based on patient-specific omics profiles. Coming soon!")

if compare_toggle and second_clinical_file and second_expression_file:
    st.header("🧪 Comparison: Second Dataset")
    try:
        second_clinical_df = pd.read_csv(second_clinical_file)
        second_expression_df = pd.read_csv(second_expression_file)
        st.success("Second dataset uploaded successfully.")
        st.write("Comparison features in progress...")
    except Exception as e:
        st.error(f"Error loading second dataset: {e}")
else:
    st.info("📂 Please upload your clinical and expression datasets to get started.")

