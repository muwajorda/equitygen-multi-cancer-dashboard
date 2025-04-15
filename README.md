# EquityGen: Multi-Cancer Equity Dashboard

🧬 A precision medicine dashboard designed to visualize and model racial disparities across multiple cancer types using TCGA data.

## 🌟 Features

- Upload clinical and gene expression data for any TCGA cancer type (e.g., BRCA, LUAD, COAD)
- Visualize gene expression differences by race
- Perform Kaplan-Meier survival analysis
- Automatically select top 500 most variable genes
- Predict mortality risk using ML models (Random Forest, Logistic Regression)
- Download race-stratified datasets per indication

## 🚀 How to Run (Locally)

```bash
pip install -r requirements.txt
streamlit run EquityGen_Dashboard.py

