import streamlit as st
import pandas as pd
from modules.decision_engine import run_decision_engine
from explain.explain_utils import generate_shap_plot
from reports.report_utils import generate_pdf_report

st.set_page_config(page_title="FinGuard Admin Dashboard", layout="wide")

st.title("🛡️ FinGuard Pro — Admin Dashboard")

# Upload transaction CSV
uploaded = st.file_uploader("📤 Upload transaction file", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded, parse_dates=["timestamp"])
    st.subheader("Raw Transaction Preview")
    st.dataframe(df.head())

    if st.button("🔍 Run Fraud Detection"):
        with st.spinner("Running checks..."):
            results = run_decision_engine(df)
            st.success("Detection complete.")
            st.dataframe(results)

            suspicious = results[results['flagged'] == True]
            st.subheader("🚨 Flagged Transactions")
            st.dataframe(suspicious)

            # SHAP explanation
            st.subheader("🧠 SHAP Explanation")
            txn_id = st.selectbox("Select Transaction ID for Explanation", suspicious["transaction_id"].unique())
            if st.button("Explain Transaction"):
                shap_path = generate_shap_plot(txn_id)
                st.image(shap_path, caption=f"Explanation for Transaction {txn_id}")

            # Report download
            st.subheader("📄 Download Report")
            if st.button("Generate PDF Report"):
                path = generate_pdf_report(txn_id)
                with open(path, "rb") as f:
                    st.download_button("Download PDF", f, file_name="fraud_report.pdf")
