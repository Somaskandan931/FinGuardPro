import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import os
from explain.explain_utils import explain_model_prediction
from reports.report_utils import generate_fraud_report
from reports.zip_reports import generate_batch_reports

# Hash password (fixed usage for newer streamlit-authenticator versions)
hashed_passwords = stauth.Hasher().hash(['admin123'])

admin_config = {
    'credentials': {
        'usernames': {
            'admin1': {
                'name': 'Compliance Officer',
                'password': hashed_passwords[0]
            }
        }
    },
    'cookie': {
        'name': 'finguard_admin_cookie',
        'key': 'random_admin_key',
        'expiry_days': 1
    },
    'preauthorized': {'emails': []}
}

# ---- Authenticator ----
authenticator = stauth.Authenticate(
    admin_config['credentials'],
    admin_config['cookie']['name'],
    admin_config['cookie']['key'],
    admin_config['cookie']['expiry_days'],
    admin_config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login', 'main')

# ---- Post-login ----
if authentication_status:
    authenticator.logout("Logout", "sidebar")
    st.sidebar.success(f"Welcome, {name}")
    st.title("🔐 FinGuard Pro – Admin Dashboard")
    st.markdown("Monitor and explain fraud predictions in real time.")

    # Load model and preprocessor
    @st.cache_resource
    def load_components():
        try:
            model_path = 'C:/Users/somas/PycharmProjects/FinGuardPro/models/fraud_detection_model.h5'
            scaler_path = 'C:/Users/somas/PycharmProjects/FinGuardPro/models/scaler.pkl'

            if not os.path.exists(model_path):
                st.error(f"Model file not found at: {model_path}")
                return None, None
            if not os.path.exists(scaler_path):
                st.error(f"Scaler file not found at: {scaler_path}")
                return None, None

            model = tf.keras.models.load_model(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler
        except Exception as e:
            st.error(f"Error loading components: {e}")
            return None, None

    model, scaler = load_components()

    if model is None or scaler is None:
        st.error("Cannot proceed without model and scaler. Please check file paths.")
        st.stop()

    uploaded_file = st.file_uploader("Upload Transaction CSV", type="csv")

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)

            st.subheader("📄 Uploaded Data Preview")
            st.dataframe(data.head())

            if st.button("▶️ Run Fraud Prediction"):
                # Ensure data has the right columns for the model
                try:
                    X_scaled = scaler.transform(data)
                    fraud_scores = model.predict(X_scaled)
                    data['fraud_score'] = fraud_scores.flatten()
                    data['fraud_label'] = data['fraud_score'].apply(
                        lambda x: "🟢 Safe" if x < 0.3 else ("🟡 Suspicious" if x < 0.7 else "🔴 High Risk")
                    )

                    st.subheader("🚨 Fraud Predictions")
                    st.dataframe(data[['fraud_score', 'fraud_label']])

                    st.session_state['predicted_data'] = data  # Save for later buttons

                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    st.error("Please ensure your CSV has the correct columns expected by the model.")

        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

    # After prediction, show more options
    if 'predicted_data' in st.session_state:
        data = st.session_state['predicted_data']

        # Let admin select a specific transaction
        st.subheader("🔍 Explain a Transaction")
        selected_index = st.selectbox("Select a transaction index to explain", data.index.tolist())

        if st.button("🧠 Explain Selected Transaction"):
            try:
                sample_tx = data.iloc[[selected_index]]
                # Use all data as background or sample if too large
                background = data.sample(n=min(100, len(data)), random_state=42)

                # Remove fraud_score and fraud_label from feature columns
                feature_cols = [col for col in sample_tx.columns if col not in ['fraud_score', 'fraud_label']]

                img_path = explain_model_prediction(
                    model_path="C:/Users/somas/PycharmProjects/FinGuardPro/models/fraud_detection_model.h5",
                    scaler_path="C:/Users/somas/PycharmProjects/FinGuardPro/models/scaler.pkl",
                    sample_df=sample_tx[feature_cols],
                    background_df=background[feature_cols],
                    feature_names=feature_cols,
                    save_path="C:/Users/somas/PycharmProjects/FinGuardPro/explain/shap_explanation.png"
                )

                if os.path.exists(img_path):
                    st.image(img_path, caption=f"SHAP Explanation for Txn {selected_index}", use_column_width=True)
                else:
                    st.error("SHAP explanation image was not generated successfully.")

            except Exception as e:
                st.error(f"Error generating SHAP explanation: {e}")

        # PDF Report for selected transaction
        if st.button("📄 Download PDF Report for Selected Transaction"):
            try:
                sample_data = data.iloc[selected_index].to_dict()
                fraud_score = float(sample_data['fraud_score'])
                risk_level = "High Risk" if fraud_score > 0.7 else ("Suspicious" if fraud_score > 0.3 else "Safe")

                # Ensure reports directory exists
                reports_dir = "C:/Users/somas/PycharmProjects/FinGuardPro/reports"
                if not os.path.exists(reports_dir):
                    os.makedirs(reports_dir)

                pdf_path = generate_fraud_report(
                    txn_data=sample_data,
                    fraud_score=fraud_score,
                    risk_level=risk_level,
                    save_path=f"{reports_dir}/fraud_report_{selected_index}.pdf"
                )

                with open(pdf_path, "rb") as pdf_file:
                    PDF_BYTES = pdf_file.read()

                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=PDF_BYTES,
                    file_name=f"fraud_report_{selected_index}.pdf",
                    mime="application/pdf"
                )

            except Exception as e:
                st.error(f"Error generating PDF report: {e}")

        # Batch Report
        if st.button("📁 Generate PDF Reports for All Flagged Transactions"):
            try:
                zip_path, all_pdfs = generate_batch_reports(
                    df=data,
                    fraud_score_column="fraud_score",
                    threshold=0.3,
                    output_dir="C:/Users/somas/PycharmProjects/FinGuardPro/reports/batch_reports"
                )

                with open(zip_path, "rb") as zip_file:
                    ZIP_BYTES = zip_file.read()

                st.success(f"✅ {len(all_pdfs)} reports generated and zipped!")

                st.download_button(
                    label="⬇️ Download All Fraud Reports (.zip)",
                    data=ZIP_BYTES,
                    file_name="fraud_reports.zip",
                    mime="application/zip"
                )

            except Exception as e:
                st.error(f"Error generating batch reports: {e}")

elif authentication_status is False:
    st.error("❌ Invalid username or password")

elif authentication_status is None:
    st.warning("🔒 Please enter your credentials")