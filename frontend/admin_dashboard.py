import streamlit as st
import requests
import pandas as pd

API_BASE_URL = "http://localhost:5000"  # Change as needed

st.set_page_config(page_title="FinGuard Admin Portal")

def login():
    st.title("FinGuard Admin Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if not username or not password:
            st.error("Please enter username and password")
            return None

        resp = requests.post(f"{API_BASE_URL}/login", json={"username": username, "password": password})
        if resp.status_code == 200:
            role = resp.json().get("role")
            if role != "admin":
                st.error("Access denied: You are not an admin")
                return None
            st.success(f"Logged in as admin: {username}")
            return username
        else:
            st.error("Invalid credentials")
            return None
    return None

def batch_upload():
    st.header("Batch Upload Transactions CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} records.")
            if st.button("Run Batch Prediction"):
                with st.spinner("Running batch predictions..."):
                    results = []
                    for _, row in df.iterrows():
                        txn = row.to_dict()
                        resp = requests.post(f"{API_BASE_URL}/predict", json=txn)
                        if resp.status_code == 200:
                            results.append(resp.json())
                        else:
                            st.error(f"Failed prediction for one record: {resp.text}")
                    st.success(f"Completed batch predictions for {len(results)} records.")
                    st.dataframe(pd.DataFrame(results))
        except Exception as e:
            st.error(f"Failed to process CSV: {e}")

def show_shap_summary():
    st.header("SHAP Summary Plot")
    summary_url = f"{API_BASE_URL}/explain/shap_summary.png"
    st.image(summary_url, caption="SHAP Summary")

def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""

    if not st.session_state.logged_in:
        username = login()
        if username:
            st.session_state.logged_in = True
            st.session_state.username = username
        else:
            return

    st.write(f"Welcome, admin {st.session_state.username}")

    batch_upload()
    st.markdown("---")
    show_shap_summary()

if __name__ == "__main__":
    main()
