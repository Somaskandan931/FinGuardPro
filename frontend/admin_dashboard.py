import streamlit as st
import requests
import pandas as pd

API_BASE_URL = "http://localhost:5000"  # Adjust if needed

st.set_page_config(page_title="FinGuard Admin Portal")


def login():
    st.title("FinGuard Admin Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if not username or not password:
            st.error("Please enter username and password")
            return None, None

        try:
            resp = requests.post(f"{API_BASE_URL}/login", json={"username": username, "password": password})
        except requests.exceptions.RequestException as e:
            st.error(f"Login request failed: {e}")
            return None, None

        if resp.status_code == 200:
            data = resp.json()
            role = data.get("role")
            token = data.get("token")
            if role != "admin":
                st.error("Access denied: You are not an admin")
                return None, None
            st.success(f"Logged in as admin: {username}")
            return username, token
        else:
            st.error("Invalid credentials")
            return None, None
    return None, None


def single_transaction_form(token):
    st.header("Manual Transaction Input")

    with st.form("txn_form"):
        transaction_id = st.text_input("Transaction ID", value="TX0001")
        timestamp = st.text_input("Timestamp (YYYY-MM-DDTHH:MM:SS)", value="2025-07-08T14:30:00")
        sender_name = st.text_input("Sender Name", value="John Doe")
        recipient_name = st.text_input("Recipient Name", value="Jane Smith")
        transaction_amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)

        sender_account_type = st.selectbox("Sender Account Type", ["Savings", "Checking", "Credit"])
        recipient_account_type = st.selectbox("Recipient Account Type", ["Savings", "Checking", "Credit"])
        merchant_category = st.text_input("Merchant Category", value="Retail")
        device_type = st.selectbox("Device Type", ["Mobile", "Desktop", "Tablet"])
        transaction_type = st.text_input("Transaction Type", value="Online Purchase")

        submit = st.form_submit_button("Predict Fraud Risk")

    if submit:
        txn_data = {
            "transaction_id": transaction_id,
            "timestamp": timestamp,
            "sender_name": sender_name,
            "recipient_name": recipient_name,
            "transaction_amount": transaction_amount,
            "sender_account_type": sender_account_type,
            "recipient_account_type": recipient_account_type,
            "merchant_category": merchant_category,
            "device_type": device_type,
            "transaction_type": transaction_type,
        }

        headers = {"Authorization": f"Bearer {token}"}
        try:
            resp = requests.post(f"{API_BASE_URL}/predict", json=txn_data, headers=headers)
            if resp.status_code == 200:
                result = resp.json()
                st.success("Prediction successful!")
                st.json(result)
            else:
                st.error(f"Prediction failed: {resp.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Request error: {e}")


def show_shap_summary():
    st.header("SHAP Summary Plot")
    summary_url = f"{API_BASE_URL}/explain/shap_summary.png"

    try:
        st.image(summary_url, caption="SHAP Summary", use_column_width=True)
    except Exception as e:
        st.error(f"Failed to load SHAP summary plot: {e}")


def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.token = ""
    st.experimental_rerun()


def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "token" not in st.session_state:
        st.session_state.token = ""

    if not st.session_state.logged_in:
        username, token = login()
        if username and token:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.token = token
        else:
            return

    st.sidebar.write(f"👤 Logged in as: {st.session_state.username}")
    st.sidebar.button("Logout", on_click=logout)

    st.title("FinGuard Admin Portal")
    single_transaction_form(st.session_state.token)
    st.markdown("---")
    show_shap_summary()


if __name__ == "__main__":
    main()
