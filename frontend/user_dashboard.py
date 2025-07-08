import streamlit as st
import requests
import json
from io import BytesIO
from PIL import Image

API_BASE_URL = "http://localhost:5000"  # Change if your Flask backend runs elsewhere

st.set_page_config( page_title="FinGuard User Portal" )


def login () :
    st.title( "FinGuard User Login" )

    username = st.text_input( "Username" )
    password = st.text_input( "Password", type="password" )

    if st.button( "Login" ) :
        if not username or not password :
            st.error( "Please enter username and password" )
            return None

        resp = requests.post( f"{API_BASE_URL}/login", json={"username" : username, "password" : password} )
        if resp.status_code == 200 :
            role = resp.json().get( "role" )
            if role != "user" :
                st.error( "Access denied: You are not a user" )
                return None
            st.success( f"Logged in as {username}" )
            return username
        else :
            st.error( "Invalid credentials" )
            return None
    return None


def transaction_form () :
    st.header( "Enter Transaction Details" )

    with st.form( "txn_form", clear_on_submit=True ) :
        amount = st.number_input( "Transaction Amount", min_value=0.01, step=0.01 )
        sender_name = st.text_input( "Sender Name" )
        sender_risk_profile = st.selectbox( "Sender Risk Profile", ["Low", "Medium", "High"] )
        sender_account_type = st.selectbox( "Sender Account Type", ["Savings", "Current", "Credit"] )

        recipient_known = st.checkbox( "Is Recipient Known?", value=True )
        recipient_name = ""
        recipient_account_type = ""
        if recipient_known :
            recipient_name = st.text_input( "Recipient Name" )
            recipient_account_type = st.selectbox( "Recipient Account Type", ["Savings", "Current", "Credit"] )

        device_type = st.selectbox( "Device Type", ["Mobile", "Desktop", "Tablet", "Other"] )
        merchant_category = st.selectbox( "Merchant Category",
                                          ["Retail", "Food", "Utilities", "Entertainment", "Other"] )

        transaction_type = st.selectbox( "Transaction Type", ["Payment", "Transfer", "Withdrawal", "Deposit"] )

        transaction_hour = st.slider( "Transaction Hour (24h)", 0, 23, 12 )
        day_of_week = st.selectbox( "Day of Week", list( range( 7 ) ) )  # 0=Monday

        location = st.text_input( "Transaction Location" )

        submit = st.form_submit_button( "Submit" )

    if submit :
        # Build payload dict
        txn_data = {
            "transaction_amount" : amount,
            "sender_name" : sender_name,
            "sender_risk_profile" : sender_risk_profile,
            "sender_account_type" : sender_account_type,
            "recipient_name" : recipient_name if recipient_known else "",
            "recipient_account_type" : recipient_account_type if recipient_known else "",
            "device_type" : device_type,
            "merchant_category" : merchant_category,
            "transaction_type" : transaction_type,
            "hour_of_day" : transaction_hour,
            "day_of_week" : day_of_week,
            "location" : location
        }
        return txn_data
    return None


def display_results ( results ) :
    st.subheader( "Fraud Detection Results" )
    st.write( f"**Fraud Score:** {results.get( 'fraud_score', 'N/A' ):.3f}" )
    st.write( f"**Risk Level:** {results.get( 'risk_level', 'N/A' )}" )

    if results.get( "shap_plot" ) :
        st.image( results["shap_plot"], caption="SHAP Waterfall Explanation" )

    if results.get( "report_path" ) :
        st.markdown( f"[Download Fraud Report]({results['report_path']})" )


def main () :
    if "logged_in" not in st.session_state :
        st.session_state.logged_in = False
    if "username" not in st.session_state :
        st.session_state.username = ""

    if not st.session_state.logged_in :
        username = login()
        if username :
            st.session_state.logged_in = True
            st.session_state.username = username
        else :
            return

    txn_data = transaction_form()
    if txn_data :
        with st.spinner( "Running fraud detection..." ) :
            try :
                resp = requests.post( f"{API_BASE_URL}/predict", json=txn_data )
                if resp.status_code == 200 :
                    results = resp.json()
                    # SHAP plot path is a local server path; we need to serve images properly
                    # For demo, let's fetch image from backend API or convert to base64 if needed
                    # Here, assuming Flask static serving is setup to serve /explain/shap_outputs/
                    if results.get( "shap_plot" ) :
                        shap_plot_url = f"{API_BASE_URL}/{results['shap_plot']}"
                        results["shap_plot"] = shap_plot_url
                    if results.get( "report_path" ) :
                        report_url = f"{API_BASE_URL}/{results['report_path']}"
                        results["report_path"] = report_url
                    display_results( results )
                else :
                    st.error( f"Prediction failed: {resp.text}" )
            except Exception as e :
                st.error( f"Error contacting backend: {e}" )


if __name__ == "__main__" :
    main()
