import io

import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import requests
from datetime import datetime
import time

# Configure page
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Flask API Configuration
FLASK_API_URL = "http://localhost:5000"  # Change this to your Flask server URL

# Fixed CSS for better contrast and styling
st.markdown( """
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }

    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2.5rem;
    }

    .main-header p {
        color: #e3f2fd !important;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }

    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #1e3c72;
        margin-bottom: 1rem;
        color: #333 !important;
    }

    .metric-card h3 {
        color: #1e3c72 !important;
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
    }

    .metric-card h1 {
        color: #2a5298 !important;
        margin: 0 0 0.5rem 0;
        font-size: 2rem;
    }

    .metric-card p {
        color: #666 !important;
        margin: 0;
        font-size: 0.9rem;
    }

    .alert-high {
        background: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #333 !important;
    }

    .alert-medium {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #333 !important;
    }

    .alert-low {
        background: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #333 !important;
    }

    .stButton > button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        width: 100%;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #2a5298 0%, #1e3c72 100%);
    }

    .error-message {
        background: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #d32f2f !important;
    }

    .success-message {
        background: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #2e7d32 !important;
    }
</style>
""", unsafe_allow_html=True )

# Initialize session state
if 'results' not in st.session_state :
    st.session_state.results = {}
if 'authenticated' not in st.session_state :
    st.session_state.authenticated = False
if 'username' not in st.session_state :
    st.session_state.username = None
if 'auth_token' not in st.session_state :
    st.session_state.auth_token = None


# API Helper Functions
def login_to_flask ( username, password ) :
    """Login to Flask API and get token"""
    try :
        response = requests.post(
            f"{FLASK_API_URL}/login",
            json={"username" : username, "password" : password},
            timeout=10
        )
        if response.status_code == 200 :
            data = response.json()
            return data.get( 'token' ), data.get( 'message' )
        else :
            return None, response.json().get( 'message', 'Login failed' )
    except requests.exceptions.RequestException as e :
        return None, f"Connection error: {str( e )}"


def call_flask_predict ( transaction_data, token ) :
    """Call Flask prediction endpoint"""
    try :
        headers = {'Authorization' : f'Bearer {token}'}
        response = requests.post(
            f"{FLASK_API_URL}/predict",
            json=transaction_data,
            headers=headers,
            timeout=30
        )
        if response.status_code == 200 :
            return response.json(), None
        else :
            return None, response.json().get( 'error', 'Prediction failed' )
    except requests.exceptions.RequestException as e :
        return None, f"Connection error: {str( e )}"

def call_flask_names_screening(screening_data, token):
    """Call Flask names screening endpoint"""
    try:
        headers = {'Authorization': f'Bearer {token}'}
        response = requests.post(
            f"{FLASK_API_URL}/names_screening",
            json=screening_data,
            headers=headers,
            timeout=30
        )
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, response.json().get('error', 'Names screening failed')
    except requests.exceptions.RequestException as e:
        return None, f"Connection error: {str(e)}"

def call_flask_transaction_screening(transaction_data, token):
    """Call Flask transaction screening endpoint"""
    try:
        headers = {'Authorization': f'Bearer {token}'}
        response = requests.post(
            f"{FLASK_API_URL}/transaction_screening",
            json=transaction_data,
            headers=headers,
            timeout=30
        )
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, response.json().get('error', 'Transaction screening failed')
    except requests.exceptions.RequestException as e:
        return None, f"Connection error: {str(e)}"

def authenticate_user ( username, password ) :
    """Authenticate user with Flask API"""
    token, message = login_to_flask( username, password )
    if token :
        st.session_state.authenticated = True
        st.session_state.username = username
        st.session_state.auth_token = token
        return True, message
    return False, message


def show_login () :
    """Display login form"""
    st.markdown( """
    <div class="main-header">
        <h1>🛡️ Fraud Detection System</h1>
        <p>Secure Login Portal</p>
    </div>
    """, unsafe_allow_html=True )

    col1, col2, col3 = st.columns( [1, 2, 1] )

    with col2 :

        st.markdown( "### 🔐 System Login" )

        with st.form( "login_form" ) :
            username = st.text_input( "Username", placeholder="Enter your username" )
            password = st.text_input( "Password", type="password", placeholder="Enter your password" )

            col_login, col_demo = st.columns( 2 )

            with col_login :
                login_submitted = st.form_submit_button( "🔑 Login", use_container_width=True )

            with col_demo :
                demo_submitted = st.form_submit_button( "👤 Demo Login", use_container_width=True )

            if login_submitted :
                if username and password :
                    with st.spinner( "Authenticating..." ) :
                        success, message = authenticate_user( username, password )
                        if success :
                            st.success( f"Welcome, {username}!" )
                            st.rerun()
                        else :
                            st.error( f"Login failed: {message}" )
                else :
                    st.error( "Please enter both username and password" )

            if demo_submitted :
                with st.spinner( "Authenticating..." ) :
                    success, message = authenticate_user( "admin", "password123" )
                    if success :
                        st.success( "Demo login successful!" )
                        st.rerun()
                    else :
                        st.error( f"Demo login failed: {message}" )



        # Connection info
        st.info( f"""
        **Flask API Connection:** {FLASK_API_URL}

        **Demo Credentials:**
        - Username: admin
        - Password: password123

        **Note:** Make sure your Flask server is running on {FLASK_API_URL}
        """ )


def main () :
    # Check authentication first
    if not st.session_state.authenticated :
        show_login()
        return

    # Header with logout
    col1, col2 = st.columns( [4, 1] )

    with col1 :
        st.markdown( f"""
        <div class="main-header">
            <h1>🛡️ Fraud Detection Dashboard</h1>
            <p>Welcome back, {st.session_state.username}! | Financial Crime Prevention & ML Detection</p>
        </div>
        """, unsafe_allow_html=True )

    with col2 :
        if st.button( "🚪 Logout", use_container_width=True ) :
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.auth_token = None
            st.rerun()

    # Sidebar navigation
    with st.sidebar :
        st.markdown( "### 🎛️ Control Panel" )
        st.markdown( f"**Logged in as:** {st.session_state.username}" )
        st.markdown( f"**API Status:** {'🟢 Connected' if st.session_state.auth_token else '🔴 Disconnected'}" )
        st.markdown( "---" )

        page = st.selectbox(
            "Select Module:",
            ["ML Fraud Detection", "Names Screening", "Transaction Screening", "Dashboard Overview"]
        )

        st.markdown( "---" )
        if st.button( "🔄 Refresh Data" ) :
            st.rerun()

        # API Configuration
        st.markdown( "### ⚙️ API Configuration" )
        new_url = st.text_input( "Flask API URL", value=FLASK_API_URL, key="api_url" )
        if st.button( "Update API URL" ) :
            globals()['FLASK_API_URL'] = new_url
            st.success( "API URL updated!" )

    # Page routing
    if page == "ML Fraud Detection" :
        show_ml_fraud_detection()
    elif page == "Names Screening" :
        show_names_screening()
    elif page == "Transaction Screening" :
        show_transaction_screening()
    elif page == "Dashboard Overview" :
        show_dashboard_overview()


def generate_explanations(transaction: dict) -> list:
    explanations = []

    if transaction.get("hour_of_day", 0) >= 22 or transaction.get("hour_of_day", 0) <= 5:
        explanations.append("The transaction occurred at a late or unusual hour.")

    if transaction.get("is_weekend"):
        explanations.append("The transaction was made during the weekend, which may carry higher fraud risk.")

    if transaction.get("amount", 0) > 1_000_000:
        explanations.append(f"The amount (${transaction['amount']:,}) is very high and exceeds common thresholds.")

    if transaction.get("sender_risk_profile") == "low" and transaction.get("amount", 0) > 500_000:
        explanations.append("The sender has a low risk profile but is making a large transaction.")

    if transaction.get("recipient_balance_before", 0) < 100_000:
        explanations.append("The recipient has a relatively low balance, which can indicate potential mule account.")

    if transaction.get("amount_to_avg_ratio", 1.0) > 2.0:
        explanations.append("The transaction amount is significantly higher than the sender's average transaction "
                            "value.")

    if transaction.get("device_type") == "mobile" and transaction.get("amount", 0) > 1_000_000:
        explanations.append("A very large transaction was made via mobile device, which is unusual.")

    return explanations

def show_ml_fraud_detection():
    """ML-based fraud detection with Flask API integration"""
    st.markdown("## 🤖 ML Fraud Detection Model")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📊 Transaction & User Features")

        with st.form("ml_fraud_form"):
            # Transaction Details
            st.markdown("**Transaction Information**")
            transaction_id = st.text_input("Transaction ID", value="TXN_001")
            amount = st.number_input("Amount ($)", min_value=0.0, value=1500.0, step=100.0)
            transaction_type = st.selectbox("Transaction Type", ["transfer", "payment", "withdrawal", "deposit", "purchase"])

            # User Information
            st.markdown("**User Profile**")
            col1a, col1b = st.columns(2)
            with col1a:
                sender_age = st.number_input("Sender Age", min_value=18, max_value=100, value=35)
                sender_balance_before = st.number_input("Sender Balance ($)", min_value=0.0, value=5000.0)
            with col1b:
                sender_account_type = st.selectbox("Sender Account Type", ["savings", "checking", "business"])
                sender_risk_profile = st.selectbox("Sender Risk Profile", ["low", "medium", "high"])

            # Recipient Information
            st.markdown("**Recipient Information**")
            col2a, col2b = st.columns(2)
            with col2a:
                recipient_balance_before = st.number_input("Recipient Balance ($)", min_value=0.0, value=3000.0)
                recipient_account_type = st.selectbox("Recipient Account Type", ["savings", "checking", "business"])
            with col2b:
                is_new_receiver = st.checkbox("New Recipient")

            # Transaction Context
            st.markdown("**Transaction Context**")
            col3a, col3b = st.columns(2)
            with col3a:
                hour_of_day = st.slider("Hour of Day", 0, 23, 14)
                day_of_week = st.slider("Day of Week", 0, 6, 2)
                is_weekend = st.checkbox("Weekend Transaction")
            with col3b:
                txns_last_hour = st.number_input("Transactions (last hour)", min_value=0, value=2)
                txns_last_day = st.number_input("Transactions (last 24h)", min_value=0, value=5)
                txns_last_week = st.number_input("Transactions (last week)", min_value=0, value=15)

            # Technical Details
            st.markdown("**Technical & Location Data**")
            col4a, col4b = st.columns(2)
            with col4a:
                device_type = st.selectbox("Device Type", ["mobile", "web", "atm", "unknown"])
                location = st.selectbox("Location", ["USA", "Canada", "UK", "High Risk Country"])
            with col4b:
                merchant_category = st.selectbox("Merchant Category", ["retail", "online", "gas_station", "restaurant", "other"])

            # Advanced Features
            st.markdown("**Advanced Model Features**")
            col5a, col5b = st.columns(2)
            with col5a:
                amount_to_balance_ratio = st.slider("Amount/Balance Ratio", 0.0, 2.0, 0.5)
                amount_vs_channel_limit_ratio = st.slider("Amount vs Channel Limit", 0.0, 2.0, 0.5)
                is_round_amount = st.checkbox("Round Amount (100, 500, 1000)")
                is_high_value = st.checkbox("High Value Transaction (>$10k)")
            with col5b:
                log_amount = st.number_input("Log Amount", value=7.31, step=0.01)
                sender_txn_count = st.number_input("Sender Transaction Count", min_value=0, value=150)
                amount_to_avg_ratio = st.slider("Amount vs Historical Avg", 0.5, 5.0, 1.5)

            submitted = st.form_submit_button("🔍 Run ML Detection Model")

            if submitted:
                transaction_data = {
                    'transaction_id': transaction_id,
                    'amount': amount,
                    'transaction_type': transaction_type,
                    'sender_age': sender_age,
                    'sender_balance_before': sender_balance_before,
                    'sender_account_type': sender_account_type,
                    'sender_risk_profile': sender_risk_profile,
                    'recipient_balance_before': recipient_balance_before,
                    'recipient_account_type': recipient_account_type,
                    'is_new_receiver': is_new_receiver,
                    'hour_of_day': hour_of_day,
                    'day_of_week': day_of_week,
                    'is_weekend': is_weekend,
                    'txns_last_hour': txns_last_hour,
                    'txns_last_day': txns_last_day,
                    'txns_last_week': txns_last_week,
                    'device_type': device_type,
                    'location': location,
                    'merchant_category': merchant_category,
                    'amount_to_balance_ratio': amount_to_balance_ratio,
                    'amount_vs_channel_limit_ratio': amount_vs_channel_limit_ratio,
                    'is_round_amount': is_round_amount,
                    'is_high_value': is_high_value,
                    'log_amount': log_amount,
                    'sender_txn_count': sender_txn_count,
                    'amount_to_avg_ratio': amount_to_avg_ratio
                }

                with st.spinner("Calling ML fraud detection API..."):
                    # Convert all values to native Python types
                    cleaned_data = {
                        k : (float( v ) if isinstance( v, (np.float32, np.float64) ) else
                             int( v ) if isinstance( v, (np.int32, np.int64) ) else
                             bool( v ) if isinstance( v, (np.bool_) ) else
                             v)
                        for k, v in transaction_data.items()
                    }

                    result, error = call_flask_predict( cleaned_data, st.session_state.auth_token )
                    if error:
                        st.error(f"API Error: {error}")
                        if "token" in error.lower() or "unauthorized" in error.lower():
                            st.warning("Session expired. Please login again.")
                            st.session_state.authenticated = False
                            st.rerun()
                    else:
                        st.session_state.results['ml_fraud'] = result
                        st.success("ML model analysis complete!")

    with col2:
        st.markdown("### 🎯 ML Model Results")

        if 'ml_fraud' in st.session_state.results :
            result = st.session_state.results['ml_fraud']

            fraud_score = result.get( 'fraud_score', 0 )
            risk_level = result.get( 'risk_level', 'Low' )

            # 📈 Gauge Chart
            fig = go.Figure( go.Indicator(
                mode="gauge+number",
                value=fraud_score * 100,
                domain={'x' : [0, 1], 'y' : [0, 1]},
                title={'text' : "ML Fraud Score (%)"},
                gauge={
                    'axis' : {'range' : [None, 100]},
                    'bar' : {
                        'color' : "#f44336" if risk_level == "High" else "#ff9800" if risk_level == "Medium" else "#4caf50"},
                    'steps' : [
                        {'range' : [0, 50], 'color' : "#e8f5e8"},
                        {'range' : [50, 80], 'color' : "#fff3e0"},
                        {'range' : [80, 100], 'color' : "#ffebee"}
                    ],
                    'threshold' : {
                        'line' : {'color' : "red", 'width' : 4},
                        'thickness' : 0.75,
                        'value' : 80
                    }
                }
            ) )
            fig.update_layout( height=300 )
            st.plotly_chart( fig, use_container_width=True )

            # 🧾 Key Metrics
            col_metrics1, col_metrics2 = st.columns( 2 )
            with col_metrics1 :
                st.metric( "Risk Level", risk_level )
                st.metric( "Transaction ID", result.get( 'transaction_id', 'N/A' ) )
            with col_metrics2 :
                st.metric( "Fraud Score", f"{fraud_score:.3f}" )

            # 💡 Risk Factor Explanations
            st.markdown( "#### 💡 Risk Factors (based on transaction features)" )
            explanations = generate_explanations( result )
            if explanations :
                for exp in explanations :
                    st.markdown( f"• {exp}" )
            else :
                st.markdown( "No strong individual risk indicators detected." )

            # Separator
            st.markdown( "---" )

            # 🛡️ Action Recommendations
            st.markdown( "#### 💡 Recommended Actions" )
            if risk_level == "High" :
                st.markdown( '<div class="alert-high"><strong>⚠️ HIGH RISK - IMMEDIATE ACTION REQUIRED</strong></div>',
                             unsafe_allow_html=True )
                st.markdown( "• Flag transaction for manual review" )
                st.markdown( "• Contact customer for verification" )
                st.markdown( "• Consider temporary account restriction" )
            elif risk_level == "Medium" :
                st.markdown( '<div class="alert-medium"><strong>⚠️ MEDIUM RISK - ENHANCED MONITORING</strong></div>',
                             unsafe_allow_html=True )
                st.markdown( "• Enable enhanced monitoring" )
                st.markdown( "• Log for pattern analysis" )
                st.markdown( "• Continue transaction with alerts" )
            else :
                st.markdown( '<div class="alert-low"><strong>✅ LOW RISK - TRANSACTION APPROVED</strong></div>',
                             unsafe_allow_html=True )
                st.markdown( "• Process transaction normally" )
                st.markdown( "• Standard monitoring applies" )

            # 📄 Report & SHAP Links
            if result.get( 'report_path' ) :
                st.markdown( "#### 📄 Generated Reports" )
                st.markdown( f"[📊 Fraud Report]({FLASK_API_URL}{result['report_path']})" )

            if result.get( 'shap_plot' ) :
                st.markdown( f"[🔍 SHAP Explanation]({FLASK_API_URL}{result['shap_plot']})" )

        else :
            st.info( "Submit transaction data for ML model analysis" )
            st.markdown( "#### 🤖 Model Information" )
            st.markdown( "**Model Type:** XGBoost Classifier" )
            st.markdown( "**API Endpoint:** /predict" )
            st.markdown( "**Authentication:** Bearer Token" )
            st.markdown( "**Features:** 24+ transaction & user features" )
            st.markdown( f"**API URL:** {FLASK_API_URL}" )


def show_dashboard_overview () :
    """Dashboard overview with key metrics"""
    st.markdown( "## 📊 Dashboard Overview" )

    # Connection status
    col_status1, col_status2 = st.columns( 2 )
    with col_status1 :
        if st.session_state.auth_token :
            st.markdown( '<div class="success-message">🟢 Connected to Flask API</div>', unsafe_allow_html=True )
        else :
            st.markdown( '<div class="error-message">🔴 Not connected to Flask API</div>', unsafe_allow_html=True )

    with col_status2 :
        st.markdown( f"**API Endpoint:** {FLASK_API_URL}" )

    # Key metrics (static for now - you can extend to call API endpoints)
    col1, col2, col3, col4 = st.columns( 4 )

    with col1 :
        st.markdown( """
        <div class="metric-card">
            <h3>API Status</h3>
            <h1 style="color: #4caf50 !important;">✅</h1>
            <p>Connected & Ready</p>
        </div>
        """, unsafe_allow_html=True )

    with col2 :
        st.markdown( """
        <div class="metric-card">
            <h3>Session Status</h3>
            <h1 style="color: #4caf50 !important;">🔑</h1>
            <p>Authenticated</p>
        </div>
        """, unsafe_allow_html=True )

    with col3 :
        st.markdown( """
        <div class="metric-card">
            <h3>Model Type</h3>
            <h1 style="color: #2a5298 !important;">XGB</h1>
            <p>XGBoost Classifier</p>
        </div>
        """, unsafe_allow_html=True )

    with col4 :
        st.markdown( """
        <div class="metric-card">
            <h3>Features</h3>
            <h1 style="color: #2a5298 !important;">24+</h1>
            <p>Transaction Features</p>
        </div>
        """, unsafe_allow_html=True )

    # API Testing Section
    st.markdown( "### 🧪 API Testing" )

    col_test1, col_test2 = st.columns( 2 )

    with col_test1 :
        if st.button( "🔍 Test API Connection" ) :
            with st.spinner( "Testing API connection..." ) :
                try :
                    response = requests.get( f"{FLASK_API_URL}/", timeout=5 )
                    st.success( "✅ API is reachable" )
                except requests.exceptions.RequestException as e :
                    st.error( f"❌ API connection failed: {str( e )}" )

    with col_test2 :
        if st.button( "🔑 Test Authentication" ) :
            if st.session_state.auth_token :
                st.success( "✅ Authentication token is valid" )
            else :
                st.error( "❌ No authentication token" )

    # Recent activity placeholder
    st.markdown( "### 📈 Recent Activity" )
    st.info( "Connect to Flask API endpoints to view real-time fraud detection activity" )


def show_names_screening():
    st.title("Names Screening Dashboard")

    st.markdown("## 👤 AML List Upload Module")

    with st.expander("📎 Upload AML Watchlist", expanded=True):
        uploaded_file = st.file_uploader("Upload AML CSV File", type=["csv"])

        file_bytes = None
        df = None

        if uploaded_file:
            try:
                # Read bytes and decode once
                file_bytes = uploaded_file.read()
                decoded_str = file_bytes.decode("utf-8")
                df = pd.read_csv(io.StringIO(decoded_str))

                st.success("✅ AML file loaded successfully.")
                st.dataframe(df.head())

                # Show columns to help debug schema
                st.write("Columns detected in file:", df.columns.tolist())

            except Exception as e:
                st.error(f"❌ Error reading AML file: {e}")
                df = None

        if df is not None and file_bytes is not None:
            if st.button("📤 Upload to AML Database"):
                try:
                    files = {
                        "file": (uploaded_file.name, io.BytesIO(file_bytes), "text/csv")
                    }
                    response = requests.post(
                        "http://localhost:5000/upload-aml-list",
                        files=files,
                    )

                    if response.status_code == 200:
                        st.success("✅ AML list uploaded successfully.")
                    else:
                        st.error(f"❌ Upload failed: {response.status_code} - {response.text}")

                except Exception as e:
                    st.error(f"❌ Upload error: {e}")

    st.markdown("---")  # Divider

    # --- Names Screening Section ---
    st.markdown("## 👤 Names Screening Module")

    if 'results' not in st.session_state:
        st.session_state.results = {}

    st.markdown("### 📂 Upload Transaction Dataset")
    uploaded_txn_file = st.file_uploader("Upload a CSV file with transaction data", type=["csv"], key="txn_upload")

    if uploaded_txn_file:
        try:
            uploaded_txn_file.seek(0)
            df = pd.read_csv(uploaded_txn_file)
            st.success("✅ Transaction file uploaded successfully!")

            transaction_ids = df['transaction_id'].astype(str).tolist()
            selected_txn_id = st.selectbox("Select a Transaction", transaction_ids)

            selected_row = df[df['transaction_id'].astype(str) == selected_txn_id].iloc[0]

            default_sender_name = selected_row['sender_name']
            default_recipient_name = selected_row['recipient_name']
            default_amount = float(selected_row['amount'])
        except Exception as e:
            st.error(f"❌ Error reading transaction CSV: {e}")
            df = None
            selected_row = None
            default_sender_name = "John Smith"
            default_recipient_name = "Jane Doe"
            default_amount = 5000.0
    else:
        df = None
        selected_row = None
        default_sender_name = "John Smith"
        default_recipient_name = "Jane Doe"
        default_amount = 5000.0

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📝 Screening Input")
        with st.form("names_screening_form"):
            transaction_id = st.text_input("Transaction Reference ID", value=selected_row['transaction_id'] if selected_row is not None else "TXN_NAME_001")

            # Sender Details
            st.markdown("#### 🧑 Sender Details")
            sender_name = st.text_input("Sender Full Name", value=default_sender_name)
            sender_country = st.selectbox("Sender Country", ["USA", "Canada", "UK", "Germany", "France", "High Risk Country"])
            sender_city = st.text_input("Sender City", value="New York")

            # Recipient Details
            st.markdown("#### 👤 Recipient Details")
            recipient_name = st.text_input("Recipient Full Name", value=default_recipient_name)
            recipient_country = st.selectbox("Recipient Country", ["USA", "Canada", "UK", "Germany", "France", "High Risk Country"])
            recipient_city = st.text_input("Recipient City", value="London")

            # Transaction Details
            st.markdown("#### 💰 Transaction Details")
            amount = st.number_input("Transaction Amount (USD)", min_value=0.0, value=default_amount, step=0.01)
            purpose = st.text_input("Purpose of Transaction", value="Business Payment")

            # Screening Options
            st.markdown("#### 🛡️ Screening Preferences")
            col_check1, col_check2 = st.columns(2)
            with col_check1:
                check_sanctions = st.checkbox("Sanctions List", value=True)
                check_pep = st.checkbox("PEP List", value=True)
            with col_check2:
                check_adverse_media = st.checkbox("Adverse Media", value=True)
                check_watchlist = st.checkbox("Internal Watchlist", value=True)

            submitted = st.form_submit_button("🔍 Run Names Screening")

            if submitted:
                screening_data = {
                    "transaction_id": transaction_id,
                    "sender_name": sender_name,
                    "sender_country": sender_country,
                    "sender_city": sender_city,
                    "recipient_name": recipient_name,
                    "recipient_country": recipient_country,
                    "recipient_city": recipient_city,
                    "amount": amount,
                    "purpose": purpose,
                    "check_sanctions": check_sanctions,
                    "check_pep": check_pep,
                    "check_adverse_media": check_adverse_media,
                    "check_watchlist": check_watchlist,
                }

                with st.spinner("🔄 Running names screening..."):
                    try:
                        response = requests.post(
                            "http://localhost:5000/names-screening",
                            json=screening_data,
                        )

                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.results['names_screening'] = result
                            st.success("✅ Names screening completed!")
                        elif response.status_code == 401:
                            st.error("❌ Unauthorized. Please log in again.")
                        else:
                            st.error(f"❌ Screening failed: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Error calling screening service: {e}")

    with col2:
        st.markdown("### 📟 Screening Results")
        if 'names_screening' in st.session_state.results:
            result = st.session_state.results['names_screening']

            risk_color = {
                "High": "#f44336",
                "Medium": "#ff9800",
                "Low": "#4caf50"
            }.get(result.get('overall_risk', 'Low'), "#9e9e9e")

            st.markdown(f"""
                <div style="background: {risk_color}; color: white; padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
                    <h2>Overall Risk: {result.get('overall_risk', 'Low')}</h2>
                    <p>Recommended Action: {result.get('recommended_action', 'Review')}</p>
                </div>
            """, unsafe_allow_html=True)

            # Sender Risk
            st.markdown("#### 🧑 Sender Risk")
            s = result.get('sender_screening', {})
            st.metric("Risk Score", f"{s.get('risk_score', 0):.2f}")
            st.metric("Sanctions", "Yes" if s.get('sanctions_hit') else "No")
            st.metric("PEP", "Yes" if s.get('pep_hit') else "No")
            st.metric("Matches Found", "Yes" if s.get('matches_found') else "No")
            if s.get('match_details'):
                st.markdown("**Match Details:**")
                for detail in s['match_details']:
                    st.markdown(f"• {detail}")

            # Recipient Risk
            st.markdown("#### 👤 Recipient Risk")
            r = result.get('recipient_screening', {})
            st.metric("Risk Score", f"{r.get('risk_score', 0):.2f}")
            st.metric("Sanctions", "Yes" if r.get('sanctions_hit') else "No")
            st.metric("PEP", "Yes" if r.get('pep_hit') else "No")
            st.metric("Matches Found", "Yes" if r.get('matches_found') else "No")
            if r.get('match_details'):
                st.markdown("**Match Details:**")
                for detail in r['match_details']:
                    st.markdown(f"• {detail}")

            # Recommended Action
            st.markdown("#### 💡 Recommended Action")
            action = result.get('recommended_action', 'Review')
            if action == 'Block':
                st.markdown('<div class="alert-high"><strong>🚫 BLOCK TRANSACTION</strong></div>', unsafe_allow_html=True)
            elif action == 'Review':
                st.markdown('<div class="alert-medium"><strong>⚠️ MANUAL REVIEW REQUIRED</strong></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-low"><strong>✅ APPROVED</strong></div>', unsafe_allow_html=True)
        else:
            st.info("Submit form to view screening results.")




def show_transaction_screening():
    """Transaction screening module"""
    st.markdown("## 🔍 Transaction Screening Module")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📊 Transaction Rules Engine")

        with st.form("transaction_screening_form"):
            # --- Transaction Info ---
            st.markdown("**Basic Transaction Information**")
            transaction_id = st.text_input("Transaction ID", value="TXN_RULE_001")
            amount = st.number_input("Amount ($)", min_value=0.0, value=15000.0, step=100.0)
            transaction_type = st.selectbox("Transaction Type", [
                "wire_transfer", "cash_deposit", "cash_withdrawal", "international_transfer", "domestic_transfer"
            ])

            # --- Geographic Info ---
            st.markdown("**Geographic Details**")
            col_geo1, col_geo2 = st.columns(2)
            with col_geo1:
                sender_country = st.selectbox("Sender Country", ["USA", "Canada", "UK", "Germany", "High Risk Country"])
                sender_state = st.text_input("Sender State/Province", value="NY")
            with col_geo2:
                recipient_country = st.selectbox("Recipient Country", ["USA", "Canada", "UK", "Germany", "High Risk Country"])
                recipient_state = st.text_input("Recipient State/Province", value="CA")

            # --- Timing Info ---
            st.markdown("**Timing Information**")
            col_time1, col_time2 = st.columns(2)
            with col_time1:
                hour_of_day = st.slider("Hour of Day", 0, 23, 22)
                is_weekend = st.checkbox("Weekend Transaction")
            with col_time2:
                is_holiday = st.checkbox("Holiday Transaction")
                is_after_hours = st.checkbox("After Business Hours")

            # --- Velocity Checks ---
            st.markdown("**Velocity Information**")
            col_vel1, col_vel2 = st.columns(2)
            with col_vel1:
                transactions_today = st.number_input("Transactions Today", min_value=0, value=8)
                amount_today = st.number_input("Total Amount Today ($)", min_value=0.0, value=45000.0)
            with col_vel2:
                transactions_week = st.number_input("Transactions This Week", min_value=0, value=25)
                amount_week = st.number_input("Total Amount This Week ($)", min_value=0.0, value=120000.0)

            # --- Account Info ---
            st.markdown("**Account Information**")
            col_acc1, col_acc2 = st.columns(2)
            with col_acc1:
                account_age_days = st.number_input("Account Age (Days)", min_value=0, value=180)
                account_balance = st.number_input("Account Balance ($)", min_value=0.0, value=25000.0)
            with col_acc2:
                kyc_status = st.selectbox("KYC Status", ["Complete", "Pending", "Incomplete"])
                customer_risk_rating = st.selectbox("Customer Risk Rating", ["Low", "Medium", "High"])

            # --- Flags ---
            st.markdown("**Special Conditions**")
            col_flag1, col_flag2 = st.columns(2)
            with col_flag1:
                is_cash_intensive = st.checkbox("Cash Intensive Business")
                is_high_risk_country = st.checkbox("High Risk Country Involved")
            with col_flag2:
                is_structuring = st.checkbox("Potential Structuring")
                is_round_amount = st.checkbox("Round Amount Transaction")

            submitted = st.form_submit_button("🔍 Run Transaction Screening")

        if submitted:
            transaction_data = {
                "transaction_id": transaction_id,
                "amount": amount,
                "transaction_type": transaction_type,
                "sender_country": sender_country,
                "sender_state": sender_state,
                "recipient_country": recipient_country,
                "recipient_state": recipient_state,
                "hour_of_day": hour_of_day,
                "is_weekend": is_weekend,
                "is_holiday": is_holiday,
                "is_after_hours": is_after_hours,
                "transactions_today": transactions_today,
                "amount_today": amount_today,
                "transactions_week": transactions_week,
                "amount_week": amount_week,
                "account_age_days": account_age_days,
                "account_balance": account_balance,
                "kyc_status": kyc_status,
                "customer_risk_rating": customer_risk_rating,
                "is_cash_intensive": is_cash_intensive,
                "is_high_risk_country": is_high_risk_country,
                "is_structuring": is_structuring,
                "is_round_amount": is_round_amount
            }

            with st.spinner("🚀 Sending data to rules engine..."):
                try:
                    res = requests.post("http://localhost:5000/transaction-screening", json=transaction_data)
                    if res.status_code == 200:
                        result = res.json()
                        st.session_state.results = {}
                        st.session_state.results['transaction_screening'] = result
                        st.success("✅ Transaction screening completed!")
                    else:
                        st.error(f"❌ API Error: {res.status_code} - {res.text}")
                except Exception as e:
                    st.error(f"❌ Failed to connect to backend: {e}")

    # ------- RIGHT COLUMN (Results) -------
    with col2:
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.markdown("### 🎯 Transaction Screening Results")

        if 'transaction_screening' in st.session_state.get('results', {}):
            result = st.session_state.results['transaction_screening']

            # Risk Score Gauge
            risk_score = result.get('risk_score', 0)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Rules Risk Score (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#f44336" if risk_score > 0.6 else "#ff9800" if risk_score > 0.3 else "#4caf50"},
                    'steps': [
                        {'range': [0, 30], 'color': "#e8f5e9"},
                        {'range': [30, 60], 'color': "#fff3e0"},
                        {'range': [60, 100], 'color': "#ffebee"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            # Summary Metrics
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Rules Triggered", result.get('total_rules_triggered', 0))
                st.metric("Requires Review", "Yes" if result.get('requires_review') else "No")
            with col_m2:
                st.metric("Auto Block", "Yes" if result.get('auto_block') else "No")
                st.metric("Compliance Alert", "Yes" if result.get('compliance_alert') else "No")

            # Triggered Rules
            st.markdown("#### 🚨 Triggered Rules")
            if result.get('triggered_rules'):
                for rule in result['triggered_rules']:
                    st.markdown(f"• ⚠️ {rule}")
            else:
                st.markdown("✅ No rules triggered")

            # Action Recommendations
            st.markdown("#### 💡 Recommended Actions")
            action = result.get('recommended_action', 'Approve')
            if action == 'Block':
                st.markdown('<div class="alert-high"><strong>🚫 BLOCK TRANSACTION</strong></div>',
                           unsafe_allow_html=True)
                st.markdown("• Transaction blocked automatically")
                st.markdown("• Generate SAR filing")
                st.markdown("• Escalate to compliance team")
            elif action == 'Review':
                st.markdown('<div class="alert-medium"><strong>⚠️ MANUAL REVIEW REQUIRED</strong></div>',
                           unsafe_allow_html=True)
                st.markdown("• Hold transaction for review")
                st.markdown("• Additional due diligence required")
                st.markdown("• Senior approval needed")
            else:
                st.markdown('<div class="alert-low"><strong>✅ APPROVED</strong></div>', unsafe_allow_html=True)
                st.markdown("• Process transaction normally")
                st.markdown("• Standard monitoring applies")

        else:
            st.info("Submit transaction for rules screening")
            st.markdown("#### 🔍 Rules Coverage")
            st.markdown("**Amount Thresholds:** $10K+ reporting")
            st.markdown("**Velocity Checks:** Transaction frequency")
            st.markdown("**Geographic Rules:** High-risk countries")
            st.markdown("**Timing Rules:** After hours, holidays")
            st.markdown("**Customer Rules:** KYC, risk ratings")
            st.markdown("**Pattern Rules:** Structuring detection")

if __name__ == "__main__" :
    main()