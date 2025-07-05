import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import json

# Configure page
st.set_page_config(
    page_title="FinGuardPro - Admin Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .block-container {
        background: white;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        padding: 2rem;
        margin-top: 2rem;
    }
    .metric-card, .success-card, .warning-card {
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .warning-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    .header-text {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .login-container {
        max-width: 400px;
        margin: 2rem auto;
        padding: 2rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Backend API URL
API_URL = "http://localhost:5000/api"

# Authentication functions
def validate_credentials(username, password):
    """Validate credentials with backend API"""
    try:
        response = requests.post(f"{API_URL}/auth/validate", 
                               json={"username": username, "password": password},
                               timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('valid', False), data.get('user', {})
        else:
            return False, {}
    except requests.exceptions.RequestException:
        return False, {}

def get_dashboard_data():
    """Get dashboard data from backend API"""
    try:
        response = requests.get(f"{API_URL}/dashboard/overview", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def get_system_stats():
    """Get system statistics from backend API"""
    try:
        response = requests.get(f"{API_URL}/admin/system-stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'user_data' not in st.session_state:
    st.session_state['user_data'] = {}

# Login Page
def show_login():
    st.markdown('<h1 class="header-text">🛡️ FinGuardPro Admin</h1>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        st.markdown("### 🔐 Admin Login")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if username and password:
                    valid, user_data = validate_credentials(username, password)
                    
                    if valid and user_data.get('role') in ['admin', 'compliance']:
                        st.session_state['authenticated'] = True
                        st.session_state['user_data'] = user_data
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials or insufficient permissions")
                else:
                    st.error("Please enter both username and password")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        ### 🔐 Demo Credentials  
        **Username:** admin  
        **Password:** admin123  
        **or**  
        **Username:** compliance  
        **Password:** admin123
        """)

# Main Dashboard
def show_dashboard():
    user_data = st.session_state.get('user_data', {})
    
    # Header
    st.markdown('<h1 class="header-text">🛡️ FinGuardPro Admin</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.success(f"Welcome, {user_data.get('name', 'Admin')}! 👋")
        st.markdown("---")
        
        # Navigation
        st.title("🎛️ Dashboard Navigation")
        page = st.selectbox("Select Page", 
                           ["🏠 Overview", "📊 Analytics", "👥 Users", "📈 Reports", "⚙️ Settings"])
        
        st.markdown("---")
        
        if st.button("🚪 Logout"):
            st.session_state['authenticated'] = False
            st.session_state['user_data'] = {}
            st.rerun()
    
    # Main content based on selected page
    if page == "🏠 Overview":
        show_overview()
    elif page == "📊 Analytics":
        show_analytics()
    elif page == "👥 Users":
        show_users()
    elif page == "📈 Reports":
        show_reports()
    elif page == "⚙️ Settings":
        show_settings()

def show_overview():
    st.markdown("## 📊 System Overview")
    
    # Get system stats
    stats = get_system_stats()
    
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'''
            <div class="success-card">
                <h3>👥 Total Users</h3>
                <p>{stats.get('total_users', 0)}</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="metric-card">
                <h3>🔍 Transactions</h3>
                <p>{stats.get('total_transactions', 0)}</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
            <div class="warning-card">
                <h3>🚨 Fraud Cases</h3>
                <p>{stats.get('total_fraud', 0)}</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            fraud_rate = stats.get('fraud_rate', 0)
            st.markdown(f'''
            <div class="metric-card">
                <h3>📈 Fraud Rate</h3>
                <p>{fraud_rate:.2f}%</p>
            </div>
            ''', unsafe_allow_html=True)
    else:
        st.error("❌ Unable to connect to backend API. Please ensure the API server is running.")
        st.info("To start the API server, run: `python api/api_server.py`")

def show_analytics():
    st.markdown("## 📊 Analytics Dashboard")
    
    # Generate some sample data for demonstration
    np.random.seed(42)
    
    # Sample data
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    transaction_counts = np.random.poisson(100, 30)
    fraud_counts = np.random.poisson(5, 30)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Daily Transaction Volume")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(dates, transaction_counts, marker='o', color='#4facfe', linewidth=2)
        ax.set_title('Daily Transaction Count')
        ax.set_xlabel('Date')
        ax.set_ylabel('Transactions')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.subheader("🚨 Daily Fraud Detection")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(dates, fraud_counts, color='#f5576c', alpha=0.7)
        ax.set_title('Daily Fraud Cases')
        ax.set_xlabel('Date')
        ax.set_ylabel('Fraud Cases')
        plt.xticks(rotation=45)
        st.pyplot(fig)

def show_users():
    st.markdown("## 👥 User Management")
    
    try:
        response = requests.get(f"{API_URL}/admin/users", timeout=10)
        if response.status_code == 200:
            users = response.json()
            
            # Display users table
            if users:
                df = pd.DataFrame(users)
                st.dataframe(df[['username', 'name', 'email', 'role', 'is_active']], 
                           use_container_width=True)
            else:
                st.info("No users found.")
        else:
            st.error("Failed to fetch users.")
    except requests.exceptions.RequestException:
        st.error("Unable to connect to backend API.")

def show_reports():
    st.markdown("## 📈 Reports")
    
    # Report date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2024, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    if st.button("Generate Report"):
        try:
            params = {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            }
            response = requests.get(f"{API_URL}/reports/summary", params=params, timeout=10)
            
            if response.status_code == 200:
                report_data = response.json()
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Transactions", report_data.get('total_transactions', 0))
                
                with col2:
                    st.metric("Total Amount", f"${report_data.get('total_amount', 0):,.2f}")
                
                with col3:
                    st.metric("Fraud Cases", report_data.get('fraud_transactions', 0))
                
                with col4:
                    avg_score = report_data.get('average_fraud_score', 0)
                    st.metric("Avg Fraud Score", f"{avg_score:.3f}")
                
                # Risk distribution
                risk_dist = report_data.get('risk_distribution', {})
                if risk_dist:
                    st.subheader("📊 Risk Distribution")
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    labels = list(risk_dist.keys())
                    values = list(risk_dist.values())
                    colors = ['#4CAF50', '#FF9800', '#F44336']
                    
                    ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                    ax.set_title('Risk Level Distribution')
                    st.pyplot(fig)
                
            else:
                st.error("Failed to generate report.")
        except requests.exceptions.RequestException:
            st.error("Unable to connect to backend API.")

def show_settings():
    st.markdown("## ⚙️ Settings")
    
    # Model settings
    st.subheader("🤖 Model Configuration")
    threshold = st.slider("Fraud Detection Threshold", 0.0, 1.0, 0.5, 0.01)
    st.info(f"Transactions with scores above {threshold:.2f} will be flagged as suspicious")
    
    # System settings
    st.subheader("🔧 System Configuration")
    auto_save = st.checkbox("Auto-save analysis results", True)
    notifications = st.checkbox("Enable email notifications", True)
    log_level = st.selectbox("Log Level", ["INFO", "WARNING", "ERROR", "DEBUG"])
    
    if st.button("💾 Save Settings"):
        st.success("✅ Settings saved successfully!")
        st.info(f"""
        **Current Settings:**
        - Fraud threshold: {threshold:.2f}
        - Auto-save: {'✅' if auto_save else '❌'}
        - Notifications: {'✅' if notifications else '❌'}
        - Log level: {log_level}
        """)

# Main app logic
if not st.session_state['authenticated']:
    show_login()
else:
    show_dashboard()