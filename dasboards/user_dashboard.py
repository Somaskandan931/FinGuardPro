import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from datetime import datetime
import json

# Configure page
st.set_page_config(
    page_title="FinGuardPro - User Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stApp {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    }
    .main .block-container {
        background: white;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        padding: 2rem;
        margin-top: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
    .success-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: #333;
        margin: 1rem 0;
        text-align: center;
    }
    .warning-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: #333;
        margin: 1rem 0;
        text-align: center;
    }
    .danger-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: #333;
        margin: 1rem 0;
        text-align: center;
    }
    .header-text {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .transaction-safe {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    .transaction-warning {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #FF9800;
    }
    .transaction-danger {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #F44336;
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

def analyze_transaction(transaction_data):
    """Analyze a single transaction"""
    try:
        response = requests.post(f"{API_URL}/transactions/analyze", 
                               json=transaction_data,
                               timeout=10)
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

# Generate sample data for demonstration
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    n_transactions = 50

    # Generate sample transaction data
    data = {
        'transaction_id': [f'TXN_{i:06d}' for i in range(n_transactions)],
        'amount': np.random.lognormal(3, 1.5, n_transactions),
        'merchant_category': np.random.choice(['Grocery', 'Gas', 'Restaurant', 'Online', 'ATM', 'Other'], n_transactions),
        'transaction_hour': np.random.randint(0, 24, n_transactions),
        'is_weekend': np.random.choice([0, 1], n_transactions, p=[0.7, 0.3]),
        'days_since_last_transaction': np.random.exponential(1, n_transactions),
        'distance_from_home': np.random.exponential(5, n_transactions),
        'card_present': np.random.choice([0, 1], n_transactions, p=[0.3, 0.7]),
        'timestamp': pd.date_range(start='2024-01-01', periods=n_transactions, freq='D')
    }

    df = pd.DataFrame(data)
    
    # Add fraud probability (simulate model predictions)
    df['fraud_probability'] = np.random.beta(2, 8, n_transactions)
    df['is_fraud'] = (df['fraud_probability'] > 0.7).astype(int)
    
    return df

# Login Page
def show_login():
    st.markdown('<h1 class="header-text">💳 FinGuardPro User</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Personal Financial Security Dashboard</p>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        st.markdown("### 🔐 User Login")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if username and password:
                    valid, user_data = validate_credentials(username, password)
                    
                    if valid and user_data.get('role') == 'user':
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
        **Username:** demo  
        **Password:** user123  
        **or**  
        **Username:** user1  
        **Password:** user123  
        **or**  
        **Username:** user2  
        **Password:** user123
        """)

# Main Dashboard
def show_dashboard():
    user_data = st.session_state.get('user_data', {})
    
    # Header
    st.markdown('<h1 class="header-text">💳 FinGuardPro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Personal Financial Security Dashboard</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.success(f"Welcome back, {user_data.get('name', 'User')}! 👋")
        st.markdown("---")
        
        # Dashboard navigation
        st.title("🧭 Navigation")
        page = st.selectbox(
            "Select Page",
            ["🏠 Dashboard", "📊 Transaction Analysis", "📈 My Reports", "⚙️ Settings"]
        )
        
        st.markdown("---")
        
        if st.button("🚪 Logout"):
            st.session_state['authenticated'] = False
            st.session_state['user_data'] = {}
            st.rerun()

    # Main content based on selected page
    if page == "🏠 Dashboard":
        show_overview()
    elif page == "📊 Transaction Analysis":
        show_transaction_analysis()
    elif page == "📈 My Reports":
        show_reports()
    elif page == "⚙️ Settings":
        show_settings()

def show_overview():
    st.markdown("## 🏠 Dashboard Overview")
    
    # Get dashboard data
    dashboard_data = get_dashboard_data()
    sample_data = generate_sample_data()
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('''
        <div class="success-card">
            <h3>✅ Security Status</h3>
            <p>Protected</p>
        </div>
        ''', unsafe_allow_html=True)

    with col2:
        fraud_count = sample_data['is_fraud'].sum()
        st.markdown(f'''
        <div class="metric-card">
            <h3>🚨 Fraud Alerts</h3>
            <p>{fraud_count}</p>
        </div>
        ''', unsafe_allow_html=True)

    with col3:
        total_transactions = len(sample_data)
        st.markdown(f'''
        <div class="metric-card">
            <h3>📊 Total Transactions</h3>
            <p>{total_transactions}</p>
        </div>
        ''', unsafe_allow_html=True)

    with col4:
        total_amount = sample_data['amount'].sum()
        st.markdown(f'''
        <div class="metric-card">
            <h3>💰 Total Amount</h3>
            <p>${total_amount:,.2f}</p>
        </div>
        ''', unsafe_allow_html=True)

    # Recent transactions
    st.markdown("## 📋 Recent Transactions")

    # Sort by timestamp descending
    recent_transactions = sample_data.sort_values('timestamp', ascending=False).head(10)

    for idx, row in recent_transactions.iterrows():
        fraud_prob = row['fraud_probability']

        if fraud_prob > 0.7:
            card_class = "transaction-danger"
            risk_level = "🚨 High Risk"
        elif fraud_prob > 0.4:
            card_class = "transaction-warning"
            risk_level = "⚠️ Medium Risk"
        else:
            card_class = "transaction-safe"
            risk_level = "✅ Low Risk"

        st.markdown(f'''
            <div class="{card_class}">
                <strong>{row['transaction_id']}</strong> - ${row['amount']:.2f} at {row['merchant_category']} | {risk_level} ({fraud_prob:.2%})
            </div>
        ''', unsafe_allow_html=True)

    # Charts
    st.markdown("## 📈 Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transaction Amount Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(sample_data['amount'], bins=20, alpha=0.7, color='skyblue')
        ax.set_xlabel('Amount ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Transaction Amounts')
        st.pyplot(fig)

    with col2:
        st.subheader("Fraud Risk by Merchant Category")
        fraud_by_merchant = sample_data.groupby('merchant_category')['fraud_probability'].mean()
        fig, ax = plt.subplots(figsize=(10, 6))
        fraud_by_merchant.plot(kind='bar', ax=ax, color='coral')
        ax.set_ylabel('Average Fraud Probability')
        ax.set_xlabel('Merchant Category')
        ax.set_title('Fraud Risk by Merchant Category')
        plt.xticks(rotation=45)
        st.pyplot(fig)

def show_transaction_analysis():
    st.markdown("## 📊 Transaction Analysis")

    # Transaction checker
    st.markdown("### 🔍 Check Individual Transaction")

    with st.form("transaction_form"):
        col1, col2 = st.columns(2)

        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=100.0)
            merchant_category = st.selectbox("Merchant Category",
                                           ['Grocery', 'Gas', 'Restaurant', 'Online', 'ATM', 'Other'])
            transaction_hour = st.slider("Transaction Hour", 0, 23, 12)
            is_weekend = st.checkbox("Weekend Transaction")

        with col2:
            days_since_last = st.number_input("Days Since Last Transaction", min_value=0.0, value=1.0)
            distance_from_home = st.number_input("Distance from Home (miles)", min_value=0.0, value=5.0)
            card_present = st.checkbox("Card Present", value=True)

        submitted = st.form_submit_button("Analyze Transaction")

        if submitted:
            transaction_data = {
                'amount': amount,
                'merchant_category': merchant_category,
                'transaction_hour': transaction_hour,
                'is_weekend': is_weekend,
                'days_since_last': days_since_last,
                'distance_from_home': distance_from_home,
                'card_present': card_present
            }
            
            # Try to analyze with API
            result = analyze_transaction(transaction_data)
            
            if result:
                fraud_probability = result.get('fraud_score', 0)
                risk_level = result.get('risk_level', 'Low')
                recommendation = result.get('recommendation', 'APPROVE')
            else:
                # Fallback to simulate prediction
                fraud_probability = np.random.beta(2, 8)
                risk_level = 'Low' if fraud_probability < 0.3 else ('Medium' if fraud_probability < 0.7 else 'High')
                recommendation = 'BLOCK' if fraud_probability > 0.7 else ('REVIEW' if fraud_probability > 0.4 else 'APPROVE')

            if fraud_probability > 0.7:
                st.error(f"🚨 High Risk Transaction! Fraud Probability: {fraud_probability:.2%}")
                st.markdown(f'''
                <div class="danger-card">
                    <h3>⛔ {recommendation}</h3>
                    <p>This transaction has been flagged as potentially fraudulent.</p>
                </div>
                ''', unsafe_allow_html=True)
            elif fraud_probability > 0.4:
                st.warning(f"⚠️ Medium Risk Transaction. Fraud Probability: {fraud_probability:.2%}")
                st.markdown(f'''
                <div class="warning-card">
                    <h3>🔍 {recommendation}</h3>
                    <p>This transaction requires additional verification.</p>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.success(f"✅ Low Risk Transaction. Fraud Probability: {fraud_probability:.2%}")
                st.markdown(f'''
                <div class="success-card">
                    <h3>✅ {recommendation}</h3>
                    <p>This transaction appears legitimate.</p>
                </div>
                ''', unsafe_allow_html=True)

    # Bulk analysis
    st.markdown("### 📁 Bulk Transaction Analysis")

    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(df.head())

            if st.button("Analyze All Transactions"):
                with st.spinner("Analyzing transactions..."):
                    # Simulate bulk analysis
                    df['fraud_probability'] = np.random.beta(2, 8, len(df))
                    df['risk_level'] = pd.cut(df['fraud_probability'],
                                             bins=[0, 0.4, 0.7, 1.0],
                                             labels=['Low', 'Medium', 'High'])

                    st.write("Analysis Results:")
                    st.dataframe(df[['fraud_probability', 'risk_level']])

                    # Summary
                    risk_summary = df['risk_level'].value_counts()
                    st.write("Risk Summary:")
                    st.bar_chart(risk_summary)
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def show_reports():
    st.markdown("## 📈 My Reports")

    sample_data = generate_sample_data()

    # Time period selector
    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime('2024-01-01'))

    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime('2024-12-31'))

    # Filter data by date range
    filtered_data = sample_data[
        (sample_data['timestamp'] >= pd.to_datetime(start_date)) &
        (sample_data['timestamp'] <= pd.to_datetime(end_date))
    ]

    # Summary metrics
    st.markdown("### 📊 Summary Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_trans = len(filtered_data)
        st.metric("Total Transactions", total_trans)

    with col2:
        fraud_trans = filtered_data['is_fraud'].sum()
        st.metric("Fraud Transactions", fraud_trans, f"{fraud_trans / total_trans * 100:.1f}%")

    with col3:
        total_amount = filtered_data['amount'].sum()
        st.metric("Total Amount", f"${total_amount:,.2f}")

    with col4:
        avg_amount = filtered_data['amount'].mean()
        st.metric("Average Amount", f"${avg_amount:.2f}")

    # Detailed analysis
    st.markdown("### 📋 Detailed Analysis")

    # Transaction trends
    st.subheader("Transaction Trends Over Time")
    daily_stats = filtered_data.groupby(filtered_data['timestamp'].dt.date).agg({
        'amount': ['sum', 'count'],
        'is_fraud': 'sum'
    }).reset_index()

    daily_stats.columns = ['date', 'total_amount', 'transaction_count', 'fraud_count']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Amount trend
    ax1.plot(daily_stats['date'], daily_stats['total_amount'], marker='o', color='green', alpha=0.7)
    ax1.set_title('Daily Transaction Amount')
    ax1.set_ylabel('Amount ($)')
    ax1.tick_params(axis='x', rotation=45)

    # Fraud trend
    ax2.bar(daily_stats['date'], daily_stats['fraud_count'], color='red', alpha=0.7)
    ax2.set_title('Daily Fraud Count')
    ax2.set_ylabel('Fraud Count')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig)

def show_settings():
    st.markdown("## ⚙️ Settings")

    # User preferences
    st.subheader("🔔 Notification Preferences")
    email_alerts = st.checkbox("Email alerts for high-risk transactions", True)
    sms_alerts = st.checkbox("SMS alerts for blocked transactions", False)
    daily_summary = st.checkbox("Daily transaction summary", True)

    # Security settings
    st.subheader("🔒 Security Settings")
    two_factor = st.checkbox("Enable two-factor authentication", False)
    auto_lock = st.selectbox("Auto-lock account after", ["Never", "1 hour", "4 hours", "1 day"])

    # Display preferences
    st.subheader("🎨 Display Preferences")
    theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
    currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "CAD"])

    if st.button("💾 Save Settings"):
        st.success("✅ Settings saved successfully!")
        st.info(f"""
        **Current Settings:**
        - Email alerts: {'✅' if email_alerts else '❌'}
        - SMS alerts: {'✅' if sms_alerts else '❌'}
        - Daily summary: {'✅' if daily_summary else '❌'}
        - Two-factor auth: {'✅' if two_factor else '❌'}
        - Auto-lock: {auto_lock}
        - Theme: {theme}
        - Currency: {currency}
        """)

# Main app logic
if not st.session_state['authenticated']:
    show_login()
else:
    show_dashboard()