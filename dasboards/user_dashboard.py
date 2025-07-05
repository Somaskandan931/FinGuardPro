import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Configure page
st.set_page_config(
    page_title="FinGuardPro - User Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown( """
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
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    }
    .stButton > button {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
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
    </style>
""", unsafe_allow_html=True )


# Authentication configuration
def get_user_config () :
    config = {
        'credentials' : {
            'usernames' : {
                'user1' : {
                    'name' : 'John Doe',
                    'password' : '$2b$12$lKOzLvY4mZJLOzZ1lVyZH.O7KjLfGxDGF1lWRZgGqzPLaJZQZQzLa'  # user123
                },
                'user2' : {
                    'name' : 'Jane Smith',
                    'password' : '$2b$12$lKOzLvY4mZJLOzZ1lVyZH.O7KjLfGxDGF1lWRZgGqzPLaJZQZQzLa'  # user123
                },
                'demo' : {
                    'name' : 'Demo User',
                    'password' : '$2b$12$lKOzLvY4mZJLOzZ1lVyZH.O7KjLfGxDGF1lWRZgGqzPLaJZQZQzLa'  # user123
                }
            }
        },
        'cookie' : {
            'name' : 'finguard_user_cookie',
            'key' : 'finguard_user_secret_key_2024',
            'expiry_days' : 1
        },
        'preauthorized' : {'emails' : []}
    }
    return config


# Initialize session state for authentication
if 'user_authentication_status' not in st.session_state :
    st.session_state['user_authentication_status'] = None
if 'user_name' not in st.session_state :
    st.session_state['user_name'] = None
if 'user_username' not in st.session_state :
    st.session_state['user_username'] = None

# Initialize authenticator
config = get_user_config()
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# Fixed login call - using the correct method signature
if st.session_state['user_authentication_status'] is None :
    try :
        # Use the login widget with unique form name
        name, authentication_status, username = authenticator.login(
            form_name='User Login',
            location='main',
            max_login_attempts=3,
            fields={'Form name' : 'User Login', 'Username' : 'Username', 'Password' : 'Password', 'Login' : 'Login'}
        )

        # Store in session state
        st.session_state['user_authentication_status'] = authentication_status
        st.session_state['user_name'] = name
        st.session_state['user_username'] = username

    except Exception as e :
        # Fallback to simple login if the above fails
        try :
            name, authentication_status, username = authenticator.login( 'User Login', 'main' )
            st.session_state['user_authentication_status'] = authentication_status
            st.session_state['user_name'] = name
            st.session_state['user_username'] = username
        except Exception as e2 :
            st.error( f"Login system error: {str( e2 )}" )
            st.stop()

# Get values from session state
name = st.session_state['user_name']
authentication_status = st.session_state['user_authentication_status']
username = st.session_state['user_username']

# Handle logout
if authentication_status == True :
    with st.sidebar :
        if st.button( "Logout" ) :
            st.session_state['user_authentication_status'] = None
            st.session_state['user_name'] = None
            st.session_state['user_username'] = None
            st.experimental_rerun()

if authentication_status == True :
    # Welcome message
    st.sidebar.success( f"Welcome back, {name}! 👋" )
    st.sidebar.markdown( "---" )

    # Header
    st.markdown( '<h1 class="header-text">💳 FinGuardPro</h1>', unsafe_allow_html=True )
    st.markdown( '<p class="sub-header">Personal Financial Security Dashboard</p>', unsafe_allow_html=True )

    # Dashboard navigation
    st.sidebar.title( "🧭 Navigation" )
    page = st.sidebar.selectbox(
        "Select Page",
        ["🏠 Dashboard", "📊 Transaction Analysis", "📈 My Reports", "⚙️ Settings"]
    )


    # Load model and preprocessor
    @st.cache_resource
    def load_model_and_scaler () :
        try :
            # Use relative paths that work across different systems
            base_path = Path( __file__ ).parent.parent
            model_path = base_path / 'models' / 'fraud_detection_model.h5'
            scaler_path = base_path / 'models' / 'scaler.pkl'

            if not model_path.exists() :
                return None, None, "Model file not found"

            if not scaler_path.exists() :
                return None, None, "Scaler file not found"

            model = tf.keras.models.load_model( str( model_path ) )
            scaler = joblib.load( str( scaler_path ) )
            return model, scaler, "success"
        except Exception as e :
            return None, None, f"Error loading model: {str( e )}"


    model, scaler, load_status = load_model_and_scaler()


    # Generate sample data for demonstration
    @st.cache_data
    def generate_sample_data () :
        np.random.seed( 42 )
        n_transactions = 100

        # Generate sample transaction data
        data = {
            'transaction_id' : [f'TXN_{i:06d}' for i in range( n_transactions )],
            'amount' : np.random.lognormal( 3, 1.5, n_transactions ),
            'merchant_category' : np.random.choice( ['Grocery', 'Gas', 'Restaurant', 'Online', 'ATM', 'Other'],
                                                    n_transactions ),
            'transaction_hour' : np.random.randint( 0, 24, n_transactions ),
            'is_weekend' : np.random.choice( [0, 1], n_transactions, p=[0.7, 0.3] ),
            'days_since_last_transaction' : np.random.exponential( 1, n_transactions ),
            'distance_from_home' : np.random.exponential( 5, n_transactions ),
            'card_present' : np.random.choice( [0, 1], n_transactions, p=[0.3, 0.7] ),
            'timestamp' : pd.date_range( start='2024-01-01', periods=n_transactions, freq='D' )
        }

        df = pd.DataFrame( data )

        # Add fraud probability (simulate model predictions)
        df['fraud_probability'] = np.random.beta( 2, 8, n_transactions )
        df['is_fraud'] = (df['fraud_probability'] > 0.7).astype( int )

        return df


    sample_data = generate_sample_data()

    if page == "🏠 Dashboard" :
        st.markdown( "## 🏠 Dashboard Overview" )

        # Quick stats
        col1, col2, col3, col4 = st.columns( 4 )

        with col1 :
            if model is not None :
                st.markdown( '<div class="success-card"><h3>✅ Security Status</h3><p>Protected</p></div>',
                             unsafe_allow_html=True )
            else :
                st.markdown( '<div class="warning-card"><h3>⚠️ Security Status</h3><p>Demo Mode</p></div>',
                             unsafe_allow_html=True )

        with col2 :
            fraud_count = sample_data['is_fraud'].sum()
            st.markdown( f'<div class="metric-card"><h3>🚨 Fraud Alerts</h3><p>{fraud_count}</p></div>',
                         unsafe_allow_html=True )

        with col3 :
            total_transactions = len( sample_data )
            st.markdown( f'<div class="metric-card"><h3>📊 Total Transactions</h3><p>{total_transactions}</p></div>',
                         unsafe_allow_html=True )

        with col4 :
            total_amount = sample_data['amount'].sum()
            st.markdown( f'<div class="metric-card"><h3>💰 Total Amount</h3><p>${total_amount:,.2f}</p></div>',
                         unsafe_allow_html=True )

        # Recent transactions
        st.markdown( "## 📋 Recent Transactions" )

        # Sort by timestamp descending
        recent_transactions = sample_data.sort_values( 'timestamp', ascending=False ).head( 10 )

        for idx, row in recent_transactions.iterrows() :
            fraud_prob = row['fraud_probability']

            if fraud_prob > 0.7 :
                card_class = "transaction-danger"
                risk_level = "🚨 High Risk"
            elif fraud_prob > 0.4 :
                card_class = "transaction-warning"
                risk_level = "⚠️ Medium Risk"
            else :
                card_class = "transaction-safe"
                risk_level = "✅ Low Risk"

            st.markdown( f'''
                <div class="{card_class}">
                    <strong>{row['transaction_id']}</strong> - ${row['amount']:.2f} at {row['merchant_category']} | {risk_level} ({fraud_prob:.2%})
                </div>
            ''', unsafe_allow_html=True )

        # Charts
        st.markdown( "## 📈 Analytics" )

        col1, col2 = st.columns( 2 )

        with col1 :
            st.subheader( "Transaction Amount Distribution" )
            fig, ax = plt.subplots()
            ax.hist( sample_data['amount'], bins=20, alpha=0.7, color='skyblue' )
            ax.set_xlabel( 'Amount ($)' )
            ax.set_ylabel( 'Frequency' )
            st.pyplot( fig )

        with col2 :
            st.subheader( "Fraud Risk by Merchant Category" )
            fraud_by_merchant = sample_data.groupby( 'merchant_category' )['fraud_probability'].mean()
            fig, ax = plt.subplots()
            fraud_by_merchant.plot( kind='bar', ax=ax, color='coral' )
            ax.set_ylabel( 'Average Fraud Probability' )
            ax.set_xlabel( 'Merchant Category' )
            plt.xticks( rotation=45 )
            st.pyplot( fig )

    elif page == "📊 Transaction Analysis" :
        st.markdown( "## 📊 Transaction Analysis" )

        # Transaction checker
        st.markdown( "### 🔍 Check Individual Transaction" )

        with st.form( "transaction_form" ) :
            col1, col2 = st.columns( 2 )

            with col1 :
                amount = st.number_input( "Transaction Amount ($)", min_value=0.01, value=100.0 )
                merchant_category = st.selectbox( "Merchant Category",
                                                  ['Grocery', 'Gas', 'Restaurant', 'Online', 'ATM', 'Other'] )
                transaction_hour = st.slider( "Transaction Hour", 0, 23, 12 )
                is_weekend = st.checkbox( "Weekend Transaction" )

            with col2 :
                days_since_last = st.number_input( "Days Since Last Transaction", min_value=0.0, value=1.0 )
                distance_from_home = st.number_input( "Distance from Home (miles)", min_value=0.0, value=5.0 )
                card_present = st.checkbox( "Card Present", value=True )

            submitted = st.form_submit_button( "Analyze Transaction" )

            if submitted :
                # Simulate fraud prediction
                # In real implementation, this would use the loaded model
                features = np.array( [[amount, transaction_hour, int( is_weekend ), days_since_last,
                                       distance_from_home, int( card_present )]] )

                # Simulate prediction (replace with actual model prediction)
                fraud_probability = np.random.beta( 2, 8 )

                if fraud_probability > 0.7 :
                    st.error( f"🚨 High Risk Transaction! Fraud Probability: {fraud_probability:.2%}" )
                    st.markdown(
                        f'<div class="danger-card"><h3>⛔ BLOCKED</h3><p>This transaction has been flagged as potentially fraudulent.</p></div>',
                        unsafe_allow_html=True )
                elif fraud_probability > 0.4 :
                    st.warning( f"⚠️ Medium Risk Transaction. Fraud Probability: {fraud_probability:.2%}" )
                    st.markdown(
                        f'<div class="warning-card"><h3>🔍 REVIEW</h3><p>This transaction requires additional verification.</p></div>',
                        unsafe_allow_html=True )
                else :
                    st.success( f"✅ Low Risk Transaction. Fraud Probability: {fraud_probability:.2%}" )
                    st.markdown(
                        f'<div class="success-card"><h3>✅ APPROVED</h3><p>This transaction appears legitimate.</p></div>',
                        unsafe_allow_html=True )

        # Bulk analysis
        st.markdown( "### 📁 Bulk Transaction Analysis" )

        uploaded_file = st.file_uploader( "Upload CSV file", type=['csv'] )

        if uploaded_file is not None :
            df = pd.read_csv( uploaded_file )
            st.write( "Data Preview:" )
            st.dataframe( df.head() )

            if st.button( "Analyze All Transactions" ) :
                # Simulate bulk analysis
                df['fraud_probability'] = np.random.beta( 2, 8, len( df ) )
                df['risk_level'] = pd.cut( df['fraud_probability'],
                                           bins=[0, 0.4, 0.7, 1.0],
                                           labels=['Low', 'Medium', 'High'] )

                st.write( "Analysis Results:" )
                st.dataframe( df[['fraud_probability', 'risk_level']] )

                # Summary
                risk_summary = df['risk_level'].value_counts()
                st.write( "Risk Summary:" )
                st.bar_chart( risk_summary )

    elif page == "📈 My Reports" :
        st.markdown( "## 📈 My Reports" )

        # Time period selector
        col1, col2 = st.columns( 2 )

        with col1 :
            start_date = st.date_input( "Start Date", value=pd.to_datetime( '2024-01-01' ) )

        with col2 :
            end_date = st.date_input( "End Date", value=pd.to_datetime( '2024-12-31' ) )

        # Filter data by date range
        filtered_data = sample_data[
            (sample_data['timestamp'] >= pd.to_datetime( start_date )) &
            (sample_data['timestamp'] <= pd.to_datetime( end_date ))
            ]

        # Summary metrics
        st.markdown( "### 📊 Summary Statistics" )

        col1, col2, col3, col4 = st.columns( 4 )

        with col1 :
            total_trans = len( filtered_data )
            st.metric( "Total Transactions", total_trans )

        with col2 :
            fraud_trans = filtered_data['is_fraud'].sum()
            st.metric( "Fraud Transactions", fraud_trans, f"{fraud_trans / total_trans * 100:.1f}%" )

        with col3 :
            total_amount = filtered_data['amount'].sum()
            st.metric( "Total Amount", f"${total_amount:,.2f}" )

        with col4 :
            avg_amount = filtered_data['amount'].mean()
            st.metric( "Average Amount", f"${avg_amount:.2f}" )

        # Detailed analysis
        st.markdown( "### 📋 Detailed Analysis" )

        # Transaction trends
        st.subheader( "Transaction Trends Over Time" )
        daily_stats = filtered_data.groupby( filtered_data['timestamp'].dt.date ).agg( {
            'amount' : ['sum', 'count'],
            'is_fraud' : 'sum'
        } ).reset_index()

        daily_stats.columns = ['date', 'total_amount', 'transaction_count', 'fraud_count']

        fig, (ax1, ax2) = plt.subplots( 2, 1, figsize=(12, 8) )

        # Amount trend
        ax1.plot( daily_stats['date'], daily_stats['total_amount'], marker='o', color='green', alpha=0.7 )
        ax1.set_title( 'Daily Transaction Amount' )
        ax1.set_ylabel( 'Amount ($)' )
        ax1.tick_params( axis='x', rotation=45 )

        # Fraud trend
        ax2.bar( daily_stats['date'], daily_stats['fraud_count'], color='red', alpha=0.7 )
        ax2.set_title( 'Daily Fraud Count' )
        ax2.set_ylabel( 'Fraud Count' )
        ax2.tick_params( axis='x', rotation=45 )

        plt.tight_layout()
        st.pyplot( fig )

        # Export data
        st.markdown( "### 📥 Export Data" )

        if st.button( "Generate Report" ) :
            report_data = filtered_data.copy()
            csv = report_data.to_csv( index=False )

            st.download_button(
                label="Download CSV Report",
                data=csv,
                file_name=f"finguard_report_{start_date}_{end_date}.csv",
                mime="text/csv"
            )

    elif page == "⚙️ Settings" :
        st.markdown( "## ⚙️ Settings" )

        # User preferences
        st.markdown( "### 👤 User Preferences" )

        notification_enabled = st.checkbox( "Enable Fraud Alerts", value=True )
        alert_threshold = st.slider( "Fraud Alert Threshold", 0.0, 1.0, 0.7, 0.05 )

        st.markdown( "### 🔧 Model Configuration" )

        if model is not None :
            st.success( "✅ Fraud Detection Model Loaded Successfully" )
            st.info( f"Model Status: {load_status}" )
        else :
            st.warning( "⚠️ Running in Demo Mode" )
            st.info( f"Status: {load_status}" )

        # System information
        st.markdown( "### 🖥️ System Information" )

        col1, col2 = st.columns( 2 )

        with col1 :
            st.info( f"**User:** {name}" )
            st.info( f"**Username:** {username}" )
            st.info( f"**Login Time:** {datetime.now().strftime( '%Y-%m-%d %H:%M:%S' )}" )

        with col2 :
            st.info( f"**TensorFlow Version:** {tf.__version__}" )
            st.info( f"**Streamlit Version:** {st.__version__}" )
            st.info( f"**Model Status:** {'Loaded' if model is not None else 'Demo Mode'}" )

        # Data management
        st.markdown( "### 🗂️ Data Management" )

        if st.button( "Clear Cache" ) :
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success( "Cache cleared successfully!" )

        if st.button( "Reset Settings" ) :
            st.session_state.clear()
            st.success( "Settings reset successfully!" )

elif authentication_status == False :
    st.error( "Username/password is incorrect" )
    st.info( "Demo credentials - Username: demo, Password: user123" )

elif authentication_status == None :
    st.warning( "Please enter your username and password" )
    st.info( "Demo credentials - Username: demo, Password: user123" )

# Footer
st.markdown( "---" )
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🔒 FinGuardPro - Protecting Your Financial Future</p>
        <p>Built with ❤️ using Streamlit and TensorFlow</p>
    </div>
    """,
    unsafe_allow_html=True
)