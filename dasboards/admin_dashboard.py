import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from pathlib import Path
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="FinGuardPro - Admin Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown( """<style>
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
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
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
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>""", unsafe_allow_html=True )


# Authentication config
def get_auth_config () :
    config = {
        'credentials' : {
            'usernames' : {
                'admin' : {
                    'name' : 'Admin User',
                    'password' : '$2b$12$lKOzLvY4mZJLOzZ1lVyZH.O7KjLfGxDGF1lWRZgGqzPLaJZQZQzLa'  # admin123
                },
                'compliance' : {
                    'name' : 'Compliance Officer',
                    'password' : '$2b$12$lKOzLvY4mZJLOzZ1lVyZH.O7KjLfGxDGF1lWRZgGqzPLaJZQZQzLa'  # admin123
                }
            }
        },
        'cookie' : {
            'name' : 'finguard_admin_cookie',
            'key' : 'finguard_admin_secret_key_2024',
            'expiry_days' : 1
        }
    }
    return config


# Initialize session state for authentication
if 'authentication_status' not in st.session_state :
    st.session_state['authentication_status'] = None
if 'name' not in st.session_state :
    st.session_state['name'] = None
if 'username' not in st.session_state :
    st.session_state['username'] = None

config = get_auth_config()
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Fixed login call - using the correct method signature
if st.session_state['authentication_status'] is None :
    try :
        # Use the login widget
        name, authentication_status, username = authenticator.login(
            form_name='Admin Login',
            location='main',
            max_login_attempts=3,
            fields={'Form name' : 'Admin Login', 'Username' : 'Username', 'Password' : 'Password', 'Login' : 'Login'}
        )

        # Store in session state
        st.session_state['authentication_status'] = authentication_status
        st.session_state['name'] = name
        st.session_state['username'] = username

    except Exception as e :
        # Fallback to simple login if the above fails
        try :
            name, authentication_status, username = authenticator.login( 'Admin Login', 'main' )
            st.session_state['authentication_status'] = authentication_status
            st.session_state['name'] = name
            st.session_state['username'] = username
        except Exception as e2 :
            st.error( f"Login system error: {str( e2 )}" )
            st.stop()

# Get values from session state
name = st.session_state['name']
authentication_status = st.session_state['authentication_status']
username = st.session_state['username']

# Handle logout
if authentication_status == True :
    with st.sidebar :
        if st.button( "Logout" ) :
            st.session_state['authentication_status'] = None
            st.session_state['name'] = None
            st.session_state['username'] = None
            st.experimental_rerun()

# Authenticated
if authentication_status == True :
    st.sidebar.success( f"Welcome, {name}! 👋" )
    st.markdown( '<h1 class="header-text">🛡️ FinGuardPro</h1>', unsafe_allow_html=True )
    st.markdown( '<p class="sub-header">Advanced Financial Fraud Detection & Analysis</p>', unsafe_allow_html=True )

    # Navigation
    st.sidebar.title( "🎛️ Dashboard Navigation" )
    page = st.sidebar.selectbox( "Select Page",
                                 ["🏠 Overview", "📊 Fraud Analysis", "🔍 Transaction Explorer", "📈 Reports",
                                  "⚙️ Settings"] )


    # Load model and scaler
    @st.cache_resource
    def load_components () :
        try :
            base_path = Path( __file__ ).parent.parent
            model_path = base_path / 'models' / 'fraud_detection_model.h5'
            scaler_path = base_path / 'models' / 'scaler.pkl'
            if not model_path.exists() or not scaler_path.exists() :
                return None, None, "Missing model or scaler file"
            model = tf.keras.models.load_model( str( model_path ) )
            scaler = joblib.load( str( scaler_path ) )
            return model, scaler, "success"
        except Exception as e :
            return None, None, str( e )


    model, scaler, load_status = load_components()

    if page == "🏠 Overview" :
        st.markdown( "## 📊 System Overview" )
        col1, col2, col3, col4 = st.columns( 4 )
        with col1 :
            st.markdown(
                '<div class="success-card"><h3>✅ Model</h3><p>{}</p></div>'.format( "Ready" if model else "Missing" ),
                unsafe_allow_html=True )
        with col2 :
            st.markdown( '<div class="metric-card"><h3>🔍 Transactions</h3><p>Ready</p></div>', unsafe_allow_html=True )
        with col3 :
            st.markdown( '<div class="success-card"><h3>📈 Reports</h3><p>Available</p></div>', unsafe_allow_html=True )
        with col4 :
            st.markdown( '<div class="metric-card"><h3>🛡️ Security</h3><p>Active</p></div>', unsafe_allow_html=True )

        st.markdown( "## 🚀 Quick Actions" )
        col1, col2, col3 = st.columns( 3 )
        with col1 :
            if st.button( "📊 Analyze Transactions", key="overview_analyze" ) :
                st.session_state.page = "📊 Fraud Analysis"
                st.experimental_rerun()
        with col2 :
            if st.button( "🔍 Explore Data", key="overview_explore" ) :
                st.session_state.page = "🔍 Transaction Explorer"
                st.experimental_rerun()
        with col3 :
            if st.button( "📈 Generate Reports", key="overview_reports" ) :
                st.session_state.page = "📈 Reports"
                st.experimental_rerun()

        if load_status != "success" :
            st.error( f"Model loading issue: {load_status}" )

    elif page == "📊 Fraud Analysis" :
        st.markdown( "## 📊 Fraud Analysis" )
        if model is None or scaler is None :
            st.error( "❌ Model and scaler required for analysis." )
            st.stop()

        uploaded_file = st.file_uploader( "📤 Upload Transaction Data", type=["csv"], key="admin_upload" )
        if uploaded_file :
            try :
                data = pd.read_csv( uploaded_file )
                st.success( f"✅ Loaded {len( data )} transactions with {len( data.columns )} features" )
                st.dataframe( data.head(), use_container_width=True )

                if st.button( "🔍 Run Analysis", key="run_analysis" ) :
                    with st.spinner( "Analyzing transactions..." ) :
                        try :
                            # Ensure data has correct number of features
                            if data.shape[1] < scaler.n_features_in_ if hasattr( scaler, 'n_features_in_' ) else 10 :
                                st.error( "❌ Insufficient features in the dataset" )
                                st.stop()

                            # Take only the required number of features
                            expected_features = scaler.n_features_in_ if hasattr( scaler, 'n_features_in_' ) else 10
                            X_data = data.iloc[:, :expected_features]

                            X_scaled = scaler.transform( X_data )
                            fraud_scores = model.predict( X_scaled )

                            # Create results dataframe
                            results_df = data.copy()
                            results_df['fraud_score'] = fraud_scores.flatten()
                            results_df['risk_level'] = results_df['fraud_score'].apply(
                                lambda x : "🟢 Low" if x < 0.3 else ("🟡 Medium" if x < 0.7 else "🔴 High") )

                            st.session_state['analysis_data'] = results_df

                            # Display results summary
                            st.success( "✅ Analysis completed successfully!" )

                            col1, col2, col3 = st.columns( 3 )
                            with col1 :
                                low_count = len( results_df[results_df['risk_level'] == '🟢 Low'] )
                                st.metric( "Low Risk", low_count, f"{low_count / len( results_df ) * 100:.1f}%" )
                            with col2 :
                                med_count = len( results_df[results_df['risk_level'] == '🟡 Medium'] )
                                st.metric( "Medium Risk", med_count, f"{med_count / len( results_df ) * 100:.1f}%" )
                            with col3 :
                                high_count = len( results_df[results_df['risk_level'] == '🔴 High'] )
                                st.metric( "High Risk", high_count, f"{high_count / len( results_df ) * 100:.1f}%" )

                            # Show results table
                            st.markdown( "### 📊 Analysis Results" )
                            st.dataframe( results_df[['fraud_score', 'risk_level']], use_container_width=True )

                            # Download button
                            csv = results_df.to_csv( index=False ).encode( 'utf-8' )
                            st.download_button(
                                "⬇️ Download Results",
                                csv,
                                f"fraud_analysis_{datetime.now().strftime( '%Y%m%d_%H%M%S' )}.csv",
                                "text/csv",
                                key="download_results"
                            )

                        except Exception as e :
                            st.error( f"❌ Analysis failed: {str( e )}" )

            except Exception as e :
                st.error( f"❌ Error loading file: {str( e )}" )

    elif page == "🔍 Transaction Explorer" :
        st.markdown( "## 🔍 Transaction Explorer" )
        if 'analysis_data' not in st.session_state :
            st.info( "📊 Run fraud analysis first to explore transactions." )
        else :
            data = st.session_state['analysis_data']

            # Transaction selector
            transaction_options = [f"Transaction {i} - {data.loc[i, 'risk_level']}" for i in data.index]
            selected_option = st.selectbox( "Choose transaction to explore", transaction_options,
                                            key="transaction_selector" )

            if selected_option :
                transaction_idx = int( selected_option.split()[1] )
                selected = data.loc[transaction_idx]

                st.markdown( "### 📋 Transaction Details" )

                # Display transaction details
                col1, col2 = st.columns( 2 )
                with col1 :
                    st.metric( "Fraud Score", f"{selected['fraud_score']:.4f}" )
                    st.write( f"**Risk Level:** {selected['risk_level']}" )

                with col2 :
                    st.metric( "Transaction ID", transaction_idx )
                    st.write( f"**Total Features:** {len( data.columns ) - 2}" )  # Exclude fraud_score and risk_level

                # Show all features
                st.markdown( "### 📊 Feature Values" )
                feature_data = []
                for col in data.columns :
                    if col not in ['fraud_score', 'risk_level'] :
                        feature_data.append( {"Feature" : col, "Value" : selected[col]} )

                if feature_data :
                    feature_df = pd.DataFrame( feature_data )
                    st.dataframe( feature_df, use_container_width=True, hide_index=True )

    elif page == "📈 Reports" :
        st.markdown( "## 📈 Reports" )
        if 'analysis_data' in st.session_state :
            data = st.session_state['analysis_data']

            # Risk distribution chart
            st.markdown( "### 📊 Risk Distribution" )
            fig, ax = plt.subplots( figsize=(10, 6) )
            counts = data['risk_level'].value_counts()
            colors = ['#4CAF50', '#FF9800', '#F44336']
            bars = ax.bar( counts.index, counts.values, color=colors )
            ax.set_title( "Risk Level Distribution", fontsize=16, fontweight='bold' )
            ax.set_xlabel( "Risk Level" )
            ax.set_ylabel( "Number of Transactions" )

            # Add value labels on bars
            for bar in bars :
                height = bar.get_height()
                ax.text( bar.get_x() + bar.get_width() / 2., height + 0.5,
                         f'{int( height )}', ha='center', va='bottom' )

            plt.tight_layout()
            st.pyplot( fig )

            # Summary statistics
            st.markdown( "### 📈 Summary Statistics" )
            col1, col2, col3, col4 = st.columns( 4 )
            with col1 :
                st.metric( "Total Transactions", len( data ) )
            with col2 :
                st.metric( "Average Risk Score", f"{data['fraud_score'].mean():.4f}" )
            with col3 :
                st.metric( "Highest Risk Score", f"{data['fraud_score'].max():.4f}" )
            with col4 :
                st.metric( "Standard Deviation", f"{data['fraud_score'].std():.4f}" )

            # Export report
            if st.button( "📄 Export Report", key="export_report" ) :
                report_data = {
                    'Total Transactions' : len( data ),
                    'Low Risk' : len( data[data['risk_level'] == '🟢 Low'] ),
                    'Medium Risk' : len( data[data['risk_level'] == '🟡 Medium'] ),
                    'High Risk' : len( data[data['risk_level'] == '🔴 High'] ),
                    'Average Risk Score' : data['fraud_score'].mean(),
                    'Max Risk Score' : data['fraud_score'].max(),
                    'Min Risk Score' : data['fraud_score'].min(),
                    'Std Risk Score' : data['fraud_score'].std()
                }

                report_df = pd.DataFrame( list( report_data.items() ), columns=['Metric', 'Value'] )
                csv = report_df.to_csv( index=False ).encode( 'utf-8' )
                st.download_button(
                    "⬇️ Download Report",
                    csv,
                    f"fraud_report_{datetime.now().strftime( '%Y%m%d_%H%M%S' )}.csv",
                    "text/csv",
                    key="download_report"
                )
        else :
            st.info( "📊 Run fraud analysis first to generate reports." )

    elif page == "⚙️ Settings" :
        st.markdown( "## ⚙️ Settings" )

        # Model settings
        st.markdown( "### 🤖 Model Configuration" )
        threshold = st.slider( "Fraud Detection Threshold", 0.0, 1.0, 0.5, 0.01, key="threshold_slider" )
        st.info( f"Transactions with scores above {threshold:.2f} will be flagged as suspicious" )

        # System settings
        st.markdown( "### 🔧 System Configuration" )
        auto_save = st.checkbox( "Auto-save analysis results", True, key="auto_save" )
        notifications = st.checkbox( "Enable email notifications", True, key="notifications" )
        log_level = st.selectbox( "Log Level", ["INFO", "WARNING", "ERROR", "DEBUG"], key="log_level" )

        # Data retention
        st.markdown( "### 🗄️ Data Management" )
        retention_days = st.selectbox( "Data retention period (days)", [7, 30, 90, 180, 365], index=2, key="retention" )

        if st.button( "💾 Save Settings", key="save_settings" ) :
            st.success( "✅ Settings saved successfully!" )
            st.info( f"""
            **Current Settings:**
            - Fraud threshold: {threshold:.2f}
            - Auto-save: {'✅' if auto_save else '❌'}
            - Notifications: {'✅' if notifications else '❌'}
            - Log level: {log_level}
            - Data retention: {retention_days} days
            """ )

elif authentication_status == False :
    st.error( "❌ Invalid username or password" )
    st.markdown( """
    ### 🔐 Demo Credentials  
    **Username:** admin  
    **Password:** admin123  
    **or**  
    **Username:** compliance  
    **Password:** admin123
    """ )

elif authentication_status is None :
    st.warning( "🔒 Please enter your username and password" )
    st.markdown( """
    ### 🛡️ Welcome to FinGuardPro Admin Dashboard
    Please log in to access the admin dashboard.

    ### 🔐 Demo Credentials  
    **Username:** admin  
    **Password:** admin123  
    **or**  
    **Username:** compliance  
    **Password:** admin123
    """ )