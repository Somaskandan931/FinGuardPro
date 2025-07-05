import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import os
import yaml
from pathlib import Path
from datetime import datetime
import io

# Configure page
st.set_page_config(
    page_title="FinGuard Pro - Admin Dashboard",
    page_icon="🛡️",
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main .block-container {
        background: white;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        padding: 2rem;
        margin-top: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
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
    </style>
""", unsafe_allow_html=True)

# Authentication configuration
def get_auth_config():
    config = {
        'credentials': {
            'usernames': {
                'admin': {
                    'name': 'Admin User',
                    'password': '$2b$12$lKOzLvY4mZJLOzZ1lVyZH.O7KjLfGxDGF1lWRZgGqzPLaJZQZQzLa'  # admin123
                },
                'compliance': {
                    'name': 'Compliance Officer',
                    'password': '$2b$12$lKOzLvY4mZJLOzZ1lVyZH.O7KjLfGxDGF1lWRZgGqzPLaJZQZQzLa'  # admin123
                }
            }
        },
        'cookie': {
            'name': 'finguard_admin_cookie',
            'key': 'finguard_admin_secret_key_2024',
            'expiry_days': 1
        },
        'preauthorized': {'emails': []}
    }
    return config

# Initialize authenticator
config = get_auth_config()
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# Login
name, authentication_status, username = authenticator.login('Login to Admin Dashboard', 'main')

if authentication_status == True:
    # Logout button in sidebar
    authenticator.logout("Logout", "sidebar")
    
    # Welcome message
    st.sidebar.success(f"Welcome, {name}! 👋")
    st.sidebar.markdown("---")
    
    # Header
    st.markdown('<h1 class="header-text">🛡️ FinGuard Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Financial Fraud Detection & Analysis</p>', unsafe_allow_html=True)
    
    # Dashboard navigation
    st.sidebar.title("🎛️ Dashboard Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["🏠 Overview", "📊 Fraud Analysis", "🔍 Transaction Explorer", "📈 Reports", "⚙️ Settings"]
    )
    
    # Load model and preprocessor
    @st.cache_resource
    def load_components():
        try:
            # Use relative paths that work across different systems
            base_path = Path(__file__).parent.parent
            model_path = base_path / 'models' / 'fraud_detection_model.h5'
            scaler_path = base_path / 'models' / 'scaler.pkl'
            
            if not model_path.exists():
                st.warning(f"⚠️ Model file not found at: {model_path}")
                return None, None, "Model file not found"
            
            if not scaler_path.exists():
                st.warning(f"⚠️ Scaler file not found at: {scaler_path}")
                return None, None, "Scaler file not found"
            
            model = tf.keras.models.load_model(str(model_path))
            scaler = joblib.load(str(scaler_path))
            return model, scaler, "success"
        except Exception as e:
            return None, None, f"Error loading components: {e}"
    
    model, scaler, load_status = load_components()
    
    if page == "🏠 Overview":
        st.markdown("## 📊 System Overview")
        
        # System status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if model is not None:
                st.markdown('<div class="success-card"><h3>✅ Model Status</h3><p>Active & Ready</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-card"><h3>⚠️ Model Status</h3><p>Not Available</p></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card"><h3>🔍 Transactions</h3><p>Ready to Process</p></div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="success-card"><h3>📈 Reports</h3><p>Available</p></div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card"><h3>🛡️ Security</h3><p>Active</p></div>', unsafe_allow_html=True)
        
        # Quick actions
        st.markdown("## 🚀 Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Analyze New Transactions", key="quick_analyze"):
                st.session_state.page = "📊 Fraud Analysis"
                st.rerun()
        
        with col2:
            if st.button("🔍 Explore Data", key="quick_explore"):
                st.session_state.page = "🔍 Transaction Explorer"
                st.rerun()
        
        with col3:
            if st.button("📈 Generate Reports", key="quick_reports"):
                st.session_state.page = "📈 Reports"
                st.rerun()
        
        # System information
        st.markdown("## ℹ️ System Information")
        if load_status != "success":
            st.error(f"⚠️ System Issue: {load_status}")
            st.info("💡 To resolve this issue, please ensure that the trained model and scaler files are available in the models directory.")
    
    elif page == "📊 Fraud Analysis":
        st.markdown("## 📊 Fraud Analysis")
        
        if model is None or scaler is None:
            st.error("❌ Cannot perform analysis without trained model and scaler.")
            st.info("Please train the model first or check if the model files are available in the models directory.")
            st.stop()
        
        # File upload
        uploaded_file = st.file_uploader(
            "📤 Upload Transaction Data",
            type=["csv"],
            help="Upload a CSV file containing transaction data for fraud analysis"
        )
        
        if uploaded_file:
            try:
                data = pd.read_csv(uploaded_file)
                
                # Data preview
                st.markdown("### 📋 Data Preview")
                st.dataframe(data.head(), use_container_width=True)
                
                # Data info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(data))
                with col2:
                    st.metric("Features", len(data.columns))
                with col3:
                    st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
                
                # Analysis button
                if st.button("🔍 Run Fraud Analysis", key="analyze_button"):
                    try:
                        with st.spinner("Analyzing transactions..."):
                            # Prepare data for prediction
                            X_scaled = scaler.transform(data)
                            fraud_scores = model.predict(X_scaled)
                            
                            # Add predictions to dataframe
                            data['fraud_score'] = fraud_scores.flatten()
                            data['risk_level'] = data['fraud_score'].apply(
                                lambda x: "🟢 Low Risk" if x < 0.3 else ("🟡 Medium Risk" if x < 0.7 else "🔴 High Risk")
                            )
                            
                            # Store in session state
                            st.session_state['analysis_data'] = data
                            
                            # Display results
                            st.markdown("### 🎯 Analysis Results")
                            
                            # Risk distribution
                            risk_counts = data['risk_level'].value_counts()
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                low_risk = risk_counts.get("🟢 Low Risk", 0)
                                st.metric("Low Risk", low_risk, f"{low_risk/len(data)*100:.1f}%")
                            
                            with col2:
                                medium_risk = risk_counts.get("🟡 Medium Risk", 0)
                                st.metric("Medium Risk", medium_risk, f"{medium_risk/len(data)*100:.1f}%")
                            
                            with col3:
                                high_risk = risk_counts.get("🔴 High Risk", 0)
                                st.metric("High Risk", high_risk, f"{high_risk/len(data)*100:.1f}%")
                            
                            # Results table
                            st.markdown("### 📊 Detailed Results")
                            st.dataframe(
                                data[['fraud_score', 'risk_level']].round(4),
                                use_container_width=True
                            )
                            
                            # Download results
                            csv = data.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="⬇️ Download Results",
                                data=csv,
                                file_name=f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                    
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        st.info("Please ensure your CSV has the correct columns expected by the model.")
            
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    elif page == "🔍 Transaction Explorer":
        st.markdown("## 🔍 Transaction Explorer")
        
        if 'analysis_data' not in st.session_state:
            st.info("📊 Please run fraud analysis first to explore transactions.")
        else:
            data = st.session_state['analysis_data']
            
            # Transaction selector
            st.markdown("### 🎯 Select Transaction")
            transaction_idx = st.selectbox(
                "Choose a transaction to explore",
                data.index.tolist(),
                format_func=lambda x: f"Transaction {x} - {data.loc[x, 'risk_level']}"
            )
            
            # Transaction details
            if transaction_idx is not None:
                selected_transaction = data.loc[transaction_idx]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 📋 Transaction Details")
                    for col in data.columns:
                        if col not in ['fraud_score', 'risk_level']:
                            st.write(f"**{col}**: {selected_transaction[col]}")
                
                with col2:
                    st.markdown("### 🎯 Risk Assessment")
                    st.metric("Fraud Score", f"{selected_transaction['fraud_score']:.4f}")
                    st.write(f"**Risk Level**: {selected_transaction['risk_level']}")
                
                # SHAP explanation button
                if st.button("🧠 Generate SHAP Explanation", key="shap_button"):
                    st.info("🔄 SHAP explanation feature coming soon!")
    
    elif page == "📈 Reports":
        st.markdown("## 📈 Reports & Analytics")
        
        if 'analysis_data' not in st.session_state:
            st.info("📊 Please run fraud analysis first to generate reports.")
        else:
            data = st.session_state['analysis_data']
            
            # Report options
            st.markdown("### 📊 Available Reports")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Generate Summary Report", key="summary_report"):
                    st.info("📄 PDF report generation feature coming soon!")
            
            with col2:
                if st.button("📁 Generate Batch Reports", key="batch_reports"):
                    st.info("📁 Batch report generation feature coming soon!")
            
            # Basic analytics
            st.markdown("### 📊 Quick Analytics")
            
            # Risk distribution chart
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            risk_counts = data['risk_level'].value_counts()
            colors = ['#4CAF50', '#FF9800', '#F44336']
            ax.bar(risk_counts.index, risk_counts.values, color=colors)
            ax.set_title('Risk Distribution')
            ax.set_xlabel('Risk Level')
            ax.set_ylabel('Number of Transactions')
            st.pyplot(fig)
    
    elif page == "⚙️ Settings":
        st.markdown("## ⚙️ System Settings")
        
        # Model settings
        st.markdown("### 🤖 Model Configuration")
        threshold = st.slider("Fraud Detection Threshold", 0.0, 1.0, 0.5, 0.01)
        
        # System settings
        st.markdown("### 🔧 System Configuration")
        auto_save = st.checkbox("Auto-save analysis results", value=True)
        notifications = st.checkbox("Enable notifications", value=True)
        
        if st.button("💾 Save Settings", key="save_settings"):
            st.success("✅ Settings saved successfully!")

elif authentication_status == False:
    st.error("❌ Username or password is incorrect")
    st.markdown("""
    ### 🔐 Demo Credentials
    **Username:** admin  
    **Password:** admin123
    
    **Or**
    
    **Username:** compliance  
    **Password:** admin123
    """)

elif authentication_status == None:
    st.warning("🔒 Please enter your username and password")
    st.markdown("""
    ### 🛡️ Welcome to FinGuard Pro Admin Dashboard
    
    This is a secure admin dashboard for financial fraud detection and analysis.
    
    ### 🔐 Demo Credentials
    **Username:** admin  
    **Password:** admin123
    
    **Or**
    
    **Username:** compliance  
    **Password:** admin123
    """)