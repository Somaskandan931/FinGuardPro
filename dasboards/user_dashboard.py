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
    page_title="FinGuard Pro - User Dashboard",
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
""", unsafe_allow_html=True)

# Authentication configuration
def get_user_config():
    config = {
        'credentials': {
            'usernames': {
                'user1': {
                    'name': 'John Doe',
                    'password': '$2b$12$lKOzLvY4mZJLOzZ1lVyZH.O7KjLfGxDGF1lWRZgGqzPLaJZQZQzLa'  # user123
                },
                'user2': {
                    'name': 'Jane Smith',
                    'password': '$2b$12$lKOzLvY4mZJLOzZ1lVyZH.O7KjLfGxDGF1lWRZgGqzPLaJZQZQzLa'  # user123
                },
                'demo': {
                    'name': 'Demo User',
                    'password': '$2b$12$lKOzLvY4mZJLOzZ1lVyZH.O7KjLfGxDGF1lWRZgGqzPLaJZQZQzLa'  # user123
                }
            }
        },
        'cookie': {
            'name': 'finguard_user_cookie',
            'key': 'finguard_user_secret_key_2024',
            'expiry_days': 1
        },
        'preauthorized': {'emails': []}
    }
    return config

# Initialize authenticator
config = get_user_config()
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# Login
name, authentication_status, username = authenticator.login('Login to User Dashboard', 'main')

if authentication_status == True:
    # Logout button in sidebar
    authenticator.logout("Logout", "sidebar")
    
    # Welcome message
    st.sidebar.success(f"Welcome back, {name}! 👋")
    st.sidebar.markdown("---")
    
    # Header
    st.markdown('<h1 class="header-text">💳 FinGuard Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Personal Financial Security Dashboard</p>', unsafe_allow_html=True)
    
    # Dashboard navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["🏠 Dashboard", "📊 Transaction Analysis", "📈 My Reports", "⚙️ Settings"]
    )
    
    # Load model and preprocessor
    @st.cache_resource
    def load_model_and_scaler():
        try:
            # Use relative paths that work across different systems
            base_path = Path(__file__).parent.parent
            model_path = base_path / 'models' / 'fraud_detection_model.h5'
            scaler_path = base_path / 'models' / 'scaler.pkl'
            
            if not model_path.exists():
                return None, None, "Model file not found"
            
            if not scaler_path.exists():
                return None, None, "Scaler file not found"
            
            model = tf.keras.models.load_model(str(model_path))
            scaler = joblib.load(str(scaler_path))
            return model, scaler, "success"
        except Exception as e:
            return None, None, f"Error loading model: {str(e)}"
    
    model, scaler, load_status = load_model_and_scaler()
    
    if page == "🏠 Dashboard":
        st.markdown("## 🏠 Dashboard Overview")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if model is not None:
                st.markdown('<div class="success-card"><h3>✅ Security Status</h3><p>Protected</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-card"><h3>⚠️ Security Status</h3><p>Limited</p></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card"><h3>📊 Analysis</h3><p>Ready</p></div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="success-card"><h3>📱 Mobile</h3><p>Optimized</p></div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card"><h3>🔐 Privacy</h3><p>Secure</p></div>', unsafe_allow_html=True)
        
        # Quick actions
        st.markdown("## 🚀 Quick Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Upload Transactions", key="quick_upload"):
                st.session_state.page = "📊 Transaction Analysis"
                st.rerun()
        
        with col2:
            if st.button("📊 View Analytics", key="quick_analytics"):
                st.session_state.page = "📈 My Reports"
                st.rerun()
        
        # System status
        st.markdown("## ℹ️ System Status")
        if load_status != "success":
            st.warning(f"⚠️ {load_status}")
            st.info("Some features may be limited without the fraud detection model.")
        else:
            st.success("✅ All systems operational!")
        
        # Security tips
        st.markdown("## 🛡️ Security Tips")
        st.info("""
        **Keep your transactions safe:**
        - Regularly monitor your account activity
        - Report suspicious transactions immediately
        - Use strong, unique passwords
        - Enable two-factor authentication when available
        """)
    
    elif page == "📊 Transaction Analysis":
        st.markdown("## 📊 Transaction Analysis")
        
        if model is None or scaler is None:
            st.error("❌ Transaction analysis is not available without the fraud detection model.")
            st.info("Please contact your system administrator to enable this feature.")
        else:
            # File upload
            uploaded_file = st.file_uploader(
                "📤 Upload Your Transaction Data",
                type=["csv"],
                help="Upload a CSV file containing your transaction data for fraud analysis"
            )
            
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    # Data preview
                    st.markdown("### 📋 Transaction Preview")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # File info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Transactions", len(df))
                    with col2:
                        st.metric("Data Fields", len(df.columns))
                    with col3:
                        st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
                    
                    if st.button("🔍 Analyze Transactions", key="analyze_transactions"):
                        try:
                            with st.spinner("Analyzing your transactions..."):
                                # Scale data
                                X_scaled = scaler.transform(df)
                                
                                # Predict fraud scores
                                fraud_scores = model.predict(X_scaled)
                                df["fraud_score"] = fraud_scores.flatten()
                                df["risk_level"] = df["fraud_score"].apply(
                                    lambda x: "🟢 Low Risk" if x < 0.3 else ("🟡 Medium Risk" if x < 0.7 else "🔴 High Risk")
                                )
                                
                                # Store results
                                st.session_state['user_analysis'] = df
                                
                                # Display results
                                st.markdown("### 🎯 Analysis Results")
                                
                                # Risk summary
                                risk_counts = df["risk_level"].value_counts()
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    low_risk = risk_counts.get("🟢 Low Risk", 0)
                                    st.markdown(f'<div class="success-card"><h3>🟢 Low Risk</h3><h2>{low_risk}</h2><p>{low_risk/len(df)*100:.1f}% of transactions</p></div>', unsafe_allow_html=True)
                                
                                with col2:
                                    medium_risk = risk_counts.get("🟡 Medium Risk", 0)
                                    st.markdown(f'<div class="warning-card"><h3>🟡 Medium Risk</h3><h2>{medium_risk}</h2><p>{medium_risk/len(df)*100:.1f}% of transactions</p></div>', unsafe_allow_html=True)
                                
                                with col3:
                                    high_risk = risk_counts.get("🔴 High Risk", 0)
                                    st.markdown(f'<div class="danger-card"><h3>🔴 High Risk</h3><h2>{high_risk}</h2><p>{high_risk/len(df)*100:.1f}% of transactions</p></div>', unsafe_allow_html=True)
                                
                                # Transaction details
                                st.markdown("### 📊 Transaction Details")
                                
                                # Show flagged transactions first
                                flagged_df = df[df['fraud_score'] >= 0.3].sort_values('fraud_score', ascending=False)
                                
                                if len(flagged_df) > 0:
                                    st.markdown("#### ⚠️ Flagged Transactions")
                                    for idx, row in flagged_df.head(10).iterrows():
                                        risk_class = "transaction-warning" if row['fraud_score'] < 0.7 else "transaction-danger"
                                        st.markdown(f'''
                                        <div class="{risk_class}">
                                            <strong>Transaction {idx}</strong> - Risk Score: {row['fraud_score']:.4f}<br>
                                            <small>Risk Level: {row['risk_level']}</small>
                                        </div>
                                        ''', unsafe_allow_html=True)
                                
                                # Show all results in table
                                st.markdown("#### 📋 All Results")
                                st.dataframe(
                                    df[['fraud_score', 'risk_level']].round(4),
                                    use_container_width=True
                                )
                                
                                # Download results
                                csv = df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="⬇️ Download Analysis Results",
                                    data=csv,
                                    file_name=f"transaction_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                        
                        except Exception as e:
                            st.error(f"❌ Analysis failed: {str(e)}")
                            st.info("Please ensure your CSV has the correct format and columns.")
                
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
    
    elif page == "📈 My Reports":
        st.markdown("## 📈 My Reports")
        
        if 'user_analysis' not in st.session_state:
            st.info("📊 Please analyze some transactions first to generate reports.")
            
            # Show sample visualization
            st.markdown("### 📊 Sample Analytics")
            
            # Create sample data for demonstration
            sample_data = {
                'Risk Level': ['🟢 Low Risk', '🟡 Medium Risk', '🔴 High Risk'],
                'Count': [150, 25, 5]
            }
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#4CAF50', '#FF9800', '#F44336']
            bars = ax.bar(sample_data['Risk Level'], sample_data['Count'], color=colors)
            ax.set_title('Sample Risk Distribution', fontsize=16, fontweight='bold')
            ax.set_xlabel('Risk Level')
            ax.set_ylabel('Number of Transactions')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        else:
            data = st.session_state['user_analysis']
            
            # Analytics overview
            st.markdown("### 📊 Transaction Analytics")
            
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk distribution pie chart
                risk_counts = data['risk_level'].value_counts()
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['#4CAF50', '#FF9800', '#F44336']
                wedges, texts, autotexts = ax.pie(risk_counts.values, labels=risk_counts.index, 
                                                 autopct='%1.1f%%', colors=colors, startangle=90)
                ax.set_title('Risk Distribution', fontsize=14, fontweight='bold')
                st.pyplot(fig)
            
            with col2:
                # Fraud score distribution
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(data['fraud_score'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
                ax.set_title('Fraud Score Distribution', fontsize=14, fontweight='bold')
                ax.set_xlabel('Fraud Score')
                ax.set_ylabel('Frequency')
                ax.axvline(x=0.3, color='orange', linestyle='--', label='Medium Risk Threshold')
                ax.axvline(x=0.7, color='red', linestyle='--', label='High Risk Threshold')
                ax.legend()
                st.pyplot(fig)
            
            # Summary statistics
            st.markdown("### 📈 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Average Risk Score", f"{data['fraud_score'].mean():.4f}")
            
            with col2:
                st.metric("Highest Risk Score", f"{data['fraud_score'].max():.4f}")
            
            with col3:
                st.metric("Lowest Risk Score", f"{data['fraud_score'].min():.4f}")
            
            with col4:
                st.metric("Risk Standard Deviation", f"{data['fraud_score'].std():.4f}")
            
            # Export options
            st.markdown("### 💾 Export Options")
            col1, col2 = st.columns(2)
            
            with col1:
                # Download detailed results
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 Download Detailed Report",
                    data=csv,
                    file_name=f"detailed_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Download summary
                summary_data = {
                    'Metric': ['Total Transactions', 'Low Risk', 'Medium Risk', 'High Risk', 'Average Score'],
                    'Value': [
                        len(data),
                        len(data[data['risk_level'] == '🟢 Low Risk']),
                        len(data[data['risk_level'] == '🟡 Medium Risk']),
                        len(data[data['risk_level'] == '🔴 High Risk']),
                        f"{data['fraud_score'].mean():.4f}"
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                csv_summary = summary_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 Download Summary",
                    data=csv_summary,
                    file_name=f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    elif page == "⚙️ Settings":
        st.markdown("## ⚙️ Settings")
        
        # User preferences
        st.markdown("### 👤 User Preferences")
        
        # Notification settings
        email_notifications = st.checkbox("Enable email notifications", value=True)
        sms_notifications = st.checkbox("Enable SMS alerts for high-risk transactions", value=False)
        
        # Risk threshold settings
        st.markdown("### 🎯 Risk Thresholds")
        user_threshold = st.slider("Personal Risk Alert Threshold", 0.0, 1.0, 0.5, 0.01)
        st.info(f"You will be alerted for transactions with risk scores above {user_threshold:.2f}")
        
        # Data retention
        st.markdown("### 🗂️ Data Management")
        retention_days = st.selectbox("Data retention period", [7, 30, 90, 180, 365], index=2)
        
        # Export preferences
        st.markdown("### 📤 Export Preferences")
        export_format = st.selectbox("Preferred export format", ["CSV", "Excel", "JSON"], index=0)
        
        # Save settings
        if st.button("💾 Save Settings", key="save_user_settings"):
            st.success("✅ Settings saved successfully!")
            
            # Show current settings
            st.markdown("#### Current Settings:")
            st.write(f"- Email notifications: {'✅' if email_notifications else '❌'}")
            st.write(f"- SMS alerts: {'✅' if sms_notifications else '❌'}")
            st.write(f"- Risk threshold: {user_threshold:.2f}")
            st.write(f"- Data retention: {retention_days} days")
            st.write(f"- Export format: {export_format}")

elif authentication_status == False:
    st.error("❌ Username or password is incorrect")
    st.markdown("""
    ### 🔐 Demo Credentials
    
    **Username:** user1  
    **Password:** user123
    
    **Or**
    
    **Username:** user2  
    **Password:** user123
    
    **Or**
    
    **Username:** demo  
    **Password:** user123
    """)

elif authentication_status == None:
    st.warning("🔒 Please enter your username and password")
    st.markdown("""
    ### 💳 Welcome to FinGuard Pro
    
    Your personal financial security dashboard. Monitor your transactions and get instant fraud alerts.
    
    ### 🔐 Demo Credentials
    
    **Username:** user1  
    **Password:** user123
    
    **Or**
    
    **Username:** user2  
    **Password:** user123
    
    **Or**
    
    **Username:** demo  
    **Password:** user123
    """)