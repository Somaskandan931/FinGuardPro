import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os

# Configure Streamlit page
st.set_page_config(
    page_title="FinGuard Pro - User Dashboard",
    page_icon="💳",
    layout="wide"
)

# Hash passwords correctly
hashed_passwords = stauth.Hasher( ['user123', 'pass456'] ).generate()

user_config = {
    'credentials' : {
        'usernames' : {
            'user1' : {
                'name' : 'Ravi Kumar',
                'password' : hashed_passwords[0]
            },
            'user2' : {
                'name' : 'Anjali Sharma',
                'password' : hashed_passwords[1]
            }
        }
    },
    'cookie' : {
        'name' : 'user_cookie',
        'key' : 'user_cookie_key_12345',
        'expiry_days' : 1
    },
    'preauthorized' : {
        'emails' : []
    }
}

# ==== AUTHENTICATOR ====
authenticator = stauth.Authenticate(
    user_config['credentials'],
    user_config['cookie']['name'],
    user_config['cookie']['key'],
    user_config['cookie']['expiry_days'],
    user_config['preauthorized']
)

name, authentication_status, username = authenticator.login( "Login", "main" )

if authentication_status :
    authenticator.logout( "Logout", "sidebar" )
    st.sidebar.success( f"Welcome, {name} 👋" )
    st.title( "💳 FinGuard Pro – User Dashboard" )
    st.markdown( "Here you can check your recent transactions and get fraud alerts." )


    # ==== LOAD MODEL + SCALER ====
    @st.cache_resource
    def load_model_and_scaler () :
        try :
            # Try to load from different possible paths
            model_paths = [
                "fraud_detection_model.h5",
                "models/fraud_detection_model.h5",
                "./models/fraud_detection_model.h5"
            ]

            scaler_paths = [
                "scaler.pkl",
                "models/scaler.pkl",
                "./models/scaler.pkl"
            ]

            model = None
            scaler = None

            # Try to find and load model
            for path in model_paths :
                if os.path.exists( path ) :
                    try :
                        model = tf.keras.models.load_model( path )
                        st.success( f"✅ Model loaded from {path}" )
                        break
                    except Exception as e :
                        st.warning( f"Failed to load model from {path}: {e}" )
                        continue

            # Try to find and load scaler
            for path in scaler_paths :
                if os.path.exists( path ) :
                    try :
                        scaler = joblib.load( path )
                        st.success( f"✅ Scaler loaded from {path}" )
                        break
                    except Exception as e :
                        st.warning( f"Failed to load scaler from {path}: {e}" )
                        continue

            if model is None or scaler is None :
                st.warning( "⚠️ Model or scaler files not found. Using demo mode." )
                return None, None, False

            return model, scaler, True

        except Exception as e :
            st.error( f"Error loading model components: {e}" )
            return None, None, False


    model, scaler, model_loaded = load_model_and_scaler()

    # ==== UPLOAD FILE ====
    uploaded_file = st.file_uploader( "📤 Upload Your Transaction CSV", type="csv" )

    if uploaded_file :
        try :
            df = pd.read_csv( uploaded_file )

            st.subheader( "📋 Uploaded Transactions" )
            st.dataframe( df.head() )

            # Show dataset info
            st.info( f"Dataset shape: {df.shape}" )
            st.info( f"Columns: {', '.join( df.columns.tolist() )}" )

            if model_loaded :
                try :
                    # Check if data has the right structure
                    if len( df.columns ) == 0 :
                        st.error( "❌ Empty dataset uploaded" )
                    else :
                        # Scale the data
                        X_scaled = scaler.transform( df )

                        # Predict fraud scores
                        fraud_scores = model.predict( X_scaled )

                        # Handle different prediction output shapes
                        if len( fraud_scores.shape ) > 1 :
                            fraud_scores = fraud_scores.flatten()

                        df["fraud_score"] = fraud_scores
                        df["fraud_label"] = df["fraud_score"].apply(
                            lambda x : "🟢 Safe" if x < 0.3 else ("🟡 Suspicious" if x < 0.7 else "🔴 High Risk")
                        )

                        st.subheader( "🚨 Fraud Risk Results" )

                        # Show summary statistics
                        col1, col2, col3 = st.columns( 3 )
                        with col1 :
                            safe_count = (df['fraud_score'] < 0.3).sum()
                            st.metric( "🟢 Safe Transactions", safe_count )
                        with col2 :
                            suspicious_count = ((df['fraud_score'] >= 0.3) & (df['fraud_score'] < 0.7)).sum()
                            st.metric( "🟡 Suspicious Transactions", suspicious_count )
                        with col3 :
                            high_risk_count = (df['fraud_score'] >= 0.7).sum()
                            st.metric( "🔴 High Risk Transactions", high_risk_count )

                        # Show detailed results
                        st.dataframe( df[["fraud_score", "fraud_label"]] )

                        # Highlight high-risk transactions
                        if high_risk_count > 0 :
                            st.subheader( "⚠️ High Risk Transactions" )
                            high_risk_df = df[df['fraud_score'] >= 0.7]
                            st.dataframe( high_risk_df )

                        # Download option
                        csv_download = df.to_csv( index=False ).encode( 'utf-8' )
                        st.download_button(
                            label="⬇️ Download Full Results CSV",
                            data=csv_download,
                            file_name="fraud_check_results.csv",
                            mime="text/csv"
                        )

                except ValueError as ve :
                    st.error( f"❌ Data format error: {ve}" )
                    st.info( "Please ensure your CSV has the correct features that match the trained model." )

                except Exception as e :
                    st.error( f"❌ Error processing file: {e}" )
                    st.info( "This might be due to feature mismatch between your data and the trained model." )
            else :
                # Demo mode - generate random predictions
                st.warning( "🎭 Demo Mode: Generating random fraud scores" )

                fraud_scores = np.random.rand( len( df ) )
                df["fraud_score"] = fraud_scores
                df["fraud_label"] = df["fraud_score"].apply(
                    lambda x : "🟢 Safe" if x < 0.3 else ("🟡 Suspicious" if x < 0.7 else "🔴 High Risk")
                )

                st.subheader( "🚨 Fraud Risk Results (Demo)" )

                # Show summary statistics
                col1, col2, col3 = st.columns( 3 )
                with col1 :
                    safe_count = (df['fraud_score'] < 0.3).sum()
                    st.metric( "🟢 Safe Transactions", safe_count )
                with col2 :
                    suspicious_count = ((df['fraud_score'] >= 0.3) & (df['fraud_score'] < 0.7)).sum()
                    st.metric( "🟡 Suspicious Transactions", suspicious_count )
                with col3 :
                    high_risk_count = (df['fraud_score'] >= 0.7).sum()
                    st.metric( "🔴 High Risk Transactions", high_risk_count )

                st.dataframe( df[["fraud_score", "fraud_label"]] )

                # Download option
                csv_download = df.to_csv( index=False ).encode( 'utf-8' )
                st.download_button(
                    label="⬇️ Download Demo Results CSV",
                    data=csv_download,
                    file_name="fraud_check_results_demo.csv",
                    mime="text/csv"
                )

        except pd.errors.EmptyDataError :
            st.error( "❌ The uploaded file is empty or corrupted" )
        except pd.errors.ParserError :
            st.error( "❌ Unable to parse the CSV file. Please check the file format." )
        except Exception as e :
            st.error( f"❌ Unexpected error: {e}" )

    # Add some helpful information
    with st.expander( "ℹ️ How to use this dashboard" ) :
        st.markdown( """
        **Steps to analyze your transactions:**
        1. Upload a CSV file containing your transaction data
        2. The system will automatically analyze each transaction
        3. Review the fraud risk scores and labels
        4. Download the results for your records

        **Risk Levels:**
        - 🟢 **Safe**: Fraud score < 0.3
        - 🟡 **Suspicious**: Fraud score 0.3 - 0.7
        - 🔴 **High Risk**: Fraud score > 0.7

        **Demo Mode:** If model files are not found, the system will generate random scores for demonstration purposes.
        """ )

elif authentication_status is False :
    st.error( "❌ Invalid username or password." )
    st.info( "Demo credentials: user1/user123 or user2/pass456" )

elif authentication_status is None :
    st.warning( "Please log in to continue." )
    st.info( "Demo credentials: user1/user123 or user2/pass456" )