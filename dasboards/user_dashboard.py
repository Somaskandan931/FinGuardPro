import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os

# ==== USER CREDENTIALS ====
user_config = {
    'credentials' : {
        'usernames' : {
            'user1' : {
                'name' : 'Ravi Kumar',
                'password' : stauth.Hasher( ['user123'] ).generate()[0]
            },
            'user2' : {
                'name' : 'Anjali Sharma',
                'password' : stauth.Hasher( ['pass456'] ).generate()[0]
            }
        }
    },
    'cookie' : {
        'name' : 'user_cookie',
        'key' : 'user_cookie_key',
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
            model_path = "C:/Users/somas/PycharmProjects/FinGuardPro/models/fraud_detection_model.h5"
            scaler_path = "C:/Users/somas/PycharmProjects/FinGuardPro/models/scaler.pkl"

            if not os.path.exists( model_path ) :
                st.error( f"Model file not found at: {model_path}" )
                return None, None
            if not os.path.exists( scaler_path ) :
                st.error( f"Scaler file not found at: {scaler_path}" )
                return None, None

            model = tf.keras.models.load_model( model_path )
            scaler = joblib.load( scaler_path )
            return model, scaler
        except Exception as e :
            st.error( f"Error loading model and scaler: {e}" )
            return None, None


    model, scaler = load_model_and_scaler()

    if model is None or scaler is None :
        st.error( "Cannot proceed without model and scaler. Please check file paths." )
        st.stop()

    # ==== UPLOAD FILE ====
    uploaded_file = st.file_uploader( "📤 Upload Your Transaction CSV", type="csv" )

    if uploaded_file :
        try :
            df = pd.read_csv( uploaded_file )

            st.subheader( "📋 Uploaded Transactions" )
            st.dataframe( df.head() )

            try :
                # Scale - make sure the dataframe has the right columns
                X_scaled = scaler.transform( df )

                # Predict
                fraud_scores = model.predict( X_scaled )
                df["fraud_score"] = fraud_scores.flatten()
                df["fraud_label"] = df["fraud_score"].apply(
                    lambda x : "🟢 Safe" if x < 0.3 else ("🟡 Suspicious" if x < 0.7 else "🔴 High Risk")
                )

                st.subheader( "🚨 Fraud Risk Results" )
                st.dataframe( df[["fraud_score", "fraud_label"]].head( 10 ) )

                # Summary statistics
                risk_counts = df["fraud_label"].value_counts()
                st.subheader( "📊 Risk Summary" )
                for risk, count in risk_counts.items() :
                    st.write( f"{risk}: {count} transactions" )

                # Optional: download
                csv_download = df.to_csv( index=False ).encode( 'utf-8' )
                st.download_button(
                    label="⬇️ Download Full Results CSV",
                    data=csv_download,
                    file_name="fraud_check_results.csv",
                    mime="text/csv"
                )

            except Exception as e :
                st.error( f"❌ Error processing file: {e}" )
                st.error( "Please ensure your CSV has the correct columns expected by the model." )
                st.info( "The CSV should contain the same features that the model was trained on." )

        except Exception as e :
            st.error( f"❌ Error reading CSV file: {e}" )

elif authentication_status is False :
    st.error( "❌ Invalid username or password." )

elif authentication_status is None :
    st.warning( "Please log in to continue." )