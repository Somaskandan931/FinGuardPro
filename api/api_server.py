from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import shap
import os
import sys
import os

sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )
from explain.explain_utils import explain_model_prediction  # Reusable SHAP function

app = Flask( __name__ )

# ✅ Hardcoded paths (change these if your directory moves)
MODEL_PATH = "C:/Users/somas/PycharmProjects/FinGuardPro/models/fraud_detection_model.h5"
SCALER_PATH = "C:/Users/somas/PycharmProjects/FinGuardPro/models/scaler.pkl"
ENCODER_PATH = "C:/Users/somas/PycharmProjects/FinGuardPro/models/label_encoders.pkl"
SHAP_IMAGE_PATH = "C:/Users/somas/PycharmProjects/FinGuardPro/explain/shap_explanation.png"

# ✅ Load model and tools
try :
    model = tf.keras.models.load_model( MODEL_PATH )
    scaler = joblib.load( SCALER_PATH )
    # Check if encoders file exists
    if os.path.exists( ENCODER_PATH ) :
        encoders = joblib.load( ENCODER_PATH )
    else :
        encoders = {}
    print( "✅ Models loaded successfully" )
except Exception as e :
    print( f"❌ Error loading models: {e}" )
    model = None
    scaler = None
    encoders = {}


@app.route( '/predict', methods=['POST'] )
def predict_fraud () :
    try :
        if model is None or scaler is None :
            return jsonify( {"error" : "Models not loaded properly"} ), 500

        input_json = request.json
        if not input_json :
            return jsonify( {"error" : "No JSON data provided"} ), 400

        df = pd.DataFrame( [input_json] )

        # ✅ Apply label encoders (if needed)
        for col, le in encoders.items() :
            if col in df.columns :
                try :
                    df[col] = le.transform( df[col] )
                except Exception as e :
                    print( f"Warning: Could not encode column {col}: {e}" )

        # ✅ Scale features
        X_scaled = scaler.transform( df )

        # ✅ Predict fraud probability
        fraud_score = model.predict( X_scaled )[0][0]

        # ✅ Generate SHAP explanation
        background = df.copy()
        try :
            shap_img_path = explain_model_prediction(
                model_path=MODEL_PATH,
                scaler_path=SCALER_PATH,
                sample_df=df,
                background_df=background,
                feature_names=df.columns.tolist(),
                save_path=SHAP_IMAGE_PATH
            )
            shap_available = True
        except Exception as e :
            print( f"Warning: SHAP explanation failed: {e}" )
            shap_available = False

        result = {
            "fraud_score" : round( float( fraud_score ), 4 ),
            "risk_level" : "High Risk" if fraud_score > 0.7 else ("Suspicious" if fraud_score > 0.3 else "Safe"),
            "shap_plot_url" : "/shap-image" if shap_available else None
        }
        return jsonify( result )

    except Exception as e :
        return jsonify( {"error" : str( e )} ), 500


@app.route( '/shap-image', methods=['GET'] )
def get_shap_image () :
    try :
        if os.path.exists( SHAP_IMAGE_PATH ) :
            return send_file( SHAP_IMAGE_PATH, mimetype='image/png' )
        else :
            return jsonify( {"error" : "SHAP image not found"} ), 404
    except Exception as e :
        return jsonify( {"error" : str( e )} ), 500


@app.route( '/ping', methods=['GET'] )
def health_check () :
    return jsonify( {"status" : "FinGuard API running"} )


if __name__ == '__main__' :
    app.run( debug=True, port=5000 )