import shap
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import numpy as np
import os


def explain_model_prediction (
        model_path: str,
        scaler_path: str,
        sample_df,  # DataFrame: 1 row to explain
        background_df,  # DataFrame: few rows for SHAP background
        feature_names: list = None,
        save_path: str = "C:/Users/somas/PycharmProjects/FinGuardPro/explain/shap_explanation.png"
) :
    """
    Generate SHAP explanation for a model prediction

    Args:
        model_path: Path to the trained model
        scaler_path: Path to the fitted scaler
        sample_df: DataFrame with the sample to explain
        background_df: DataFrame with background data for SHAP
        feature_names: List of feature names
        save_path: Path to save the SHAP plot

    Returns:
        Path to the saved SHAP plot
    """
    try :
        # Load model and scaler
        model = tf.keras.models.load_model( model_path )
        scaler = joblib.load( scaler_path )

        # Scale the data
        X_sample_scaled = scaler.transform( sample_df )
        X_background_scaled = scaler.transform( background_df )

        # Use SHAP's default feature names if not provided
        if feature_names is None :
            feature_names = sample_df.columns.tolist()

        # Create SHAP explainer
        explainer = shap.DeepExplainer( model, X_background_scaled )

        # Compute SHAP values for sample
        shap_values = explainer.shap_values( X_sample_scaled )

        # Handle different SHAP value formats
        if isinstance( shap_values, list ) :
            # For multi-output models, take the first output
            shap_values = shap_values[0]

        # Ensure save directory exists
        save_dir = os.path.dirname( save_path )
        if save_dir and not os.path.exists( save_dir ) :
            os.makedirs( save_dir )

        # Create waterfall plot for single prediction
        plt.figure( figsize=(10, 6) )

        # Use waterfall plot for single prediction explanation
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_sample_scaled[0],
                feature_names=feature_names
            ),
            show=False
        )

        plt.tight_layout()
        plt.savefig( save_path, dpi=300, bbox_inches='tight' )
        plt.close()

        print( f"✅ SHAP explanation saved to: {save_path}" )
        return save_path

    except Exception as e :
        print( f"❌ Error generating SHAP explanation: {e}" )

        # Fallback: create a simple bar plot if waterfall fails
        try :
            plt.figure( figsize=(10, 6) )
            if isinstance( shap_values, list ) :
                shap_values = shap_values[0]

            # Create simple bar plot
            importance_values = shap_values[0] if len( shap_values.shape ) > 1 else shap_values

            plt.barh( range( len( importance_values ) ), importance_values )
            plt.yticks( range( len( importance_values ) ), feature_names )
            plt.xlabel( 'SHAP Value' )
            plt.title( 'Feature Importance for Fraud Prediction' )
            plt.tight_layout()
            plt.savefig( save_path, dpi=300, bbox_inches='tight' )
            plt.close()

            print( f"✅ Fallback SHAP explanation saved to: {save_path}" )
            return save_path

        except Exception as e2 :
            print( f"❌ Even fallback SHAP explanation failed: {e2}" )
            return None