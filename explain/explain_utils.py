import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ---------------------------------------------------------

# Load model, scaler, and encoders
def load_components(model_path, scaler_path, encoder_path):
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    encoders = joblib.load(encoder_path)
    return model, scaler, encoders

# ---------------------------------------------------------
# Preprocess test data for model input
def preprocess_input(df, scaler, encoders):
    df = df.copy()

    # Encode categorical features
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    # Drop non-feature columns
    transaction_ids = df["transaction_id"]
    feature_df = df.drop(columns=["transaction_id"])

    # Scale features
    scaled = scaler.transform(feature_df)
    return scaled, feature_df.columns.tolist(), transaction_ids

# ---------------------------------------------------------
# Generate SHAP explanation and save as PNG
def explain_transaction(model, X_scaled, feature_names, idx=0, save_path="C:/Users/somas/PycharmProjects/FinGuardPro/explain/shap_explanation.png"):
    # Background = reference data (first 100 samples)
    explainer = shap.DeepExplainer(model, X_scaled[:100])
    shap_values = explainer.shap_values(X_scaled[idx:idx+1])

    # Create and save bar plot
    shap.initjs()
    plt.figure()
    shap.summary_plot(
        shap_values[0],
        features=X_scaled[idx:idx+1],
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"[INFO] SHAP explanation saved to: {save_path}")
    return save_path

# ---------------------------------------------------------
def generate_shap_plot(transaction_id, model_path, scaler_path, encoder_path, transaction_data, output_dir):
    """
    Generate SHAP explanation plot for a specific transaction.
    This is the main function called by the Flask server.

    Parameters:
    - transaction_id: ID of the transaction
    - model_path, scaler_path, encoder_path: paths to load model components
    - transaction_data: pd.DataFrame for the transaction(s)
    - output_dir: directory path as string to save the plot (without trailing slash)

    Returns:
    - path to the saved SHAP plot PNG
    """
    try:
        # Build file path manually (no os.path.join)
        plot_path = output_dir + "/shap_" + str(transaction_id) + ".png"

        # Optionally: skip file existence check to avoid os module usage
        # If you want to skip regeneration if file exists, you could do:
        # try:
        #     with open(plot_path, 'rb'):
        #         print(f"[INFO] Using existing SHAP plot for transaction {transaction_id}")
        #         return plot_path
        # except FileNotFoundError:
        #     pass  # proceed to generate

        # Load model components
        model, scaler, encoders = load_components(model_path, scaler_path, encoder_path)

        # Preprocess the data
        X_scaled, feature_names, _ = preprocess_input(transaction_data, scaler, encoders)

        # Generate SHAP explanation plot
        saved_path = explain_transaction(model, X_scaled, feature_names, idx=0, save_path=plot_path)

        return saved_path

    except Exception as e:
        print(f"[ERROR] Failed to generate SHAP plot for transaction {transaction_id}: {str(e)}")
        raise
