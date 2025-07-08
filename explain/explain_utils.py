import shap
import matplotlib.pyplot as plt
import os
import pandas as pd
from typing import Optional, Dict, List, Union


def preprocess_input_for_shap(
    input_df: pd.DataFrame,
    scaler,
    encoders: Dict[str, object],
    feature_columns: List[str]
) -> pd.DataFrame:
    """
    Preprocess input data by encoding categorical columns and scaling features,
    preparing it for SHAP explanation.
    """
    df = input_df.copy()

    # Encode categorical columns, mapping unseen categories to -1
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col + "_encoded"] = df[col].map(
                lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
            )
        else:
            # If column not present, fill encoded column with -1
            df[col + "_encoded"] = -1

    # Drop original categorical columns after encoding
    df.drop(columns=list(encoders.keys()), inplace=True, errors='ignore')

    # Select feature columns and apply scaler
    X = df[feature_columns].copy()
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_columns, index=X.index)
    return X_scaled


def explain_transaction(
    model,
    input_df: pd.DataFrame,
    scaler,
    encoders,
    feature_columns,
    output_dir: str = "explain/shap_outputs",
    top_n: int = 10,
    transaction_ids: Optional[List[Union[str, int]]] = None,
    background_data: Optional[pd.DataFrame] = None
) -> List[str]:
    """
    Generate SHAP waterfall plots for individual transactions.

    Returns list of filepaths to saved images.
    """
    os.makedirs(output_dir, exist_ok=True)

    X = preprocess_input_for_shap(input_df, scaler, encoders, feature_columns)

    try:
        explainer = (
            shap.TreeExplainer(model, data=background_data)
            if background_data is not None else shap.TreeExplainer(model)
        )
    except Exception as e:
        print(f"[WARN] TreeExplainer failed with error: {e}. Falling back to KernelExplainer.")
        explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))

    shap_values = explainer.shap_values(X)
    saved_files = []

    max_plots = min(len(X), top_n)
    for i in range(max_plots):
        tid = transaction_ids[i] if transaction_ids and i < len(transaction_ids) else f"tx_{i}"
        plt.figure(figsize=(8, 5))

        # Handle multiclass shap_values (list) vs single-class
        if isinstance(shap_values, list):
            vals = shap_values[1][i]  # Typically class 1 for binary classifiers
            base_value = explainer.expected_value[1]
        else:
            vals = shap_values[i]
            base_value = explainer.expected_value

        shap.plots._waterfall.waterfall_legacy(base_value, vals, X.iloc[i])

        filepath = os.path.join(output_dir, f"shap_{tid}.png")
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()

        saved_files.append(filepath)
        print(f"[INFO] SHAP explanation saved: {filepath}")

    return saved_files


def explain_summary_plot(
    model,
    input_df: pd.DataFrame,
    scaler,
    encoders,
    feature_columns,
    output_path: str = "explain/shap_summary.png",
    background_data: Optional[pd.DataFrame] = None
) -> str:
    """
    Generate and save SHAP summary plot for a batch of transactions.

    Returns filepath to saved image.
    """
    X = preprocess_input_for_shap(input_df, scaler, encoders, feature_columns)

    try:
        explainer = (
            shap.TreeExplainer(model, data=background_data)
            if background_data is not None else shap.TreeExplainer(model)
        )
    except Exception as e:
        print(f"[WARN] TreeExplainer failed with error: {e}. Falling back to KernelExplainer.")
        explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))

    shap_values = explainer.shap_values(X)

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"[INFO] SHAP summary plot saved: {output_path}")
    return output_path
