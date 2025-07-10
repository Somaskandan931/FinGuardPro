import shap
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend suitable for servers
import matplotlib.pyplot as plt
import os
import pandas as pd
from typing import Optional, Dict, List, Union, Any

from utils.preprocessing import preprocess_features  # Centralized preprocessing logic


def explain_transaction(
    model: Any,
    input_df: pd.DataFrame,
    scaler: Any,
    encoders: Dict[str, object],
    feature_columns: List[str],
    output_dir: str = "output/shap_outputs",
    top_n: int = 10,
    transaction_ids: Optional[List[Union[str, int]]] = None,
    background_data: Optional[pd.DataFrame] = None
) -> List[str]:
    """
    Generate SHAP waterfall plots for individual transactions.
    Returns list of filepaths to saved images.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Preprocess input dataframe using centralized logic
    X = preprocess_features(input_df, label_encoders=encoders, scaler=scaler, categorical_columns=list(encoders.keys()))

    # Create SHAP explainer, fallback gracefully if TreeExplainer fails
    try:
        explainer = shap.TreeExplainer(model, data=background_data) if background_data is not None else shap.TreeExplainer(model)
    except Exception as e:
        print(f"[WARN] TreeExplainer failed: {e}. Using KernelExplainer fallback.")
        background_sample = shap.sample(X, 100) if len(X) > 100 else X
        explainer = shap.KernelExplainer(model.predict, background_sample)

    shap_values = explainer.shap_values(X)

    saved_files = []
    max_plots = min(len(X), top_n)
    for i in range(max_plots):
        tid = transaction_ids[i] if transaction_ids and i < len(transaction_ids) else f"tx_{i}"
        plt.figure(figsize=(8, 5))

        # Patch plt.show to no-op to prevent GUI display error
        original_show = plt.show
        plt.show = lambda *args, **kwargs: None

        # Handle SHAP values for binary classification and multi-class
        if isinstance(shap_values, list):
            # Assuming binary classification, class 1 shap values
            vals = shap_values[1][i]
            base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, tuple)) else explainer.expected_value
        else:
            vals = shap_values[i]
            base_value = explainer.expected_value

        # Use the legacy waterfall plot for stability
        shap.plots._waterfall.waterfall_legacy(base_value, vals, X.iloc[i])

        # Restore plt.show
        plt.show = original_show

        filepath = os.path.join(output_dir, f"shap_{tid}.png")
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()

        saved_files.append(filepath)
        print(f"[INFO] SHAP explanation saved: {filepath}")

    return saved_files


def explain_summary_plot(
    model: Any,
    input_df: pd.DataFrame,
    scaler: Any,
    encoders: Dict[str, object],
    feature_columns: List[str],
    output_path: str = "output/shap_summary.png",
    background_data: Optional[pd.DataFrame] = None
) -> str:
    """
    Generate and save SHAP summary plot for batch input.
    Returns the filepath to saved plot image.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    X = preprocess_features(input_df, label_encoders=encoders, scaler=scaler, categorical_columns=list(encoders.keys()))

    try:
        explainer = shap.TreeExplainer(model, data=background_data) if background_data is not None else shap.TreeExplainer(model)
    except Exception as e:
        print(f"[WARN] TreeExplainer failed: {e}. Using KernelExplainer fallback.")
        background_sample = shap.sample(X, 100) if len(X) > 100 else X
        explainer = shap.KernelExplainer(model.predict, background_sample)

    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"[INFO] SHAP summary plot saved: {output_path}")
    return output_path
