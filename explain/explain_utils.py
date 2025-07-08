# explain_utils.py

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
    df = input_df.copy()

    # Label encode and create _encoded columns
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col + "_encoded"] = df[col].map(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
        else:
            df[col + "_encoded"] = -1

    # Drop original categorical columns
    df.drop(columns=encoders.keys(), inplace=True, errors='ignore')

    # Extract and scale features
    X = df[feature_columns].copy()
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_columns)
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
    os.makedirs(output_dir, exist_ok=True)

    X = preprocess_input_for_shap(input_df, scaler, encoders, feature_columns)

    try:
        explainer = shap.TreeExplainer(model, data=background_data) if background_data is not None else shap.TreeExplainer(model)
    except Exception as e:
        print("[WARN] TreeExplainer failed, falling back to KernelExplainer")
        explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))

    shap_values = explainer.shap_values(X)
    saved_files = []

    for i in range(min(len(X), top_n)):
        tid = transaction_ids[i] if transaction_ids and i < len(transaction_ids) else f"tx_{i}"
        plt.figure()

        if isinstance(shap_values, list):
            vals = shap_values[1][i]
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
    X = preprocess_input_for_shap(input_df, scaler, encoders, feature_columns)

    explainer = shap.TreeExplainer(model, data=background_data) if background_data is not None else shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"[INFO] SHAP summary plot saved: {output_path}")
    return output_path
