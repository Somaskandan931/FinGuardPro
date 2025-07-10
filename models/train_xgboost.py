import pandas as pd
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve
)
import json
import os
import warnings
import sys

warnings.filterwarnings("ignore")

DATA_DIR = "//data"
MODEL_DIR = "//models"

def check_xgboost_version():
    ver = tuple(map(int, xgb.__version__.split(".")))
    if ver < (1, 6, 0):
        print(f"⚠️ Warning: Your xgboost version is {xgb.__version__}. "
              "Versions before 1.6.0 may not support early_stopping_rounds in XGBClassifier.fit().")
        return False
    return True

def main():
    # Load preprocessed data
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

    # Load feature columns to enforce column order if exists
    feature_columns_path = os.path.join(DATA_DIR, "feature_columns.json")
    if os.path.exists(feature_columns_path):
        with open(feature_columns_path, "r") as f:
            feature_columns = json.load(f)
        X_train = X_train[feature_columns]
        X_test = X_test[feature_columns]

    print(f"📊 Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    print(f"🔎 Train fraud rate: {y_train.mean():.2%} | Test fraud rate: {y_test.mean():.2%}")

    # Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"⚖️ Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # Initialize XGBoost model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    print("🚀 Training XGBoost model...")

    eval_set = [(X_test, y_test)]
    # Check xgboost version for early stopping support
    support_early_stop = check_xgboost_version()
    try:
        if support_early_stop:
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=20,
                verbose=True
            )
        else:
            print("⚠️ early_stopping_rounds not supported, training without early stopping.")
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=True
            )
    except TypeError as e:
        print(f"⚠️ Caught TypeError: {e}")
        print("Retrying training without early stopping...")
        model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    print("\n🧾 Classification Report:\n", classification_report(y_test, y_pred))
    print(f"📈 AUC Score: {auc:.4f}")

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"✔️ TP: {tp} | ❌ FP: {fp} | ✔️ TN: {tn} | ❌ FN: {fn}")
    print(f"🎯 Recall (Fraud Detection Rate): {tp / (tp + fn):.2%}")
    print(f"🚨 False Alarm Rate: {fp / (fp + tn):.2%}")
    print(f"🎯 Precision: {tp / (tp + fp):.2%}")

    # Save the model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "xgboost_best_model.pkl")
    joblib.dump(model, model_path)
    print(f"💾 Model saved to '{model_path}'")

    # Plot evaluation graphs
    plt.figure(figsize=(18, 6))

    # Confusion Matrix
    plt.subplot(1, 3, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
    plt.title('Confusion Matrix')

    # ROC Curve
    plt.subplot(1, 3, 2)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc='lower right')

    # Precision-Recall Curve
    plt.subplot(1, 3, 3)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.plot(recall, precision, color="purple")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "evaluation_plots.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"📊 Evaluation plots saved to '{plot_path}'")


if __name__ == "__main__":
    main()
