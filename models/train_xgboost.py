# train_xgboost_best_split.py

import pandas as pd
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve
)
import warnings
warnings.filterwarnings("ignore")


def main():
    # -------------------- Load Preprocessed Data -------------------- #
    X_train = pd.read_csv("C:/Users/somas/PycharmProjects/FinGuardPro/data/X_train.csv")
    X_test = pd.read_csv("C:/Users/somas/PycharmProjects/FinGuardPro/data/X_test.csv")
    y_train = pd.read_csv("C:/Users/somas/PycharmProjects/FinGuardPro/data/y_train.csv").values.ravel()
    y_test = pd.read_csv("C:/Users/somas/PycharmProjects/FinGuardPro/data/y_test.csv").values.ravel()

    print(f"📊 Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    print(f"🔎 Train fraud rate: {y_train.mean():.2%} | Test fraud rate: {y_test.mean():.2%}")

    # -------------------- Handle Class Imbalance -------------------- #
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"⚖️ Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # -------------------- Initialize XGBoost Model -------------------- #
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

    # -------------------- Train -------------------- #
    print("🚀 Training XGBoost model...")
    model.fit(X_train, y_train)

    # -------------------- Predict & Evaluate -------------------- #
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

    # -------------------- Save Model -------------------- #
    joblib.dump(model, "C:/Users/somas/PycharmProjects/FinGuardPro/models/xgboost_best_model.pkl")
    print("💾 Model saved to 'models/xgboost_best_model.pkl'")

    # -------------------- Evaluation Plots -------------------- #
    plt.figure(figsize=(16, 5))

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
    plt.legend()

    # Precision-Recall Curve
    plt.subplot(1, 3, 3)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.plot(recall, precision, color="purple")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
