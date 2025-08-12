# train_catboost.py

import pandas as pd
import joblib
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve
)
import warnings
warnings.filterwarnings("ignore")

def main():
    X_train = pd.read_csv( "C:/Users/somas/PycharmProjects/FinGuardPro/data/X_train.csv" )
    X_test = pd.read_csv( "C:/Users/somas/PycharmProjects/FinGuardPro/data/X_test.csv" )
    y_train = pd.read_csv( "C:/Users/somas/PycharmProjects/FinGuardPro/data/y_train.csv" ).values.ravel()
    y_test = pd.read_csv( "C:/Users/somas/PycharmProjects/FinGuardPro/data/y_test.csv" ).values.ravel()

    print(f"📊 Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    print(f"🔎 Train fraud rate: {y_train.mean():.2%} | Test fraud rate: {y_test.mean():.2%}")

    # CatBoost can handle categorical features natively,
    # but since you already encoded, train normally
    model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=5,
        eval_metric='AUC',
        random_seed=42,
        verbose=0,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
    )

    print("🚀 Training CatBoost model...")
    model.fit(X_train, y_train)

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

    joblib.dump( model, "//models/catboost_best_model.pkl" )
    print("💾 Model saved to 'models/catboost_best_model.pkl'")

if __name__ == "__main__":
    main()
