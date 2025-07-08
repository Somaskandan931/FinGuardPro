import optuna
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import joblib

def objective(trial):
    X = pd.read_csv("C:/Users/somas/PycharmProjects/FinGuardPro/data/X_train.csv")
    y = pd.read_csv("C:/Users/somas/PycharmProjects/FinGuardPro/data/y_train.csv").values.ravel()

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'use_label_encoder': False,
        'verbosity': 0,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'lambda': trial.suggest_float('lambda', 0.0, 5.0),
        'alpha': trial.suggest_float('alpha', 0.0, 5.0),
        'scale_pos_weight': (y == 0).sum() / (y == 1).sum(),
        'random_state': 42,
        'n_jobs': -1
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(**params, n_estimators=300)
        model.fit(X_train, y_train)

        preds = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, preds))

    return sum(aucs) / len(aucs)

def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("🎯 Best AUC:", study.best_value)
    print("🏆 Best hyperparameters:", study.best_params)

    # Train final model on full data
    X = pd.read_csv("C:/Users/somas/PycharmProjects/FinGuardPro/data/X_train.csv")
    y = pd.read_csv("C:/Users/somas/PycharmProjects/FinGuardPro/data/y_train.csv").values.ravel()

    best_params = study.best_params
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'use_label_encoder': False,
        'verbosity': 0,
        'scale_pos_weight': (y == 0).sum() / (y == 1).sum(),
        'random_state': 42,
        'n_jobs': -1
    })

    final_model = xgb.XGBClassifier(**best_params, n_estimators=300)
    final_model.fit(X, y)

    joblib.dump(final_model, "C:/Users/somas/PycharmProjects/FinGuardPro/models/xgboost_tuned_model.pkl")
    print("💾 Final tuned XGBoost model saved!")

if __name__ == "__main__":
    main()
