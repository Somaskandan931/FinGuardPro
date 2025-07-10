import os
import joblib
import optuna
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DATA_DIR = "//data"
MODEL_DIR = "//models"

def objective(trial):
    X = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'scale_pos_weight': (y == 0).sum() / (y == 1).sum(),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'verbosity': 0,
        'n_estimators': trial.suggest_int('n_estimators', 50, 300)
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    aucs = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=params['n_estimators'],
            evals=[(dval, "eval")],
            early_stopping_rounds=30,
            verbose_eval=False
        )

        preds = bst.predict(dval)
        aucs.append(roc_auc_score(y_val, preds))

    return sum(aucs) / len(aucs)


if __name__ == "__main__":
    # Load existing model (optional)
    base_model_path = os.path.join(MODEL_DIR, "xgboost_best_model.pkl")
    if os.path.exists(base_model_path):
        base_model = joblib.load(base_model_path)
        print(f"✅ Loaded existing model from {base_model_path}")
    else:
        print("⚠️ Existing model not found, tuning from scratch.")

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print("🎯 Best AUC:", study.best_value)
    print("🏆 Best hyperparameters:", study.best_params)

    # Train best model on full train data
    X_full = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y_full = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()

    best_params = study.best_params
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'scale_pos_weight': (y_full == 0).sum() / (y_full == 1).sum(),
        'verbosity': 0
    })

    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(X_full, y_full)

    tuned_model_path = os.path.join(MODEL_DIR, "xgboost_tuned_model.pkl")
    joblib.dump(best_model, tuned_model_path)
    print(f"💾 Tuned model saved to '{tuned_model_path}'")
