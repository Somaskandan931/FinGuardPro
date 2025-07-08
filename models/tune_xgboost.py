import optuna
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def objective(trial):
    X = pd.read_csv("C:/Users/somas/PycharmProjects/FinGuardPro/data/X_train.csv")
    y = pd.read_csv("C:/Users/somas/PycharmProjects/FinGuardPro/data/y_train.csv").values.ravel()

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'scale_pos_weight': (y == 0).sum() / (y == 1).sum(),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'verbosity': 0,
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
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    print("🎯 Best AUC:", study.best_value)
    print("🏆 Best hyperparameters:", study.best_params)
