import optuna
import logging
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import numpy as np
import json
import matplotlib.pyplot as plt

logging.basicConfig(filename="optuna.log", level=logging.INFO)

# Carica i dati
X_norm = np.load("./X_robust.npy")
y_norm = np.load("./Y_current_robust.npy")
print(y_norm.shape)
# Funzione obiettivo
def objective(trial):
    params = {
        #"n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "gamma": trial.suggest_float("gamma", 0.00001, 5, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1, log=True),
        "max_leaves": trial.suggest_int("num_leaves", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.5, 1, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.5, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
        "device":"cuda"
    }

    #model = xgb.XGBRegressor(**params)

    # K-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=13)
    train_rmse_list = []
    val_rmse_list = []
    r2_list = []

    for train_index, val_index in kf.split(X_norm):
        X_train, X_val = X_norm[train_index], X_norm[val_index]
        y_train, y_val = y_norm[train_index], y_norm[val_index]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(params,dtrain, 770, evals = [(dtrain, 'train'), (dval, 'eval')],verbose_eval=False, early_stopping_rounds=50)

        train_rmse = mean_squared_error(y_train, model.predict(dtrain))
        val_rmse = mean_squared_error(y_val, model.predict(dval))
        r2 = r2_score(y_val, model.predict(dval))

        train_rmse_list.append(train_rmse)
        val_rmse_list.append(val_rmse)
        r2_list.append(r2)

    # Media delle metriche
    mean_train_rmse = np.mean(train_rmse_list)
    mean_val_rmse = np.mean(val_rmse_list)
    mean_r2 = np.mean(r2_list)

    val_diff = (mean_val_rmse - mean_train_rmse) / mean_train_rmse
    neg_r2 = -mean_r2
    alpha = 0.5
    # se voglio massimizzare r2 significa anche che voglio andare a minimizzare 1-r2
    composite_score = alpha * mean_val_rmse + (1-alpha)*(1 - mean_r2) + 0.4 * val_diff
    logging.info(f"Trial - Mean Train RMSE: {mean_train_rmse}, Mean Val RMSE: {mean_val_rmse}, Mean R2: {composite_score}, Params: {params}")

    return val_diff, composite_score

study = optuna.create_study(directions=["minimize", "minimize"], study_name="test_optuna")
study.optimize(objective, n_trials=550)

# Estrai i trial dal fronte di Pareto
pareto_trials = study.best_trials
all_trials = study.trials
# Salva tutti i trial del fronte di Pareto in un file JSON
pareto_results = []
for i, trial in enumerate(pareto_trials):
    trial_info = {
        "index": i,
        "values": trial.values,
        "params": trial.params
    }
    pareto_results.append(trial_info)
all_res = []
for i, trial in enumerate(all_trials):
    trial_info = {
        "index": i,
        "values": trial.values,
        "params": trial.params
    }
    all_res.append(trial_info)

with open('composite_pareto_results_current.json', 'w') as f:
    json.dump(pareto_results, f, indent=4)
with open('composite_all_trials_results_current.json', 'w') as f:
    json.dump(all_res, f, indent=4)
# Plotting the Pareto front
val_diffs = [trial.values[0] for trial in pareto_trials]
neg_r2 = [trial.values[1] for trial in pareto_trials]