import optuna
import logging
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import numpy as np
import json
import matplotlib.pyplot as plt

logging.basicConfig(filename="optuna.log", level=logging.INFO)

X_norm = np.load("./X_robust.npy")
y_norm = np.load("./Y_current_robust.npy")
print(y_norm.shape)


def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "gamma": trial.suggest_float("gamma", 0.00001, 5, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1, log=True),
        "max_leaves": trial.suggest_int("num_leaves", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.5, 1, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.5, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
        "device": "cuda",
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=13)
    train_rmse_list = []
    val_rmse_list = []
    r2_list = []

    for train_index, val_index in kf.split(X_norm):
        X_train, X_val = X_norm[train_index], X_norm[val_index]
        y_train, y_val = y_norm[train_index], y_norm[val_index]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params,
            dtrain,
            770,
            evals=[(dtrain, "train"), (dval, "eval")],
            verbose_eval=False,
            early_stopping_rounds=50,
        )

        train_rmse = mean_squared_error(y_train, model.predict(dtrain))
        val_rmse = mean_squared_error(y_val, model.predict(dval))
        r2 = r2_score(y_val, model.predict(dval))

        train_rmse_list.append(train_rmse)
        val_rmse_list.append(val_rmse)
        r2_list.append(r2)

    mean_train_rmse = np.mean(train_rmse_list)
    mean_val_rmse = np.mean(val_rmse_list)
    mean_r2 = np.mean(r2_list)

    overfit_quality = (mean_val_rmse - mean_train_rmse) / mean_train_rmse

    alpha = 0.5
    composite_score = (
        alpha * mean_val_rmse + (1 - alpha) * (1 - mean_r2) + 0.4 * overfit_quality
    )
    logging.info(
        f"Trial - Mean Train RMSE: {mean_train_rmse}, Mean Val RMSE: {mean_val_rmse}, Composite Score: {composite_score}, Params: {params}"
    )

    return composite_score


study = optuna.create_study(direction="minimize", study_name="test_optuna")
study.optimize(objective, n_trials=550)

best_trial = study.best_trial

results = []
for i, trial in enumerate(study.trials):
    trial_info = {"index": i, "values": trial.value, "params": trial.params}
    results.append(trial_info)

with open("single_objective_results.json", "w") as f:
    json.dump(results, f, indent=4)

# Plotting the best result
plt.figure(figsize=(10, 5))
plt.title("Optimization History")
plt.plot([t.value for t in study.trials], marker="o")
plt.xlabel("Trial")
plt.ylabel("Composite Score")
plt.show()

print(f"Best Trial: {best_trial.params}, Composite Score: {best_trial.value}")
