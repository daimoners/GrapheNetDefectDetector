import optuna
import logging
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import torch
logging.basicConfig(filename="optuna.log", level=logging.INFO)

X_norm = np.load("./X_norm.npy")
y_norm = np.load("./y_norm.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y_norm, test_size=0.1, random_state=13
)
X_train = torch.tensor(X_train).to(torch.device(device='cuda'))   
y_train = torch.tensor(y_train).to(torch.device(device='cuda'))   
def objective(trial):

    params = {
    "n_estimators": trial.suggest_int("n_estimators", 600, 2000),
    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
    "max_depth": trial.suggest_int("max_depth", 3, 10),
    "gamma" : trial.suggest_float("gamma", 0.000001, 5, log=True), 
    "min_child_weight":trial.suggest_float("min_child_weight", 1, 20, log=True),
    "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1, log=True),
    "subsample":trial.suggest_float("subsample", 0.5, 1, log=True)
}

    #model = GradientBoostingRegressor(**params)
    model = xgb.XGBRegressor(device = 'cuda',**params)

    # Valutazione del modello tramite cross-validation
    score = cross_val_score(
        model, X_train.cpu(), y_train.cpu(), cv=5, scoring="neg_mean_squared_error"
    ).mean()

    logging.info(f"Trial - Score: {score}, Params: {params}")

    return score


study = optuna.create_study(direction="maximize", study_name="test_optuna")
study.optimize(objective, n_trials=150)

print("Best trial:")
trial = study.best_trial
print("Value: ", trial.value)
print("Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
