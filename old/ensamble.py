try:
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    from lib.lib_utils import Utils
    import seaborn as sns
    from lib.lib_defect_analysis import Features
    from tqdm import tqdm
    import torch
    import torch.nn as nn
    import xgboost as xgb
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.preprocessing import MinMaxScaler
except Exception as e:
    print(f"Some module are missing: {e}\n")

data_path = Path().resolve().joinpath("data")
xyz_files_path = data_path.joinpath("xyz_files")
yolo_model_path = data_path.joinpath("models", "best_100_campioni_new.pt")
images_path = data_path.joinpath("images")
crops_path = data_path.joinpath("crops")

plt.style.use("seaborn-v0_8-paper")


class NeuralNet(nn.Module):
    def __init__(
        self, input_dim, num_hidden_layers, hidden_dim, X_train, y_train, params
    ):
        super(NeuralNet, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.X_train = X_train.cpu().numpy()
        self.y_train = y_train.cpu().numpy()
        self.params = params
        xgb_feature_importance = self.train_xgboost()

        self.first_layer = nn.Linear(input_dim, hidden_dim)
        with torch.no_grad():
            self.first_layer.weight.copy_(
                torch.tensor(xgb_feature_importance, dtype=torch.float32).unsqueeze(0)
            )
            self.first_layer.bias.fill_(0.0)

        # Costruiamo i layers hidden
        self.hidden_layers = []
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers.append(nn.ReLU())

        # Aggiungiamo il layer di output
        self.output_layer = nn.Linear(hidden_dim, 1)

        # Creiamo il modello sequenziale
        layers = [self.first_layer] + self.hidden_layers + [self.output_layer]
        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)

    def train_xgboost(self):
        xgb_model = xgb.XGBRegressor(device = "cuda",**self.params)
        xgb_model.fit(self.X_train, self.y_train)
        return xgb_model.feature_importances_


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
grouped_df = pd.read_csv(data_path.joinpath("features.csv"))
target = "band_gap_ev"
features_list = [
    "area",
    "perimeter",
    "circularity",
    "solidity",
    "feret_diameter",
    "number_of_edges",
    "edge_density",
    "GLCM_correlation",
    "GLCM_energy",
    "GLCM_homogeneity",
    "GLCM_contrast",
]

X = grouped_df[features_list].values

y = grouped_df[target].values.flatten()  # array con tutti i valori raget total_energy

# Addestriamo il modello XGBoost
params = {
    "n_estimators": 770,
    "learning_rate": 0.010562915246990538,
    "max_depth": 10,
    "gamma": 0.0002985546393874009,
    "min_child_weight": 1.8685613953859606,
    "colsample_bytree": 0.9633894246849674,
    "subsample": 0.5195778253826636,
}
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.2, random_state=111
)


X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
input_dim = X_train.shape[1]
neural_net = NeuralNet(
    input_dim=input_dim,
    num_hidden_layers=10,
    hidden_dim=64,
    X_train=X_train_tensor,
    y_train=y_train_tensor,
    params=params,
).to(device=device)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(neural_net.parameters(), lr=1e-2)
# #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
#optimizer = torch.optim.SGD(neural_net.parameters(), lr=0.001, momentum=0.9)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

for epoch in tqdm(range(4000)):
    optimizer.zero_grad()
    outputs = neural_net(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    if epoch % 50 == 0:
        print(loss)
        print(scheduler.get_last_lr())

# Facciamo previsioni con la rete neurale addestrata
neural_net.eval()
# Fai previsioni sulla CPU
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

with torch.no_grad():
    neural_predictions = (
        neural_net(torch.tensor(X_test_tensor, dtype=torch.float32))
        .cpu()
        .squeeze()
        .numpy()
    )

# Valutiamo le prestazioni della rete neurale
neural_mse = np.mean((neural_predictions - y_test) ** 2)
residual_errors = y_test - neural_predictions
mse = mean_squared_error(y_test, neural_predictions)
mae = mean_absolute_error(y_test, neural_predictions)
r2 = r2_score(y_test, neural_predictions)
print("r2", r2)
print("mse", mse)
print("mae", mae)

# Plot sulla CPU
plt.figure(figsize=(8, 8))
sns.regplot(
    x=np.ravel(y_test),
    y=np.ravel(neural_predictions),
    scatter_kws={"s": 140},
    color=[31 / 255, 44 / 255, 73 / 255],
    line_kws={"color": "green"},
)
plt.ylabel("Predicted Values", fontsize=40)
plt.xlabel("True Values", fontsize=40)
plt.tick_params(axis="both", which="major", length=20, width=2)  # Adjust tick size

for spine in plt.gca().spines.values():
    spine.set_linewidth(2)
plt.tight_layout()
plt.savefig(f"ensable_pred_{target}.png")

plt.figure(figsize=(8, 8))
sns.histplot(
    residual_errors,
    kde=True,
    color=[31 / 255, 44 / 255, 73 / 255],
    bins=int(np.sqrt(X.shape[0])),
    line_kws={"color": "green"},
)
plt.tight_layout()
plt.savefig(f"res_errors{target}.png")