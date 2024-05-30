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
    from torch.utils.data import DataLoader, Dataset, random_split
except Exception as e:
    print(f"Some module are missing: {e}\n")

data_path = Path().resolve().joinpath("data")
xyz_files_path = data_path.joinpath("xyz_files")
yolo_model_path = data_path.joinpath("models", "best_100_campioni_new.pt")
images_path = data_path.joinpath("images")
crops_path = data_path.joinpath("crops")

plt.style.use("seaborn-v0_8-paper")


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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

        self.hidden_layers = []
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers.append(nn.ReLU())

        self.output_layer = nn.Linear(hidden_dim, 1)

        layers = [self.first_layer] + self.hidden_layers + [self.output_layer]
        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)

    def train_xgboost(self):
        xgb_model = xgb.XGBRegressor(device="cuda", **self.params)
        xgb_model.fit(self.X_train, self.y_train)
        return xgb_model.feature_importances_


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
grouped_df = pd.read_csv(data_path.joinpath("features.csv"))
target = "fermi_level_ev"
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

# Addestriamo il modellte  XGBoost
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
    X_norm, y, test_size=0.2, random_state=3333
)


X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_subset, batch_size=256, shuffle=True)
val_dataloader = DataLoader(val_subset, batch_size=256, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)


train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)
input_dim = X_train.shape[1]
neural_net = NeuralNet(
    input_dim=input_dim,
    num_hidden_layers=20,
    hidden_dim=128,
    X_train=X_train_tensor,
    y_train=y_train_tensor,
    params=params,
).to(device=device)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(neural_net.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=50, verbose=True
)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.1)
# optimizer = torch.optim.SGD(neural_net.parameters(), lr=0.001, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
train_loss_list = []
val_loss_list = []
for epoch in tqdm(range(3000)):
    neural_net.train()
    # epoch_loss = 0.0
    train_loss_batch = []
    val_loss_batch = []
    for X_batch, y_batch in train_dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = neural_net(X_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()
        # epoch_loss += loss.item()
        train_loss_batch.append(loss.item())
    train_loss_list.append(np.mean(train_loss_batch))

    neural_net.eval()
    with torch.no_grad():
        for X_batch, y_batch in val_dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = neural_net(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss_batch.append(loss.item())
        val_loss_list.append(np.mean(val_loss_batch))
    scheduler.step(val_loss_list[-1])
    if epoch % 50 == 0:
        print(
            f"Epoch {epoch}, Train Loss: {train_loss_list[-1]}, Val Loss: {val_loss_list[-1]}, LR: {scheduler.get_last_lr()}"
        )


# Facciamo previsioni con la rete neurale addestrata
neural_net.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

with torch.no_grad():
    neural_predictions = neural_net(X_test_tensor).cpu().squeeze().numpy()

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
    stat="density",
    bins=int(np.sqrt(X.shape[0])),
    line_kws={"color": "green"},
)
plt.tight_layout()
plt.savefig(f"res_errors{target}.png")

# Plot della loss di addestramento e validazione
plt.figure(figsize=(10, 5))
plt.plot(train_loss_list, label="Train Loss")
plt.plot(val_loss_list, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(f"train_val_loss_{target}.png")
