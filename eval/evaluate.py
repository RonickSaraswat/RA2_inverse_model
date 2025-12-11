# eval/evaluate.py
import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_OUT = os.path.join(BASE_DIR, "data_out")
MODELS_OUT = os.path.join(BASE_DIR, "models_out")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# options
use_mc = True
mc_samples = 50
noise_levels = [0.0, 0.1, 0.5, 1.0]  # multiplies standard dev of features
bootstrap_sizes = [500, 1000, 2000, 5000, None]  # None = full

# load data
X = np.load(os.path.join(DATA_OUT, "features.npy"))
y = np.load(os.path.join(DATA_OUT, "params.npy"))

# load model & scaler
model = load_model(os.path.join(MODELS_OUT, "bi_lstm_inverse_model.keras"))
with open(os.path.join(MODELS_OUT, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

# reshape features to (N, T, F)
T = 10
F_total = X.shape[1]
F = F_total // T
X = X[:, :T*F].reshape(X.shape[0], T, F)
X_scaled = scaler.transform(X.reshape(-1, F)).reshape(X.shape)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42)

def predict_with_mc(model, X, mc_samples=50):
    preds = [model(X, training=True).numpy() for _ in range(mc_samples)]
    preds = np.stack(preds, axis=0)  # shape (mc, N, P)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std

# main eval
if use_mc:
    y_mean, y_std = predict_with_mc(model, X_test, mc_samples=mc_samples)
else:
    y_mean = model.predict(X_test)
    y_std = np.zeros_like(y_mean)

mse = mean_squared_error(y_test, y_mean)
print("Test MSE:", mse)

# save basic result table: first 10 examples
for i in range(min(10, len(y_test))):
    print("True:", y_test[i], "Pred:", y_mean[i], "Std:", y_std[i].mean())

# noise-sensitivity: add gaussian noise in feature space (not raw EEG)
base_feat = X_test.copy()
mse_noise = []
for nl in noise_levels:
    noisy = base_feat + nl * np.std(base_feat) * np.random.randn(*base_feat.shape)
    if use_mc:
        pred_mean, _ = predict_with_mc(model, noisy, mc_samples=mc_samples)
    else:
        pred_mean = model.predict(noisy)
    mse_n = mean_squared_error(y_test, pred_mean)
    mse_noise.append(mse_n)
    print(f"Noise level {nl}: MSE={mse_n}")

# bootstrap MSE vs dataset size using repeated training subsets 
mse_vs_size = {}
N_total = X_scaled.shape[0]
for sz in bootstrap_sizes:
    if sz is None or sz >= N_total:
        # wil use full dataset
        Xb = X_scaled
        yb = y
    else:
        idx = np.random.choice(N_total, sz, replace=False)
        Xb = X_scaled[idx]
        yb = y[idx]
    # train/test split
    Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.15, random_state=42)
    # retrain small model for benchmarking (fewer epochs)
    from models.bi_lstm_model import build_bi_lstm_model
    m = build_bi_lstm_model(T, F, y.shape[1], dropout_rate=0.1, use_mc_dropout=False)
    m.compile(optimizer='adam', loss='mse')
    m.fit(Xtr, ytr, epochs=10, batch_size=32, verbose=0)
    ypred = m.predict(Xte)
    mse_vs_size[str(sz)] = mean_squared_error(yte, ypred)
    print("Train size", sz, "-> MSE", mse_vs_size[str(sz)])

# Save plots for noise and bootstrap
plt.figure()
plt.plot(noise_levels, mse_noise, marker='o')
plt.xlabel("Noise multiplier (feature space)")
plt.ylabel("Test MSE")
plt.title("Noise sensitivity")
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, "noise_sensitivity.png"), dpi=200)
plt.close()

sizes = [int(k) if k != 'None' else X_scaled.shape[0] for k in mse_vs_size.keys()]
mses = [mse_vs_size[k] for k in mse_vs_size.keys()]
plt.figure()
plt.plot(sizes, mses, marker='o')
plt.xlabel("Training set size")
plt.ylabel("MSE (bootstrap short retrain)")
plt.title("MSE vs training size (quick bootstrap)")
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, "mse_vs_training_size.png"), dpi=200)
plt.close()

print("Saved noise and bootstrap plots to", PLOTS_DIR)
