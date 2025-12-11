# eval/plot_results.py
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py
from tensorflow.keras.models import load_model
from scipy.signal import welch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_OUT = os.path.join(BASE_DIR, "data_out")
DATA_FILE = os.path.join(BASE_DIR, "data", "synthetic_eeg_dataset.h5")
MODEL_PATH = os.path.join(BASE_DIR, "models_out", "bi_lstm_inverse_model.keras")
HIST_PATH = os.path.join(BASE_DIR, "models_out", "training_history.pkl")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

X = np.load(os.path.join(DATA_OUT, "features.npy"))
y_true = np.load(os.path.join(DATA_OUT, "params.npy"))
model = load_model(MODEL_PATH)

T = 10
F = X.shape[1] // T
X_seq = X[:, :T*F].reshape(X.shape[0], T, F)
y_pred = model.predict(X_seq)

# 1 Loss curve
if os.path.exists(HIST_PATH):
    with open(HIST_PATH, "rb") as f:
        hist = pickle.load(f)
    plt.figure()
    plt.plot(hist["loss"], label="train")
    plt.plot(hist["val_loss"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("MSE")
    plt.legend(); plt.title("Loss curve")
    plt.savefig(os.path.join(PLOTS_DIR, "loss_curve.png"), dpi=200)
    plt.close()

# 2 Scatter per parameter
n_params = y_true.shape[1]
for i in range(n_params):
    plt.figure(figsize=(4,4))
    plt.scatter(y_true[:,i], y_pred[:,i], alpha=0.4)
    mn = min(y_true[:,i].min(), y_pred[:,i].min())
    mx = max(y_true[:,i].max(), y_pred[:,i].max())
    plt.plot([mn,mx],[mn,mx], 'r--')
    plt.xlabel("True"); plt.ylabel("Pred")
    plt.title(f"Param {i+1} true vs pred")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"param_scatter_param{i+1}.png"), dpi=200)
    plt.close()

# 3 Param timeseries first samples
samples_to_plot = 5
for i in range(n_params):
    plt.figure()
    plt.plot(range(samples_to_plot), y_true[:samples_to_plot,i], 'o-', label='true')
    plt.plot(range(samples_to_plot), y_pred[:samples_to_plot,i], 'x--', label='pred')
    plt.legend(); plt.title(f"Param {i+1} (first {samples_to_plot} samples)")
    plt.savefig(os.path.join(PLOTS_DIR, f"param_timeseries_param{i+1}.png"), dpi=200)
    plt.close()

# 4 Example EEG traces + PSD of a channel
with h5py.File(DATA_FILE, "r") as f:
    eeg = f["EEG"][:]  # (N, ch, t)
n_examples = min(3, eeg.shape[0])
for s in range(n_examples):
    plt.figure(figsize=(10,4))
    offset = 0
    for ch in range(eeg.shape[1]):
        plt.plot(eeg[s,ch,:] + offset, label=f"Ch{ch+1}")
        offset += 100  # bigger offset for visualization
    plt.title(f"Example EEG sample {s}"); plt.xlabel("Samples")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"example_eeg_sample{s}.png"), dpi=200)
    plt.close()

# PSD of first channel of first sample
fs=250
f, pxx = welch(eeg[0,0,:], fs=fs, nperseg=512)
plt.figure()
plt.semilogy(f, pxx)
plt.xlim(0,60)
plt.xlabel("Hz"); plt.ylabel("PSD")
plt.title("PSD channel 1 sample 0")
plt.savefig(os.path.join(PLOTS_DIR, "eeg_power_spectrum.png"), dpi=200)
plt.close()

# 5 Feature histogram
plt.figure(figsize=(8,4))
plt.hist(X.flatten(), bins=80, color='skyblue')
plt.title("Feature distribution"); plt.xlabel("Value"); plt.ylabel("Count")
plt.savefig(os.path.join(PLOTS_DIR, "feature_histograms.png"), dpi=200)
plt.close()

print("Saved plots to", PLOTS_DIR)
