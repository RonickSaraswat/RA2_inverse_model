import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

DATA_FILE = os.path.join(BASE_DIR, "data", "synthetic_eeg_dataset.h5")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load saved predictions
theta_mean = np.load(os.path.join(PLOTS_DIR, "theta_mean.npy"))
theta_std = np.load(os.path.join(PLOTS_DIR, "theta_std.npy"))

# Load ground truth (test split recreated here)
y = np.load(os.path.join(BASE_DIR, "data_out", "params.npy"))
N = y.shape[0]
idx = np.arange(N)

from sklearn.model_selection import train_test_split
train_idx, test_idx, _, y_test = train_test_split(idx, y, test_size=0.15, random_state=42)

# Param names
bounds = np.load(os.path.join(BASE_DIR, "models_out", "param_bounds.npz"))
param_names = [x.decode("utf-8") for x in bounds["param_names"]]

# Scatter plots with uncertainty bars (subset)
subset = min(400, len(test_idx))
sel = test_idx[:subset]
y_true = y[sel]
y_pred = theta_mean[:subset]
y_unc = theta_std[:subset]

for i, nm in enumerate(param_names):
    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.4, s=10)
    lims = [min(y_true[:, i].min(), y_pred[:, i].min()),
            max(y_true[:, i].max(), y_pred[:, i].max())]
    plt.plot(lims, lims, "r--")
    plt.xlabel("True")
    plt.ylabel("Predicted mean")
    plt.title(f"{nm}: true vs predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"scatter_{nm}.png"), dpi=200)
    plt.close()

# ERP example plots from HDF5
with h5py.File(DATA_FILE, "r") as f:
    eeg = f["EEG"][:]  # (N, C, T)
    fs = float(f.attrs["fs"])
    stim_onset = float(f.attrs["stim_onset"])

t = np.arange(eeg.shape[2]) / fs

for s in range(min(3, eeg.shape[0])):
    plt.figure(figsize=(10, 4))
    offset = 0.0
    for ch in range(min(8, eeg.shape[1])):  # plot first 8 channels for readability
        plt.plot(t, eeg[s, ch] + offset)
        offset += 50.0
    plt.axvline(stim_onset, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("uV (offset per channel)")
    plt.title(f"ERP-like EEG example {s} (first 8 channels)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"erp_example_{s}.png"), dpi=200)
    plt.close()

print("Saved plots to:", PLOTS_DIR)
