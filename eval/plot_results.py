import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import h5py

from sklearn.model_selection import train_test_split
from scipy.stats import norm

import tensorflow as tf
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

DATA_FILE  = os.path.join(BASE_DIR, "data", "synthetic_eeg_dataset.h5")
DATA_OUT   = os.path.join(BASE_DIR, "data_out")
MODELS_OUT = os.path.join(BASE_DIR, "models_out")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

FEATURES_PATH = os.path.join(DATA_OUT, "features.npy")
PARAMS_PATH   = os.path.join(DATA_OUT, "params.npy")

MODEL_PATH  = os.path.join(MODELS_OUT, "jr_paramtoken_inverse_model.keras")
SCALER_PATH = os.path.join(MODELS_OUT, "scaler.pkl")
BOUNDS_PATH = os.path.join(MODELS_OUT, "param_bounds.npz")
HIST_PATH   = os.path.join(MODELS_OUT, "training_history.pkl")


def scale_tokens(X_batch, scaler):
    B, T, F = X_batch.shape
    flat = X_batch.reshape(-1, F)
    flat_s = scaler.transform(flat).astype(np.float32)
    return flat_s.reshape(B, T, F)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def theta_to_z(theta, low, high, eps=1e-6):
    x = (theta - low) / (high - low)
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def z_to_theta(z, low, high):
    s = sigmoid(z)
    return low + (high - low) * s


def predict_split(X_memmap, y_theta, indices, model, scaler, low, high,
                  batch_size=32, mc_samples=64, seed=0):
    rng = np.random.default_rng(seed)
    P = y_theta.shape[1]
    n = len(indices)

    theta_mean_all = np.zeros((n, P), dtype=np.float32)
    theta_std_all  = np.zeros((n, P), dtype=np.float32)
    mu_z_all       = np.zeros((n, P), dtype=np.float32)
    std_z_all      = np.zeros((n, P), dtype=np.float32)
    z_true_all     = np.zeros((n, P), dtype=np.float32)
    nll_all        = np.zeros((n,), dtype=np.float32)
    pit_all        = np.zeros((n, P), dtype=np.float32)

    out_ptr = 0
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        xb = np.asarray(X_memmap[batch_idx], dtype=np.float32)  # (B,T,F)
        yb = np.asarray(y_theta[batch_idx], dtype=np.float32)   # (B,P)
        xb = scale_tokens(xb, scaler)

        pred = model.predict(xb, verbose=0).astype(np.float32)  # (B,2P)
        mu_z = pred[:, :P]
        logvar_z = pred[:, P:]
        logvar_z = np.clip(logvar_z, -10.0, 10.0)
        std_z = np.exp(0.5 * logvar_z)

        z_true = theta_to_z(yb, low, high)

        inv_var = np.exp(-logvar_z)
        nll = 0.5 * (inv_var * (z_true - mu_z) ** 2 + logvar_z + np.log(2.0 * np.pi))
        nll = np.sum(nll, axis=1)

        pit_u = norm.cdf((z_true - mu_z) / (std_z + 1e-8)).astype(np.float32)

        S = mc_samples
        eps = rng.standard_normal(size=(S, mu_z.shape[0], P)).astype(np.float32)
        z_samp = mu_z[None, :, :] + eps * std_z[None, :, :]
        theta_samp = z_to_theta(z_samp, low, high)

        theta_mean = theta_samp.mean(axis=0).astype(np.float32)
        theta_std  = theta_samp.std(axis=0).astype(np.float32)

        B = mu_z.shape[0]
        theta_mean_all[out_ptr:out_ptr+B] = theta_mean
        theta_std_all[out_ptr:out_ptr+B]  = theta_std
        mu_z_all[out_ptr:out_ptr+B]       = mu_z
        std_z_all[out_ptr:out_ptr+B]      = std_z
        z_true_all[out_ptr:out_ptr+B]     = z_true
        nll_all[out_ptr:out_ptr+B]        = nll.astype(np.float32)
        pit_all[out_ptr:out_ptr+B]        = pit_u
        out_ptr += B

    return {
        "theta_mean": theta_mean_all,
        "theta_std": theta_std_all,
        "mu_z": mu_z_all,
        "std_z": std_z_all,
        "z_true": z_true_all,
        "nll_z": nll_all,
        "pit_u": pit_all,
    }


def main():
    print("Loading data...")
    X = np.load(FEATURES_PATH, mmap_mode="r")  # (N,tokens,feat)
    y = np.load(PARAMS_PATH, mmap_mode="r")    # (N,P)
    N, n_tokens, feat_dim = X.shape
    P = y.shape[1]
    idx = np.arange(N)

    bounds = np.load(BOUNDS_PATH)
    param_names = [x.decode("utf-8") for x in bounds["param_names"]]
    low = bounds["prior_low"].astype(np.float32)
    high = bounds["prior_high"].astype(np.float32)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Load model safely
    try:
        from models import bi_lstm_model  # noqa: F401
    except Exception:
        pass
    model = load_model(MODEL_PATH, compile=False)

    # Train/test split (yes, you are splitting data)
    train_idx, test_idx = train_test_split(idx, test_size=0.15, random_state=42)
    print(f"Split: train={len(train_idx)} test={len(test_idx)}")

    # Predict
    print("Predicting train...")
    train_out = predict_split(X, y, train_idx, model, scaler, low, high, seed=123)
    print("Predicting test...")
    test_out  = predict_split(X, y, test_idx, model, scaler, low, high, seed=456)

    # Save indices for reproducibility
    np.save(os.path.join(PLOTS_DIR, "train_idx.npy"), train_idx)
    np.save(os.path.join(PLOTS_DIR, "test_idx.npy"), test_idx)

    # Training curve (if available)
    if os.path.exists(HIST_PATH):
        with open(HIST_PATH, "rb") as f:
            hist = pickle.load(f)
        if "loss" in hist and "val_loss" in hist:
            plt.figure()
            plt.plot(hist["loss"], label="train")
            plt.plot(hist["val_loss"], label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Gaussian NLL (z-space)")
            plt.title("Training curve")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, "loss_curve_train_val.png"), dpi=200)
            plt.close()

    # Scatter plots
    y_train = np.asarray(y[train_idx], dtype=np.float32)
    y_test  = np.asarray(y[test_idx], dtype=np.float32)

    for i, nm in enumerate(param_names):
        # Train scatter
        plt.figure(figsize=(4.5, 4.5))
        plt.scatter(y_train[:, i], train_out["theta_mean"][:, i], alpha=0.25, s=10)
        lims = [
            min(y_train[:, i].min(), train_out["theta_mean"][:, i].min()),
            max(y_train[:, i].max(), train_out["theta_mean"][:, i].max()),
        ]
        plt.plot(lims, lims, "r--", linewidth=1)
        plt.xlabel("True")
        plt.ylabel("Predicted mean")
        plt.title(f"TRAIN scatter — {nm}")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"scatter_train_{nm}.png"), dpi=200)
        plt.close()

        # Test scatter
        plt.figure(figsize=(4.5, 4.5))
        plt.scatter(y_test[:, i], test_out["theta_mean"][:, i], alpha=0.35, s=10)
        lims = [
            min(y_test[:, i].min(), test_out["theta_mean"][:, i].min()),
            max(y_test[:, i].max(), test_out["theta_mean"][:, i].max()),
        ]
        plt.plot(lims, lims, "r--", linewidth=1)
        plt.xlabel("True")
        plt.ylabel("Predicted mean")
        plt.title(f"TEST scatter — {nm}")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"scatter_test_{nm}.png"), dpi=200)
        plt.close()

    # Overconfidence diagnostics: PIT + standardized residuals + NLL
    for i, nm in enumerate(param_names):
        # PIT hist
        plt.figure(figsize=(6, 3))
        plt.hist(train_out["pit_u"][:, i], bins=30, alpha=0.6, label="train")
        plt.hist(test_out["pit_u"][:, i], bins=30, alpha=0.6, label="test")
        plt.xlabel("PIT u = Φ((z_true−μ)/σ)")
        plt.ylabel("Count")
        plt.title(f"PIT histogram — {nm}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"pit_{nm}.png"), dpi=200)
        plt.close()

        # Standardized residuals
        r_train = (train_out["z_true"][:, i] - train_out["mu_z"][:, i]) / (train_out["std_z"][:, i] + 1e-8)
        r_test  = (test_out["z_true"][:, i] - test_out["mu_z"][:, i]) / (test_out["std_z"][:, i] + 1e-8)

        plt.figure(figsize=(6, 3))
        plt.hist(r_train, bins=40, alpha=0.6, density=True,
                 label=f"train std={np.std(r_train):.2f}")
        plt.hist(r_test, bins=40, alpha=0.6, density=True,
                 label=f"test std={np.std(r_test):.2f}")
        xs = np.linspace(-5, 5, 400)
        plt.plot(xs, norm.pdf(xs, 0, 1), "k--", linewidth=1, label="N(0,1)")
        plt.xlim(-5, 5)
        plt.xlabel("r = (z_true−μ)/σ")
        plt.ylabel("Density")
        plt.title(f"Std residuals (overconf if std>1) — {nm}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"std_residuals_{nm}.png"), dpi=200)
        plt.close()

    plt.figure(figsize=(6, 3))
    plt.hist(train_out["nll_z"], bins=40, alpha=0.6,
             label=f"train mean={train_out['nll_z'].mean():.2f}")
    plt.hist(test_out["nll_z"], bins=40, alpha=0.6,
             label=f"test mean={test_out['nll_z'].mean():.2f}")
    plt.xlabel("Gaussian NLL (z-space) per sample")
    plt.ylabel("Count")
    plt.title("Train vs Test NLL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "nll_train_vs_test.png"), dpi=200)
    plt.close()

    # ERP plots requested by your professor
    with h5py.File(DATA_FILE, "r") as f:
        eeg = f["EEG"][:]  # (N,C,T) ok for N=1000
        fs = float(f.attrs["fs"])
        stim_onset = float(f.attrs["stim_onset"])

    t = np.arange(eeg.shape[2]) / fs

    # Example ERPs (use a few TEST samples)
    ex = test_idx[:3]
    for j, s in enumerate(ex):
        plt.figure(figsize=(10, 4))
        offset = 0.0
        for ch in range(min(8, eeg.shape[1])):
            plt.plot(t, eeg[s, ch] + offset, linewidth=1.0)
            offset += 50.0
        plt.axvline(stim_onset, color="k", linestyle="--", linewidth=1)
        plt.xlabel("Time (s)")
        plt.ylabel("uV (offset per channel)")
        plt.title(f"Example ERP-like EEG (test sample {int(s)}) — first 8 channels")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"erp_example_test_{j}.png"), dpi=200)
        plt.close()

    # Grand-average ERP over TEST set (baseline-corrected)
    pre = 0.5  # seconds baseline window
    pre_mask = (t >= (stim_onset - pre)) & (t < stim_onset)
    eeg_test = eeg[test_idx]  # (Ntest, C, T)

    baseline = eeg_test[:, :, pre_mask].mean(axis=2, keepdims=True)
    eeg_bc = eeg_test - baseline  # baseline corrected

    ga = eeg_bc.mean(axis=0)  # (C,T)
    plt.figure(figsize=(10, 4))
    offset = 0.0
    for ch in range(min(8, ga.shape[0])):
        plt.plot(t, ga[ch] + offset, linewidth=1.2)
        offset += 30.0
    plt.axvline(stim_onset, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("uV (grand average, offset)")
    plt.title("Grand-average ERP (TEST set, baseline-corrected) — first 8 channels")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "erp_grand_average_test.png"), dpi=200)
    plt.close()

    # Sanity: stim_amp vs ERP amplitude proxy (peak-to-peak in post window)
    # Use true stim_amp in y_test (param index found via param_names)
    if "stim_amp" in param_names:
        stim_i = param_names.index("stim_amp")
        y_test_theta = y_test
        stim_amp_true = y_test_theta[:, stim_i]

        post_mask = (t >= stim_onset) & (t <= stim_onset + 0.6)
        # global field power-like proxy: RMS across channels, then peak-to-peak
        gfp = np.sqrt((eeg_bc[:, :, post_mask] ** 2).mean(axis=1))  # (Ntest, Tpost)
        p2p = gfp.max(axis=1) - gfp.min(axis=1)

        plt.figure(figsize=(5, 4))
        plt.scatter(stim_amp_true, p2p, s=12, alpha=0.5)
        plt.xlabel("True stim_amp")
        plt.ylabel("ERP amplitude proxy (GFP peak-to-peak)")
        plt.title("Sanity: stim_amp vs ERP amplitude proxy (TEST)")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "stim_amp_vs_erp_amplitude.png"), dpi=200)
        plt.close()

    print("Saved plots to:", PLOTS_DIR)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.get_logger().setLevel("ERROR")
    main()
