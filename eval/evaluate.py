# eval/evaluate.py
import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

DATA_OUT = os.path.join(BASE_DIR, "data_out")
MODELS_OUT = os.path.join(BASE_DIR, "models_out")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

from models.bi_lstm_model import build_bi_lstm_model
from models.param_transforms import sample_theta_from_gaussian_z

eps = 1e-8

# -------- Load data --------
X = np.load(os.path.join(DATA_OUT, "features.npy"), mmap_mode="r")
y_theta = np.load(os.path.join(DATA_OUT, "params.npy"), mmap_mode="r")

meta = np.load(os.path.join(DATA_OUT, "tfr_meta.npz"))
n_time = int(meta["n_time_patches"])
n_freq = int(meta["n_freq_patches"])
n_tokens_erp = int(meta["n_tokens_erp"])
time_centers = meta["time_patch_centers"]
freq_centers = meta["freq_patch_centers"]

N, n_tokens, feature_dim = X.shape
P = y_theta.shape[1]

# -------- Load model + scaler + bounds --------
model = load_model(os.path.join(MODELS_OUT, "jr_paramtoken_inverse_model.keras"), compile=False)

with open(os.path.join(MODELS_OUT, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

bounds = np.load(os.path.join(MODELS_OUT, "param_bounds.npz"))
param_names = [x.decode("utf-8") for x in bounds["param_names"]]
low = bounds["prior_low"].astype(np.float32)
high = bounds["prior_high"].astype(np.float32)

# -------- Split (consistent with training split seed) --------
idx = np.arange(N)
_, test_idx, _, y_test = train_test_split(
    idx, np.asarray(y_theta, dtype=np.float32), test_size=0.15, random_state=42
)

def scale_X(X_batch):
    flat = X_batch.reshape(-1, feature_dim)
    flat_s = scaler.transform(flat).astype(np.float32)
    return flat_s.reshape(X_batch.shape[0], n_tokens, feature_dim)

# -------- Predict in chunks --------
chunk = 128
preds = []
for start in range(0, len(test_idx), chunk):
    sl = test_idx[start:start + chunk]
    xb = np.asarray(X[sl], dtype=np.float32)
    xb = scale_X(xb)
    yp = model.predict(xb, verbose=0)
    preds.append(yp.astype(np.float32))
pred = np.concatenate(preds, axis=0)

mu_z = pred[:, :P]
logvar_z = np.clip(pred[:, P:], -10.0, 10.0)

# Posterior samples in theta-space
theta_samps = sample_theta_from_gaussian_z(mu_z, logvar_z, low, high, n_samples=200, seed=0)
theta_mean = theta_samps.mean(axis=0)
theta_std = theta_samps.std(axis=0)

# -------- Metrics (prof-friendly) --------
abs_err = np.abs(theta_mean - y_test)
rel_err = abs_err / (np.abs(y_test) + eps) * 100.0

rel_mean = rel_err.mean(axis=0)
rel_std = rel_err.std(axis=0)
rel_med = np.median(rel_err, axis=0)
rel_q05 = np.quantile(rel_err, 0.05, axis=0)
rel_q95 = np.quantile(rel_err, 0.95, axis=0)

# SNR (dB): nominal norm / error norm
snr_db = 20.0 * np.log10(
    np.linalg.norm(y_test, axis=0) / (np.linalg.norm(y_test - theta_mean, axis=0) + eps)
)

rmse = np.sqrt(np.mean((theta_mean - y_test) ** 2, axis=0))

# 90% interval coverage
lo = np.quantile(theta_samps, 0.05, axis=0)
hi = np.quantile(theta_samps, 0.95, axis=0)
coverage = ((y_test >= lo) & (y_test <= hi)).mean(axis=0)

# Posterior contraction
prior_std = (high - low) / np.sqrt(12.0)
contr = theta_std.mean(axis=0) / (prior_std + eps)

print("\n=== Test Relative Error (%) [mean ± std] ===")
for i, nm in enumerate(param_names):
    print(f"{nm:8s}: {rel_mean[i]:.2f}% ± {rel_std[i]:.2f}%")

print("\n=== Test Relative Error (%) [median, 5–95%] ===")
for i, nm in enumerate(param_names):
    print(f"{nm:8s}: {rel_med[i]:.2f}%  ( {rel_q05[i]:.2f}% – {rel_q95[i]:.2f}% )")

print("\n=== SNR per parameter (dB) ===")
for i, nm in enumerate(param_names):
    print(f"{nm:8s}: {snr_db[i]:.2f} dB")

print("\n=== Test RMSE (theta units) ===")
for i, nm in enumerate(param_names):
    print(f"{nm:8s}: {rmse[i]:.4f}")

print("\n=== 90% Interval Coverage ===")
for i, nm in enumerate(param_names):
    print(f"{nm:8s}: {coverage[i]*100:.1f}%")

print("\n=== Posterior Contraction (avg post std / prior std) ===")
for i, nm in enumerate(param_names):
    print(f"{nm:8s}: {contr[i]:.3f}")

# Save arrays for other scripts
np.save(os.path.join(PLOTS_DIR, "theta_mean.npy"), theta_mean)
np.save(os.path.join(PLOTS_DIR, "theta_std.npy"), theta_std)
np.save(os.path.join(PLOTS_DIR, "rel_err_mean.npy"), rel_mean)
np.save(os.path.join(PLOTS_DIR, "rel_err_median.npy"), rel_med)
np.save(os.path.join(PLOTS_DIR, "snr_db.npy"), snr_db)
np.save(os.path.join(PLOTS_DIR, "coverage90.npy"), coverage)
np.save(os.path.join(PLOTS_DIR, "contraction.npy"), contr)

# -------- Attention maps (optional) --------
try:
    attn_model = build_bi_lstm_model(
        n_tokens=n_tokens,
        feature_dim=feature_dim,
        n_params=P,
        n_time_patches=n_time,
        n_freq_patches=n_freq,
        n_tokens_erp=n_tokens_erp,   # IMPORTANT FIX
        d_model=128,
        num_layers=4,
        num_heads=4,
        ff_dim=256,
        dropout_rate=0.15,
        return_attention=True,
    )
    attn_model.set_weights(model.get_weights())

    viz = min(5, len(test_idx))
    viz_ids = test_idx[:viz]
    xb = np.asarray(X[viz_ids], dtype=np.float32)
    xb = scale_X(xb)

    pred2, scores = attn_model.predict(xb, verbose=0)   # scores: (B, H, P, tokens)
    scores = scores.mean(axis=1).astype(np.float32)     # (B, P, tokens)

    for bi in range(viz):
        for pi, nm in enumerate(param_names):
            w = scores[bi, pi]  # (tokens,)

            w_erp = w[:n_tokens_erp]                     # (n_time,)
            w_tfr = w[n_tokens_erp:].reshape(n_time, n_freq).T   # (freq, time)

            plt.figure(figsize=(7, 2.5))
            plt.plot(time_centers, w_erp)
            plt.xlabel("Time (s) rel stim")
            plt.ylabel("Attention")
            plt.title(f"ERP relevance – sample {bi} – {nm}")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"attn_erp_s{bi}_{nm}.png"), dpi=200)
            plt.close()

            plt.figure(figsize=(8, 3))
            plt.imshow(
                w_tfr, aspect="auto", origin="lower",
                extent=[time_centers[0], time_centers[-1], freq_centers[0], freq_centers[-1]],
            )
            plt.colorbar(label="Attention")
            plt.xlabel("Time (s) rel stim")
            plt.ylabel("Frequency (Hz)")
            plt.title(f"TFR relevance – sample {bi} – {nm}")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"attn_tfr_s{bi}_{nm}.png"), dpi=200)
            plt.close()

    print("\nSaved metrics + attention plots to:", PLOTS_DIR)

except Exception as e:
    print("\n[WARN] Attention plotting skipped due to error:")
    print(" ", repr(e))
    print("Metrics were still saved to:", PLOTS_DIR)
