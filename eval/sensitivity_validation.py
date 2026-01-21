# eval/sensitivity_validation.py
import os
import sys
import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

DATA_FILE = os.path.join(BASE_DIR, "data", "synthetic_eeg_dataset.h5")
DATA_OUT = os.path.join(BASE_DIR, "data_out")
MODELS_OUT = os.path.join(BASE_DIR, "models_out")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

from simulate.simulator import simulate_eeg
from features.feature_extraction import extract_features
from models.bi_lstm_model import build_bi_lstm_model

meta = np.load(os.path.join(DATA_OUT, "tfr_meta.npz"))
param_meta = np.load(os.path.join(DATA_OUT, "param_meta.npz"))

fs = int(meta["fs"])
decim = int(meta["decim"])
fmin = float(meta["fmin"])
fmax = float(meta["fmax"])
n_freqs = int(meta["n_freqs"])
stim_onset = float(meta["stim_onset"])
pre_sec = float(meta["pre_sec"])
post_sec = float(meta["post_sec"])
freq_patch = int(meta["freq_patch"])
time_patch = int(meta["time_patch"])
n_time = int(meta["n_time_patches"])
n_freq = int(meta["n_freq_patches"])
n_tokens_erp = int(meta["n_tokens_erp"])
time_centers = meta["time_patch_centers"]
freq_centers = meta["freq_patch_centers"]

param_names = [x.decode("utf-8") for x in param_meta["param_names"]]
P = len(param_names)

X = np.load(os.path.join(DATA_OUT, "features.npy"), mmap_mode="r")
_, n_tokens, feature_dim = X.shape

model = load_model(os.path.join(MODELS_OUT, "jr_paramtoken_inverse_model.keras"), compile=False)

with open(os.path.join(MODELS_OUT, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

attn_model = build_bi_lstm_model(
    n_tokens=n_tokens,
    feature_dim=feature_dim,
    n_params=P,
    n_time_patches=n_time,
    n_freq_patches=n_freq,
    n_tokens_erp=n_tokens_erp,
    d_model=128, num_layers=4, num_heads=4, ff_dim=256,
    dropout_rate=0.15,
    return_attention=True,
)
attn_model.set_weights(model.get_weights())

def scale_tokens(tokens):
    flat = tokens.reshape(-1, feature_dim)
    flat_s = scaler.transform(flat).astype(np.float32)
    return flat_s.reshape(1, n_tokens, feature_dim)

def patchify_hybrid(eeg_sample):
    # Match prepare_training_data.py exactly
    tfr = extract_features(eeg_sample, fs=fs, fmin=fmin, fmax=fmax, n_freqs=n_freqs, decim=decim)

    C, F, Tdec_full = tfr.shape
    start_idx = int(np.round((stim_onset - pre_sec) * fs / decim))
    start_idx = max(0, start_idx)

    Freq_use = n_freq * freq_patch
    T_use = n_time * time_patch

    # ERP tokens from decimated EEG
    eeg_dec = eeg_sample[:, ::decim]
    win = eeg_dec[:, start_idx:start_idx + T_use]               # (C, T_use)
    erp_patch = win.reshape(C, n_time, time_patch).mean(axis=2) # (C, n_time)
    erp_tok = erp_patch.T.astype(np.float32)                    # (n_time, C)

    # TFR tokens
    tfr_win = tfr[:, :Freq_use, start_idx:start_idx + T_use]
    rs = tfr_win.reshape(C, n_freq, freq_patch, n_time, time_patch)
    patch = rs.mean(axis=(2, 4))                                # (C, n_freq, n_time)
    tfr_tok = patch.transpose(2, 1, 0).reshape(n_time * n_freq, C).astype(np.float32)

    tokens = np.concatenate([erp_tok, tfr_tok], axis=0).astype(np.float32)  # (tokens, C)
    return tokens

# Choose one example
example_idx = 0
with h5py.File(DATA_FILE, "r") as f:
    eeg0 = f["EEG"][example_idx]
    params0 = f["params"][example_idx]
    leadfield = f["leadfield"][:]
    pnames_h5 = [x.decode("utf-8") for x in f.attrs["param_names"]]
    base_params = {pnames_h5[i]: float(params0[i]) for i in range(len(pnames_h5))}

    sim_args = dict(
        fs=int(f.attrs["fs"]),
        duration=float(f.attrs["duration"]),
        n_channels=int(f.attrs["n_channels"]),
        bandpass=tuple(f.attrs["bandpass"]),
        stim_onset=float(f.attrs["stim_onset"]),
        stim_sigma=float(f.attrs["stim_sigma"]),
        n_sources=int(f.attrs["n_sources"]),
        leadfield=leadfield,
        sensor_noise_std=float(f.attrs["sensor_noise_std"]),
        n_trials=int(f.attrs["n_trials"]),
        input_noise_std=float(f.attrs["input_noise_std"]),
    )

tokens0 = patchify_hybrid(eeg0)
xb = scale_tokens(tokens0)

# Attention
pred, scores = attn_model.predict(xb, verbose=0)
scores = scores.mean(axis=1)[0]  # (P, tokens)

# ---------- Gradient sensitivity (Jacobian) ----------
xb_tf = tf.convert_to_tensor(xb, dtype=tf.float32)

with tf.GradientTape() as tape:
    tape.watch(xb_tf)
    y_pred = model(xb_tf, training=False)     # (1, 2P)
    mu_z = y_pred[:, :P]                      # (1, P)

J = tape.jacobian(mu_z, xb_tf)                # (1, P, 1, tokens, feat)
if J is None:
    raise RuntimeError("Jacobian returned None (no gradient). This should not happen; check TF build/device.")

J = tf.squeeze(J, axis=(0, 2))                # (P, tokens, feat)
grads = tf.norm(J, axis=-1).numpy().astype(np.float32)  # (P, tokens)

# ---------- Finite-difference sensitivity (token change per param delta) ----------
delta_frac = 0.02
sens_fd = []

for i, nm in enumerate(pnames_h5):
    base = base_params[nm]
    delta = delta_frac * (abs(base) + 1e-6)

    params_plus = dict(base_params)
    params_plus[nm] = base + delta

    eeg_plus = simulate_eeg(params=params_plus, seed=123, **sim_args)
    tok_plus = patchify_hybrid(eeg_plus)

    diff = (tok_plus - tokens0) / delta
    s = np.linalg.norm(diff, axis=1).astype(np.float32)  # (tokens,)
    sens_fd.append(s)

sens_fd = np.stack(sens_fd, axis=0)  # (P, tokens)

def corr(a, b):
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return float((a * b).mean())

print("\nAttention vs finite-difference sensitivity correlations (ERP vs TFR blocks):")
for i, nm in enumerate(pnames_h5):
    att = scores[i]
    grd = grads[i]
    fd = sens_fd[i]

    att_erp, att_tfr = att[:n_tokens_erp], att[n_tokens_erp:]
    grd_erp, grd_tfr = grd[:n_tokens_erp], grd[n_tokens_erp:]
    fd_erp, fd_tfr = fd[:n_tokens_erp], fd[n_tokens_erp:]

    print(
        f"{nm:8s}:  att~fd ERP {corr(att_erp, fd_erp): .3f} | TFR {corr(att_tfr, fd_tfr): .3f}   "
        f"grad~fd ERP {corr(grd_erp, fd_erp): .3f} | TFR {corr(grd_tfr, fd_tfr): .3f}"
    )

# Save a few comparison plots
for i, nm in enumerate(pnames_h5[:min(4, P)]):
    att_tfr = scores[i][n_tokens_erp:].reshape(n_time, n_freq).T
    fd_tfr = sens_fd[i][n_tokens_erp:].reshape(n_time, n_freq).T
    grd_tfr = grads[i][n_tokens_erp:].reshape(n_time, n_freq).T

    plt.figure(figsize=(12, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(att_tfr, aspect="auto", origin="lower",
               extent=[time_centers[0], time_centers[-1], freq_centers[0], freq_centers[-1]])
    plt.title(f"Attention (TFR) – {nm}")
    plt.xlabel("Time (s)")
    plt.ylabel("Hz")

    plt.subplot(1, 3, 2)
    plt.imshow(grd_tfr, aspect="auto", origin="lower",
               extent=[time_centers[0], time_centers[-1], freq_centers[0], freq_centers[-1]])
    plt.title(f"Gradient sens (TFR) – {nm}")
    plt.xlabel("Time (s)")
    plt.ylabel("Hz")

    plt.subplot(1, 3, 3)
    plt.imshow(fd_tfr, aspect="auto", origin="lower",
               extent=[time_centers[0], time_centers[-1], freq_centers[0], freq_centers[-1]])
    plt.title(f"Finite-diff sens (TFR) – {nm}")
    plt.xlabel("Time (s)")
    plt.ylabel("Hz")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"sens_compare_{nm}.png"), dpi=200)
    plt.close()

print("\nSaved sensitivity comparison plots to:", PLOTS_DIR)
