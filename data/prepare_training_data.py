# data/prepare_training_data.py
import os
import sys
import time
import numpy as np
import h5py
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from features.feature_extraction import extract_features

DATA_FILE = os.path.join(BASE_DIR, "data", "synthetic_eeg_dataset.h5")
OUT_DIR = os.path.join(BASE_DIR, "data_out")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- TFR settings ----
fs = 250
fmin = 1.0
fmax = 45.0
n_freqs = 30
decim = 4
n_cycles = None

# ---- Window around stimulus (seconds) ----
pre_sec = 1.0
post_sec = 1.0

# ---- Patch sizes ----
FREQ_PATCH = 2
TIME_PATCH = 5

print("Preparing training data (HYBRID tokens: ERP + TFR patches)...")
print("Input:", DATA_FILE)
print("Output:", OUT_DIR)

# ---------- Load data + priors SAFELY ----------
with h5py.File(DATA_FILE, "r") as f:
    eeg = f["EEG"][:]         # (N, C, T)
    params = f["params"][:]   # (N, P)
    stim_onset = float(f.attrs["stim_onset"])
    param_names = [x.decode("utf-8") for x in f.attrs["param_names"]]

    # IMPORTANT: copy priors while file is open (attrs handle becomes invalid after close)
    priors_attrs = dict(f["priors"].attrs.items())

N, C, T = eeg.shape
P = params.shape[1]

print("EEG shape:", eeg.shape)
print("Params shape:", params.shape)
print("Stim onset:", stim_onset, "sec")
print("Param names:", param_names)

# Build prior arrays (now safe; priors_attrs is a normal dict)
prior_low = np.array([float(priors_attrs[nm][0]) for nm in param_names], dtype=np.float32)
prior_high = np.array([float(priors_attrs[nm][1]) for nm in param_names], dtype=np.float32)

# ---- Determine shapes from first sample ----
tfr0 = extract_features(
    eeg[0],
    fs=fs, fmin=fmin, fmax=fmax, n_freqs=n_freqs, n_cycles=n_cycles, decim=decim
)
C0, F0, Tdec_full = tfr0.shape

win_start = stim_onset - pre_sec
win_end = stim_onset + post_sec
start_idx = int(np.round(win_start * fs / decim))
end_idx = int(np.round(win_end * fs / decim))
start_idx = max(0, start_idx)
end_idx = min(Tdec_full, end_idx)

Tdec_win = end_idx - start_idx

n_f = (F0 // FREQ_PATCH)
n_t = (Tdec_win // TIME_PATCH)
Freq_use = n_f * FREQ_PATCH
T_use = n_t * TIME_PATCH

print("\nTFR full:", tfr0.shape, "(C, Freq, Tdec_full)")
print("Window idx:", start_idx, ":", end_idx, "=> Tdec_win", Tdec_win)
print("Using Freq_use =", Freq_use, "of", F0, "with FREQ_PATCH =", FREQ_PATCH, "=> n_f =", n_f)
print("Using T_use    =", T_use, "of", Tdec_win, "with TIME_PATCH =", TIME_PATCH, "=> n_t =", n_t)

# Tokenization:
# ERP tokens: n_t tokens, each token is (C,) averaged in TIME_PATCH bins
# TFR tokens: n_t*n_f tokens, each token is (C,) averaged over (freq_patch,time_patch)
n_tokens_erp = n_t
n_tokens_tfr = n_t * n_f
n_tokens = n_tokens_erp + n_tokens_tfr

print("Tokens ERP:", n_tokens_erp, "Tokens TFR:", n_tokens_tfr, "Total tokens:", n_tokens)
print("Feature dim per token:", C)

# Time centers (seconds relative to stimulus)
t_dec = (np.arange(Tdec_full) * decim / fs) - stim_onset
t_win = t_dec[start_idx:start_idx + T_use]
time_patch_centers = []
for j in range(n_t):
    sl = slice(j * TIME_PATCH, (j + 1) * TIME_PATCH)
    time_patch_centers.append(float(t_win[sl].mean()))
time_patch_centers = np.array(time_patch_centers, dtype=np.float32)

# Frequency patch centers
freqs = np.linspace(fmin, fmax, n_freqs).astype(np.float32)
freq_use = freqs[:Freq_use]
freq_patch_centers = []
for k in range(n_f):
    sl = slice(k * FREQ_PATCH, (k + 1) * FREQ_PATCH)
    freq_patch_centers.append(float(freq_use[sl].mean()))
freq_patch_centers = np.array(freq_patch_centers, dtype=np.float32)

# Token types: 0 = ERP, 1 = TFR
token_types = np.zeros((n_tokens,), dtype=np.int32)
token_types[n_tokens_erp:] = 1

X = np.zeros((N, n_tokens, C), dtype=np.float32)
y = np.asarray(params, dtype=np.float32)

start_time = time.time()

def patchify_tfr(tfr):
    # tfr: (C, F, Tdec_full)
    tfr_win = tfr[:, :Freq_use, start_idx:start_idx + T_use]  # (C, Freq_use, T_use)
    rs = tfr_win.reshape(C, n_f, FREQ_PATCH, n_t, TIME_PATCH)
    patch = rs.mean(axis=(2, 4))  # (C, n_f, n_t)
    tokens = patch.transpose(2, 1, 0).reshape(n_t * n_f, C).astype(np.float32)
    return tokens

def erp_tokens_from_eeg(eeg_sample):
    eeg_dec = eeg_sample[:, ::decim]                         # (C, Tdec_full approx)
    win = eeg_dec[:, start_idx:start_idx + T_use]            # (C, T_use)
    patch = win.reshape(C, n_t, TIME_PATCH).mean(axis=2)     # (C, n_t)
    tokens = patch.T.astype(np.float32)                      # (n_t, C)
    return tokens

print(f"\nProcessing {N} samples...")
for i in tqdm(range(N), desc="Processing samples", unit="sample"):
    erp_tok = erp_tokens_from_eeg(eeg[i])  # (n_t, C)

    tfr = extract_features(
        eeg[i],
        fs=fs, fmin=fmin, fmax=fmax, n_freqs=n_freqs, n_cycles=n_cycles, decim=decim
    )
    tfr_tok = patchify_tfr(tfr)  # (n_t*n_f, C)

    X[i, :n_tokens_erp, :] = erp_tok
    X[i, n_tokens_erp:, :] = tfr_tok

elapsed = time.time() - start_time

np.save(os.path.join(OUT_DIR, "features.npy"), X)
np.save(os.path.join(OUT_DIR, "params.npy"), y)

np.savez(
    os.path.join(OUT_DIR, "tfr_meta.npz"),
    fs=fs,
    decim=decim,
    fmin=fmin,
    fmax=fmax,
    n_freqs=n_freqs,
    stim_onset=stim_onset,
    pre_sec=pre_sec,
    post_sec=post_sec,
    freq_patch=FREQ_PATCH,
    time_patch=TIME_PATCH,
    n_time_patches=n_t,
    n_freq_patches=n_f,
    n_tokens=n_tokens,
    n_tokens_erp=n_tokens_erp,
    n_tokens_tfr=n_tokens_tfr,
    token_types=token_types,
    time_patch_centers=time_patch_centers,
    freq_patch_centers=freq_patch_centers,
)

np.savez(
    os.path.join(OUT_DIR, "param_meta.npz"),
    param_names=np.array(param_names, dtype="S"),
    prior_low=prior_low,
    prior_high=prior_high,
)

print("\nSaved:")
print(" -", os.path.join(OUT_DIR, "features.npy"))
print(" -", os.path.join(OUT_DIR, "params.npy"))
print(" -", os.path.join(OUT_DIR, "tfr_meta.npz"))
print(" -", os.path.join(OUT_DIR, "param_meta.npz"))
print(f"Total time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
print("Final X shape:", X.shape, "y shape:", y.shape)
