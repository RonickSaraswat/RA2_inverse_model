from __future__ import annotations

import argparse
import os
import sys
import pickle
import logging
from typing import Dict, Tuple

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

DATA_OUT = os.path.join(BASE_DIR, "data_out")
MODELS_OUT = os.path.join(BASE_DIR, "models_out")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

from simulate.simulator import simulate_eeg  # noqa: E402
from features.feature_extraction import extract_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
LOGGER = logging.getLogger("generalization")


def _import_tf_or_die():
    try:
        import tensorflow as tf  # noqa: F401
    except ModuleNotFoundError as e:
        raise SystemExit("TensorFlow missing.") from e


def _load_model():
    _import_tf_or_die()
    from tensorflow.keras.models import load_model
    best = os.path.join(MODELS_OUT, "jr_paramtoken_inverse_model_best.keras")
    final = os.path.join(MODELS_OUT, "jr_paramtoken_inverse_model.keras")
    path = best if os.path.exists(best) else final
    try:
        import models.bi_lstm_model  # noqa: F401
    except Exception:
        pass
    return load_model(path, compile=False, safe_mode=False)


def _scale_tokens(tokens: np.ndarray, scaler) -> np.ndarray:
    # tokens: (N, T, C)
    n, t, c = tokens.shape
    flat = tokens.reshape(-1, c)
    flat_s = scaler.transform(flat).astype(np.float32)
    return flat_s.reshape(n, t, c)


def _make_leadfield(n_channels: int, n_sources: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lf = rng.normal(size=(n_channels, n_sources)).astype(np.float32)
    lf /= (np.linalg.norm(lf, axis=0, keepdims=True) + 1e-9)
    return lf


def _patchify_hybrid(eeg: np.ndarray, meta: Dict) -> np.ndarray:
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

    tfr = extract_features(eeg, fs=fs, fmin=fmin, fmax=fmax, n_freqs=n_freqs, decim=decim)
    C, F, Tdec_full = tfr.shape

    start_idx = int(np.round((stim_onset - pre_sec) * fs / decim))
    start_idx = max(0, start_idx)

    F_use = n_freq * freq_patch
    T_use = n_time * time_patch

    eeg_dec = eeg[:, ::decim]
    win = eeg_dec[:, start_idx:start_idx + T_use]
    erp_patch = win.reshape(C, n_time, time_patch).mean(axis=2)
    erp_tok = erp_patch.T.astype(np.float32)

    tfr_win = tfr[:, :F_use, start_idx:start_idx + T_use]
    rs = tfr_win.reshape(C, n_freq, freq_patch, n_time, time_patch)
    patch = rs.mean(axis=(2, 4))
    tfr_tok = patch.transpose(2, 1, 0).reshape(n_time * n_freq, C).astype(np.float32)

    return np.concatenate([erp_tok, tfr_tok], axis=0).astype(np.float32)  # (tokens, C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    meta = np.load(os.path.join(DATA_OUT, "tfr_meta.npz"))
    meta_d = {k: meta[k] for k in meta.files}

    bounds = np.load(os.path.join(MODELS_OUT, "param_bounds.npz"))
    param_names = [x.decode("utf-8") for x in bounds["param_names"]]
    low = bounds["prior_low"].astype(np.float32)
    high = bounds["prior_high"].astype(np.float32)

    with open(os.path.join(MODELS_OUT, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    model = _load_model()

    # Base sim config (match your training H5 settings ideally)
    fs = 250
    duration = 10.0
    n_channels = 16
    n_sources = 3
    stim_onset = 2.0
    stim_sigma = 0.05
    bandpass = (0.5, 45.0)

    shifts = [
        ("noise_low", dict(sensor_noise_std=1.0, input_noise_std=1.0)),
        ("noise_high", dict(sensor_noise_std=4.0, input_noise_std=4.0)),
        ("trials_1", dict(n_trials=1)),
        ("trials_20", dict(n_trials=20)),
        ("bandpass_none", dict(bandpass=None)),
        ("leadfield_new", dict(leadfield=_make_leadfield(n_channels, n_sources, seed=999))),
    ]

    results = {}

    for name, cfg in shifts:
        LOGGER.info("Running shift: %s", name)

        # sample params from prior
        thetas = np.stack([rng.uniform(low[i], high[i], size=args.n) for i in range(len(param_names))], axis=1).astype(np.float32)

        # simulate + tokenize
        toks = []
        for i in range(args.n):
            params = {param_names[j]: float(thetas[i, j]) for j in range(len(param_names))}
            eeg = simulate_eeg(
                params=params,
                fs=fs,
                duration=duration,
                n_channels=n_channels,
                seed=int(args.seed + 10000 * i),
                bandpass=cfg.get("bandpass", bandpass),
                stim_onset=stim_onset,
                stim_sigma=stim_sigma,
                n_sources=n_sources,
                leadfield=cfg.get("leadfield", None),
                sensor_noise_std=float(cfg.get("sensor_noise_std", 2.0)),
                n_trials=int(cfg.get("n_trials", 10)),
                input_noise_std=float(cfg.get("input_noise_std", 2.0)),
            )
            toks.append(_patchify_hybrid(eeg, meta_d))
        toks = np.stack(toks, axis=0)  # (N, tokens, C)

        toks_s = _scale_tokens(toks, scaler)
        pred = model.predict(toks_s, verbose=0).astype(np.float32)
        P = len(param_names)
        mu_z = pred[:, :P]
        logvar = np.clip(pred[:, P:], -10.0, 10.0)

        # posterior mean via sampling (small S for speed)
        from models.param_transforms import sample_theta_from_gaussian_z
        theta_samps = sample_theta_from_gaussian_z(mu_z, logvar, low, high, n_samples=100, seed=args.seed)
        theta_mean = theta_samps.mean(axis=0)

        abs_err = np.abs(theta_mean - thetas)
        rel_err = abs_err / (np.abs(thetas) + 1e-8) * 100.0
        results[name] = dict(rel_mean=rel_err.mean(axis=0), rel_med=np.median(rel_err, axis=0))

    np.savez(os.path.join(PLOTS_DIR, "generalization_sweeps.npz"),
             param_names=np.array(param_names, dtype="S"),
             **{k: v["rel_mean"] for k, v in results.items()})

    LOGGER.info("Saved: %s", os.path.join(PLOTS_DIR, "generalization_sweeps.npz"))


if __name__ == "__main__":
    main()
