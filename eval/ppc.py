from __future__ import annotations

import argparse
import os
import sys
import pickle
import logging
from typing import Dict, Any

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

DATA_OUT = os.path.join(BASE_DIR, "data_out")
MODELS_OUT = os.path.join(BASE_DIR, "models_out")
DATA_FILE_DEFAULT = os.path.join(BASE_DIR, "data", "synthetic_eeg_dataset.h5")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

from data.splits import ensure_splits  # noqa: E402
from simulate.simulator import simulate_eeg  # noqa: E402
from models.param_transforms import sample_theta_from_gaussian_z  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
LOGGER = logging.getLogger("ppc")


def _import_tf_or_die():
    try:
        import tensorflow as tf  # noqa: F401
    except ModuleNotFoundError as e:
        raise SystemExit("TensorFlow missing in this env.") from e


def _load_model(models_out: str):
    _import_tf_or_die()
    from tensorflow.keras.models import load_model

    try:
        import models.bi_lstm_model  # noqa: F401
    except Exception:
        pass

    best = os.path.join(models_out, "jr_paramtoken_inverse_model_best.keras")
    final = os.path.join(models_out, "jr_paramtoken_inverse_model.keras")
    path = best if os.path.exists(best) else final
    return load_model(path, compile=False, custom_objects=None, safe_mode=False)


def _scale_X(tokens: np.ndarray, scaler) -> np.ndarray:
    # tokens: (1, T, C)
    b, t, c = tokens.shape
    flat = tokens.reshape(-1, c)
    flat_s = scaler.transform(flat).astype(np.float32)
    return flat_s.reshape(b, t, c)


def _gfp(eeg: np.ndarray) -> np.ndarray:
    # eeg: (C, T)
    return np.sqrt((eeg.astype(np.float64) ** 2).mean(axis=0)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-file", type=str, default=DATA_FILE_DEFAULT)
    ap.add_argument("--n-examples", type=int, default=10)
    ap.add_argument("--n-draws", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import h5py

    rng = np.random.default_rng(args.seed)

    # Load artifacts
    X = np.load(os.path.join(DATA_OUT, "features.npy"), mmap_mode="r")
    meta = np.load(os.path.join(DATA_OUT, "tfr_meta.npz"))
    bounds = np.load(os.path.join(MODELS_OUT, "param_bounds.npz"))
    param_names = [x.decode("utf-8") for x in bounds["param_names"]]
    low = bounds["prior_low"].astype(np.float32)
    high = bounds["prior_high"].astype(np.float32)

    with open(os.path.join(MODELS_OUT, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    model = _load_model(MODELS_OUT)

    splits = ensure_splits(DATA_OUT, seed=42, train_frac=0.70, val_frac=0.15, test_frac=0.15, overwrite=False)
    test_idx = np.asarray(splits["test_idx"], dtype=np.int64)

    chosen = rng.choice(test_idx, size=min(args.n_examples, len(test_idx)), replace=False)

    # Load simulator args from H5 + leadfield
    with h5py.File(args.data_file, "r") as f:
        eeg_ds = f["EEG"]
        leadfield = f["leadfield"][:]
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

        # Run PPC
        cover = []
        rmse = []
        for k, idx in enumerate(chosen):
            # Observed EEG + GFP
            eeg_obs = np.asarray(eeg_ds[int(idx)], dtype=np.float32)
            gfp_obs = _gfp(eeg_obs)

            # Predict posterior for this example
            tokens = np.asarray(X[int(idx)], dtype=np.float32)[None, :, :]  # (1, tokens, C)
            tokens_s = _scale_X(tokens, scaler=scaler)
            pred = model.predict(tokens_s, verbose=0).astype(np.float32)
            P = len(param_names)
            mu_z = pred[:, :P]
            logvar_z = np.clip(pred[:, P:], -10.0, 10.0)

            theta_samps = sample_theta_from_gaussian_z(
                mu_z, logvar_z, low, high, n_samples=int(args.n_draws), seed=int(args.seed + k)
            )  # (S, 1, P)
            theta_samps = theta_samps[:, 0, :]  # (S, P)

            # Simulate posterior predictive EEGs and GFPs
            gfp_sims = []
            for s in range(theta_samps.shape[0]):
                params = {param_names[j]: float(theta_samps[s, j]) for j in range(P)}
                eeg_sim = simulate_eeg(params=params, seed=int(args.seed + 1000 * k + s), **sim_args)
                gfp_sims.append(_gfp(eeg_sim))
            gfp_sims = np.stack(gfp_sims, axis=0)  # (S, T)

            gfp_mean = gfp_sims.mean(axis=0)
            lo = np.quantile(gfp_sims, 0.05, axis=0)
            hi = np.quantile(gfp_sims, 0.95, axis=0)

            # Coverage in time: fraction of timepoints where observed lies in 90% band
            cover_k = float(np.mean((gfp_obs >= lo) & (gfp_obs <= hi)))
            cover.append(cover_k)

            # RMSE between observed and PPC mean curve
            rmse_k = float(np.sqrt(np.mean((gfp_obs - gfp_mean) ** 2)))
            rmse.append(rmse_k)

        cover = np.asarray(cover, dtype=np.float32)
        rmse = np.asarray(rmse, dtype=np.float32)

    np.savez(
        os.path.join(PLOTS_DIR, "ppc_summary.npz"),
        chosen_idx=chosen.astype(np.int64),
        coverage_time_fraction=cover,
        rmse_gfp=rmse,
        n_draws=int(args.n_draws),
    )

    LOGGER.info("PPC done. Mean time-coverage=%.3f, mean GFP-RMSE=%.3f",
                float(cover.mean()), float(rmse.mean()))
    LOGGER.info("Saved: %s", os.path.join(PLOTS_DIR, "ppc_summary.npz"))


if __name__ == "__main__":
    main()
