# data/prepare_training_data.py
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Tuple

import h5py
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from features.feature_extraction import extract_features  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("data.prepare_training_data")

DATA_FILE = os.path.join(BASE_DIR, "data", "synthetic_eeg_dataset.h5")
OUT_DIR = os.path.join(BASE_DIR, "data_out")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- TFR settings ----
FS = 250
FMIN = 1.0
FMAX = 45.0
N_FREQS = 30
DECIM = 4
N_CYCLES = None

# ---- Window around stimulus (seconds) ----
PRE_SEC = 1.0
POST_SEC = 1.0

# ---- Patch sizes ----
FREQ_PATCH = 2
TIME_PATCH = 5


def _compute_patch_meta(
    tfr_shape: Tuple[int, int, int],
    stim_onset: float,
) -> Tuple[int, int, int, int, int, int, int, np.ndarray, np.ndarray]:
    """
    Compute patch grid sizes and centers from the first sample's TFR shape.

    Returns
    -------
    n_t, n_f, start_idx, T_use, Freq_use, n_tokens, n_tokens_erp, time_centers, freq_centers
    """
    c0, f0, tdec_full = tfr_shape

    win_start = stim_onset - PRE_SEC
    win_end = stim_onset + POST_SEC
    start_idx = int(np.round(win_start * FS / DECIM))
    end_idx = int(np.round(win_end * FS / DECIM))
    start_idx = max(0, start_idx)
    end_idx = min(tdec_full, end_idx)

    tdec_win = end_idx - start_idx
    if tdec_win <= 0:
        raise ValueError("Invalid time window: check PRE_SEC/POST_SEC/stim_onset/DECIM.")

    n_f = f0 // FREQ_PATCH
    n_t = tdec_win // TIME_PATCH
    if n_f <= 0 or n_t <= 0:
        raise ValueError(
            f"Patch grid is empty: n_f={n_f}, n_t={n_t}. "
            f"Check FREQ_PATCH={FREQ_PATCH}, TIME_PATCH={TIME_PATCH}."
        )

    freq_use = n_f * FREQ_PATCH
    t_use = n_t * TIME_PATCH

    # Time patch centers (seconds relative to stimulus)
    t_dec = (np.arange(tdec_full) * DECIM / FS) - stim_onset
    t_win = t_dec[start_idx : start_idx + t_use]
    time_patch_centers = np.array(
        [
            float(t_win[j * TIME_PATCH : (j + 1) * TIME_PATCH].mean())
            for j in range(n_t)
        ],
        dtype=np.float32,
    )

    # Frequency patch centers
    freqs = np.linspace(FMIN, FMAX, N_FREQS).astype(np.float32)
    freqs_use = freqs[:freq_use]
    freq_patch_centers = np.array(
        [
            float(freqs_use[k * FREQ_PATCH : (k + 1) * FREQ_PATCH].mean())
            for k in range(n_f)
        ],
        dtype=np.float32,
    )

    n_tokens_erp = n_t
    n_tokens_tfr = n_t * n_f
    n_tokens = n_tokens_erp + n_tokens_tfr

    return (
        n_t,
        n_f,
        start_idx,
        t_use,
        freq_use,
        n_tokens,
        n_tokens_erp,
        time_patch_centers,
        freq_patch_centers,
    )


def main() -> None:
    LOGGER.info("Preparing training data (HYBRID tokens: ERP + TFR patches)...")
    LOGGER.info("Input: %s", DATA_FILE)
    LOGGER.info("Output dir: %s", OUT_DIR)

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Missing dataset: {DATA_FILE}. Run: python data/generate_dataset.py"
        )

    # Read minimal metadata + priors safely
    with h5py.File(DATA_FILE, "r") as f:
        eeg_ds = f["EEG"]        # (N, C, T)
        params_ds = f["params"]  # (N, P)

        n, c, t = eeg_ds.shape
        stim_onset = float(f.attrs["stim_onset"])
        param_names = [x.decode("utf-8") for x in f.attrs["param_names"]]
        priors_attrs = dict(f["priors"].attrs.items())

        # Determine shapes from first sample
        eeg0 = np.asarray(eeg_ds[0], dtype=np.float32)
        tfr0 = extract_features(
            eeg0, fs=FS, fmin=FMIN, fmax=FMAX, n_freqs=N_FREQS, n_cycles=N_CYCLES, decim=DECIM
        )

    p = int(params_ds.shape[1])

    LOGGER.info("EEG shape: (%d, %d, %d)", n, c, t)
    LOGGER.info("Params shape: (%d, %d)", n, p)
    LOGGER.info("Stim onset: %.3f sec", stim_onset)
    LOGGER.info("Param names: %s", param_names)

    prior_low = np.array([float(priors_attrs[nm][0]) for nm in param_names], dtype=np.float32)
    prior_high = np.array([float(priors_attrs[nm][1]) for nm in param_names], dtype=np.float32)

    (
        n_t,
        n_f,
        start_idx,
        t_use,
        freq_use,
        n_tokens,
        n_tokens_erp,
        time_patch_centers,
        freq_patch_centers,
    ) = _compute_patch_meta(tfr0.shape, stim_onset=stim_onset)

    LOGGER.info("TFR0 shape: %s (C, Freq, Tdec_full)", tfr0.shape)
    LOGGER.info("Derived: n_time_patches=%d n_freq_patches=%d tokens=%d", n_t, n_f, n_tokens)

    # Token types: 0=ERP, 1=TFR
    token_types = np.zeros((n_tokens,), dtype=np.int32)
    token_types[n_tokens_erp:] = 1

    features_path = os.path.join(OUT_DIR, "features.npy")
    params_path = os.path.join(OUT_DIR, "params.npy")

    # Stream outputs to disk (avoids holding 1.6GB EEG in RAM)
    X_mm = open_memmap(
        features_path,
        mode="w+",
        dtype=np.float32,
        shape=(n, n_tokens, c),
    )
    y_mm = open_memmap(
        params_path,
        mode="w+",
        dtype=np.float32,
        shape=(n, p),
    )

    start_time = time.time()

    def patchify_tfr(tfr: np.ndarray) -> np.ndarray:
        # tfr: (C, F, Tdec_full)
        tfr_win = tfr[:, :freq_use, start_idx : start_idx + t_use]  # (C, Freq_use, T_use)
        rs = tfr_win.reshape(c, n_f, FREQ_PATCH, n_t, TIME_PATCH)
        patch = rs.mean(axis=(2, 4))  # (C, n_f, n_t)
        tokens = patch.transpose(2, 1, 0).reshape(n_t * n_f, c).astype(np.float32)
        return tokens

    def erp_tokens_from_eeg(eeg_sample: np.ndarray) -> np.ndarray:
        eeg_dec = eeg_sample[:, ::DECIM]  # (C, Tdec_full approx)
        win = eeg_dec[:, start_idx : start_idx + t_use]  # (C, T_use)
        patch = win.reshape(c, n_t, TIME_PATCH).mean(axis=2)  # (C, n_t)
        tokens = patch.T.astype(np.float32)  # (n_t, C)
        return tokens

    LOGGER.info("Processing %d samples...", n)

    with h5py.File(DATA_FILE, "r") as f:
        eeg_ds = f["EEG"]
        params_ds = f["params"]

        for i in tqdm(range(n), desc="Processing samples", unit="sample"):
            eeg_i = np.asarray(eeg_ds[i], dtype=np.float32)
            y_mm[i] = np.asarray(params_ds[i], dtype=np.float32)

            erp_tok = erp_tokens_from_eeg(eeg_i)  # (n_t, C)

            tfr = extract_features(
                eeg_i,
                fs=FS,
                fmin=FMIN,
                fmax=FMAX,
                n_freqs=N_FREQS,
                n_cycles=N_CYCLES,
                decim=DECIM,
            )
            tfr_tok = patchify_tfr(tfr)  # (n_t*n_f, C)

            X_mm[i, :n_tokens_erp, :] = erp_tok
            X_mm[i, n_tokens_erp:, :] = tfr_tok

    X_mm.flush()
    y_mm.flush()

    elapsed = time.time() - start_time

    np.savez(
        os.path.join(OUT_DIR, "tfr_meta.npz"),
        fs=FS,
        decim=DECIM,
        fmin=FMIN,
        fmax=FMAX,
        n_freqs=N_FREQS,
        stim_onset=stim_onset,
        pre_sec=PRE_SEC,
        post_sec=POST_SEC,
        freq_patch=FREQ_PATCH,
        time_patch=TIME_PATCH,
        n_time_patches=n_t,
        n_freq_patches=n_f,
        n_tokens=n_tokens,
        n_tokens_erp=n_tokens_erp,
        n_tokens_tfr=(n_t * n_f),
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

    LOGGER.info("Saved:")
    LOGGER.info(" - %s", features_path)
    LOGGER.info(" - %s", params_path)
    LOGGER.info(" - %s", os.path.join(OUT_DIR, "tfr_meta.npz"))
    LOGGER.info(" - %s", os.path.join(OUT_DIR, "param_meta.npz"))
    LOGGER.info("Total time: %.2f minutes (%.1f seconds)", elapsed / 60.0, elapsed)
    LOGGER.info("Final X shape: (%d, %d, %d) | y shape: (%d, %d)", n, n_tokens, c, n, p)


if __name__ == "__main__":
    main()
