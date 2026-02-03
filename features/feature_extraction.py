# features/feature_extraction.py
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import mne


def extract_features(
    eeg: np.ndarray,
    fs: int = 250,
    fmin: float = 1.0,
    fmax: float = 45.0,
    n_freqs: int = 30,
    n_cycles: Optional[Union[float, np.ndarray]] = None,
    decim: int = 4,
    log_eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute Morlet time-frequency representation (log-power) for one EEG sample.

    Parameters
    ----------
    eeg:
        Array of shape (C, T) in arbitrary units (typically microvolts).
    fs:
        Sampling rate in Hz.
    fmin, fmax:
        Frequency range for Morlet wavelets (Hz).
    n_freqs:
        Number of frequencies (linearly spaced between fmin and fmax).
    n_cycles:
        If None, uses a safe heuristic: max(3 cycles, f/2).
        Can also be a float or array-like of length n_freqs.
    decim:
        Decimation factor applied inside MNE's Morlet routine. Must be >= 1.
    log_eps:
        Small epsilon for log(power + eps) stability.

    Returns
    -------
    tfr:
        Array of shape (C, n_freqs, T_decimated), log(power).
    """
    eeg = np.asarray(eeg, dtype=np.float32)

    # ---- Basic input validation ----
    if eeg.ndim != 2:
        raise ValueError(f"Expected eeg with shape (C, T), got {eeg.shape}")

    n_ch, n_t = eeg.shape
    if n_ch <= 0 or n_t <= 0:
        raise ValueError(f"Invalid eeg shape: {eeg.shape}")

    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    if n_freqs <= 0:
        raise ValueError(f"n_freqs must be positive, got {n_freqs}")
    if decim < 1:
        raise ValueError(f"decim must be >= 1, got {decim}")

    if not (0.0 < fmin < fmax):
        raise ValueError(f"Require 0 < fmin < fmax, got fmin={fmin}, fmax={fmax}")

    nyq = 0.5 * float(fs)
    if fmax >= nyq:
        raise ValueError(
            f"fmax must be < Nyquist (fs/2). Got fmax={fmax}, fs={fs}, Nyquist={nyq}"
        )

    if log_eps <= 0:
        raise ValueError(f"log_eps must be positive, got {log_eps}")

    # ---- Frequency grid ----
    freqs = np.linspace(fmin, fmax, n_freqs).astype(np.float32)

    # ---- n_cycles handling ----
    if n_cycles is None:
        # Safer low-freq wavelets: avoid huge wavelets at 1–2 Hz
        n_cycles_arr = np.maximum(3.0, freqs / 2.0).astype(np.float32)
    else:
        if np.isscalar(n_cycles):
            n_cycles_arr = float(n_cycles)
        else:
            n_cycles_arr = np.asarray(n_cycles, dtype=np.float32)
            if n_cycles_arr.shape[0] != n_freqs:
                raise ValueError(
                    f"If n_cycles is an array, it must have length n_freqs={n_freqs}; "
                    f"got shape {n_cycles_arr.shape}."
                )

    # MNE expects shape (epochs, channels, times)
    data = eeg[np.newaxis, :, :]  # (1, C, T)

    power = mne.time_frequency.tfr_array_morlet(
        data,
        sfreq=float(fs),
        freqs=freqs,
        n_cycles=n_cycles_arr,
        output="power",
        decim=int(decim),
        n_jobs=1,          # deterministic + stable for pipeline
        verbose=False,
    )  # (1, C, F, Tdec)

    tfr = power[0].astype(np.float32)  # (C, F, Tdec)
    tfr = np.log(tfr + float(log_eps)).astype(np.float32)
    return tfr
