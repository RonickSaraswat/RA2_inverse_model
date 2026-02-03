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
        Decimation factor applied inside MNE's Morlet routine.

    Returns
    -------
    tfr:
        Array of shape (C, n_freqs, T_decimated), log(power).
    """
    eeg = np.asarray(eeg, dtype=np.float32)
    if eeg.ndim != 2:
        raise ValueError(f"Expected eeg with shape (C, T), got {eeg.shape}")

    n_ch, n_t = eeg.shape
    if n_ch <= 0 or n_t <= 0:
        raise ValueError(f"Invalid eeg shape: {eeg.shape}")

    freqs = np.linspace(fmin, fmax, n_freqs).astype(np.float32)

    if n_cycles is None:
        # Safer low-freq wavelets: avoid huge wavelets at 1–2 Hz
        n_cycles = np.maximum(3.0, freqs / 2.0)

    data = eeg[np.newaxis, :, :]  # (1, C, T)

    power = mne.time_frequency.tfr_array_morlet(
        data,
        sfreq=float(fs),
        freqs=freqs,
        n_cycles=n_cycles,
        output="power",
        decim=int(decim),
        n_jobs=1,
        verbose=False,
    )  # (1, C, F, Tdec)

    tfr = power[0].astype(np.float32)
    tfr = np.log(tfr + 1e-10).astype(np.float32)
    return tfr
