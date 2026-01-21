# features/feature_extraction.py
import numpy as np
import mne


def extract_features(
    eeg,
    fs=250,
    fmin=1.0,
    fmax=45.0,
    n_freqs=30,
    n_cycles=None,
    decim=4,
):
    """
    Morlet TFR (log-power) for one EEG sample.

    eeg: (C, T)
    returns: (C, n_freqs, Tdec)
    """
    eeg = np.asarray(eeg, dtype=np.float32)
    C, T = eeg.shape

    freqs = np.linspace(fmin, fmax, n_freqs).astype(np.float32)

    # Safer low-freq wavelets: avoid huge wavelets at 1–2 Hz
    if n_cycles is None:
        # 3 cycles at low freq, gradually increasing
        n_cycles = np.maximum(3.0, freqs / 2.0)

    data = eeg[np.newaxis, :, :]  # (1, C, T)

    power = mne.time_frequency.tfr_array_morlet(
        data,
        sfreq=fs,
        freqs=freqs,
        n_cycles=n_cycles,
        output="power",
        decim=decim,
        n_jobs=1,
        verbose=False,
    )  # (1, C, F, Tdec)

    tfr = power[0].astype(np.float32)  # (C, F, Tdec)
    tfr = np.log(tfr + 1e-10).astype(np.float32)
    return tfr
