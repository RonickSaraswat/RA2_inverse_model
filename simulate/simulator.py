# simulate/simulator.py
import numpy as np
from scipy.signal import lfilter, butter, filtfilt

def _toy_oscillator(freq, t, phase=0, amp=1.0):
    return amp * np.sin(2 * np.pi * freq * t + phase)

def simulate_eeg(params, fs=250, duration=2.0, n_channels=8, seed=None,
                 bandpass=(1.0, 40.0)):
    
    rng = np.random.default_rng(seed)
    n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs

    a = float(params.get('a', 0.02))
    b = float(params.get('b', 0.2))
    d = float(params.get('d', 2.0))
    coupling = float(params.get('couple', 1.0))

    eeg = np.zeros((n_channels, n_samples), dtype=np.float32)
    base_freqs = np.linspace(6, 30, n_channels)

    # Bandpass Filter
    lowcut, highcut = bandpass
    nyq = 0.5 * fs
    if lowcut <= 0 or highcut >= nyq:
    
        b_filt, a_filt = None, None
    else:
        b_filt, a_filt = butter(4, [lowcut/nyq, highcut/nyq], btype='band')

    for ch in range(n_channels):
        f = base_freqs[ch] * (1.0 + 0.5 * (a - 0.02))
        amp = 10.0 * (1.0 + 2.0 * (b - 0.2))
        phase = rng.uniform(0, 2*np.pi)

        sig = _toy_oscillator(f, t, phase=phase, amp=amp)

        noise = rng.normal(0, 1.0, size=n_samples)
        ar_coeff = 0.9 * np.tanh(0.1*(d-1.0))
        colored = lfilter([1.0], [1.0, -ar_coeff], noise)

        raw_signal = sig + coupling * colored * 0.5

        # zero-phase bandpass
        if b_filt is not None:
            try:
                filtered_signal = filtfilt(b_filt, a_filt, raw_signal)
            except Exception:
                # fallback for filter
                filtered_signal = lfilter(b_filt, a_filt, raw_signal)
        else:
            filtered_signal = raw_signal

        eeg[ch] = filtered_signal

    # normalize per channel (zero mean, unit var) then scale (µV-ish)
    eeg = (eeg - eeg.mean(axis=1, keepdims=True))
    eeg /= (eeg.std(axis=1, keepdims=True) + 1e-9)
    eeg *= 50.0
    return eeg
