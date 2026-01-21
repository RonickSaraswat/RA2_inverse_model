# simulate/simulator.py
import numpy as np
from scipy.signal import butter, filtfilt


def _sigmoid(v, e0=2.5, v0=6.0, r=0.56):
    # Jansen-Rit sigmoid (firing rate)
    return 2.0 * e0 / (1.0 + np.exp(r * (v0 - v)))


def _jr_one_source(
    params,
    internal_fs,
    duration,
    stim_onset,
    stim_sigma,
    input_noise_std,
    seed=None,
):
    """
    Simulate a single Jansen–Rit cortical column (one source).
    Returns pyramidal potential proxy (mV-ish) at internal_fs.
    """
    rng = np.random.default_rng(seed)

    A = float(params["A"])
    B = float(params["B"])
    a = float(params["a"])
    b = float(params["b"])
    C = float(params["C"])
    p0 = float(params["p0"])
    stim_amp = float(params["stim_amp"])

    # Canonical JR connectivity proportions
    C1 = 1.0 * C
    C2 = 0.8 * C
    C3 = 0.25 * C
    C4 = 0.25 * C

    dt = 1.0 / float(internal_fs)
    n_int = int(np.round(duration * internal_fs))
    t = np.arange(n_int, dtype=np.float64) * dt

    stim = stim_amp * np.exp(-0.5 * ((t - stim_onset) / (stim_sigma + 1e-12)) ** 2)
    p = p0 + stim + rng.normal(0.0, input_noise_std, size=n_int)
    p = np.maximum(p, 0.0)  # keep nonnegative drive

    # States: y0,y1,y2 and derivatives y3,y4,y5
    y0 = 0.0
    y1 = 0.0
    y2 = 0.0
    y3 = 0.0
    y4 = 0.0
    y5 = 0.0

    out = np.zeros(n_int, dtype=np.float64)

    # Small random initial condition helps break symmetry
    y0 += 0.01 * rng.normal()
    y1 += 0.01 * rng.normal()
    y2 += 0.01 * rng.normal()

    for i in range(n_int):
        # pyramidal membrane potential proxy
        v_pyr = y1 - y2

        S_pyr = _sigmoid(v_pyr)
        S_y0 = _sigmoid(C1 * y0)

        # JR ODEs (second-order PSP kernels expanded to first-order system)
        dy0 = y3
        dy3 = A * a * S_pyr - 2.0 * a * y3 - (a ** 2) * y0

        dy1 = y4
        dy4 = A * a * (p[i] + C2 * S_y0) - 2.0 * a * y4 - (a ** 2) * y1

        dy2 = y5
        dy5 = B * b * (C4 * _sigmoid(C3 * y0)) - 2.0 * b * y5 - (b ** 2) * y2

        # Euler integration (stable at internal_fs=1000 for your ranges)
        y0 += dt * dy0
        y3 += dt * dy3
        y1 += dt * dy1
        y4 += dt * dy4
        y2 += dt * dy2
        y5 += dt * dy5

        out[i] = v_pyr

    return out


def simulate_eeg(
    params,
    fs=250,
    duration=10.0,
    n_channels=16,
    seed=None,
    bandpass=(0.5, 40.0),
    stim_onset=2.0,
    stim_sigma=0.05,
    n_sources=3,
    leadfield=None,
    sensor_noise_std=2.0,
    n_trials=10,
    input_noise_std=2.0,
    internal_fs=1000,
    baseline_correct=True,
    uV_scale=1000.0,
):
    """
    Jansen–Rit ERP EEG forward simulator.

    Key point for identifiability:
      - NO per-trial standardization (no division by std).
      - Baseline correction is allowed and recommended.
    """
    rng = np.random.default_rng(seed)

    if internal_fs % fs != 0:
        raise ValueError("internal_fs must be an integer multiple of fs (e.g., 1000 and 250).")
    ds = internal_fs // fs

    n_int = int(np.round(duration * internal_fs))
    n_out = int(np.round(duration * fs))

    if leadfield is None:
        # fixed but random mixing if not provided
        lf = rng.normal(size=(n_channels, n_sources)).astype(np.float32)
        lf /= (np.linalg.norm(lf, axis=0, keepdims=True) + 1e-9)
        leadfield = lf
    else:
        leadfield = np.asarray(leadfield, dtype=np.float32)
        assert leadfield.shape == (n_channels, n_sources)

    eeg_acc = np.zeros((n_channels, n_out), dtype=np.float64)

    # Trial averaging (ERP-like)
    for tr in range(int(n_trials)):
        # independent seeds per trial and source
        src = np.zeros((n_sources, n_int), dtype=np.float64)
        for s in range(n_sources):
            s_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            src[s] = _jr_one_source(
                params=params,
                internal_fs=internal_fs,
                duration=duration,
                stim_onset=stim_onset,
                stim_sigma=stim_sigma,
                input_noise_std=input_noise_std,
                seed=s_seed,
            )

        # downsample sources to sensor fs
        src_ds = src[:, ::ds]  # (n_sources, n_out)

        # mix to sensors and scale to uV
        eeg = (leadfield @ src_ds) * uV_scale  # (n_channels, n_out)

        # add sensor noise (uV)
        eeg += rng.normal(0.0, sensor_noise_std, size=eeg.shape)

        eeg_acc += eeg

    eeg = (eeg_acc / float(n_trials)).astype(np.float64)

    # Baseline correction (pre-stim mean)
    if baseline_correct:
        pre = int(max(1, np.round(stim_onset * fs)))
        base = eeg[:, :pre].mean(axis=1, keepdims=True)
        eeg = eeg - base

    # Bandpass (optional)
    if bandpass is not None:
        lo, hi = bandpass
        nyq = 0.5 * fs
        lo_n = max(1e-6, lo / nyq)
        hi_n = min(0.999, hi / nyq)
        if lo_n < hi_n:
            b, a = butter(4, [lo_n, hi_n], btype="band")
            eeg = filtfilt(b, a, eeg, axis=1)

    return eeg.astype(np.float32)
