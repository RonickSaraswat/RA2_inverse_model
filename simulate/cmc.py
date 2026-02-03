from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly

LOGGER = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray, gain: float = 1.0, bias: float = 0.0) -> np.ndarray:
    """Smooth firing-rate nonlinearity (logistic) with safe clipping."""
    x = np.asarray(x, dtype=np.float64)
    z = gain * (x - bias)
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _match_len_1d(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.shape[0] == n:
        return x
    if x.shape[0] > n:
        return x[:n]
    return np.pad(x, (0, n - x.shape[0]), mode="edge")


def _compute_baseline_indices(
    fs: int,
    n_out: int,
    stim_onset: float,
    baseline_window: Optional[Tuple[float, float]],
) -> slice:
    if baseline_window is None:
        end = int(max(1, np.round(stim_onset * fs)))
        end = min(end, n_out)
        return slice(0, end)

    tmin_rel, tmax_rel = float(baseline_window[0]), float(baseline_window[1])
    if tmax_rel <= tmin_rel:
        raise ValueError(f"Invalid baseline_window={baseline_window}. Need tmax > tmin.")

    start = int(np.round((stim_onset + tmin_rel) * fs))
    end = int(np.round((stim_onset + tmax_rel) * fs))
    start = max(0, min(start, n_out))
    end = max(0, min(end, n_out))
    if end <= start:
        raise ValueError("Baseline window maps outside data range.")
    return slice(start, end)


def _cmc_one_source(
    params: Dict[str, float],
    internal_fs: int,
    duration: float,
    stim_onset: float,
    stim_sigma: float,
    input_noise_std: float,
    seed: Optional[int],
) -> np.ndarray:
    """
    Minimal 6-population CMC-like neural mass (Wilson–Cowan style).

    Returns an LFP-like readout at internal_fs.
    """
    rng = np.random.default_rng(seed)

    dt = 1.0 / float(internal_fs)
    n_int = int(np.round(duration * internal_fs))
    if n_int <= 2:
        raise ValueError("duration too small for internal_fs.")
    t = np.arange(n_int, dtype=np.float64) * dt

    # External drive
    p0 = float(params.get("p0", 0.5))
    stim_amp = float(params.get("stim_amp", 1.0))
    stim = stim_amp * np.exp(-0.5 * ((t - stim_onset) / (stim_sigma + 1e-12)) ** 2)
    u = p0 + stim + rng.normal(0.0, input_noise_std, size=n_int)

    tau_e = float(params.get("tau_e", 0.02))
    tau_i = float(params.get("tau_i", 0.01))
    if tau_e <= 0 or tau_i <= 0:
        raise ValueError("tau_e and tau_i must be positive.")

    g = float(params.get("g", 1.0))

    # Connectivity
    W = np.zeros((6, 6), dtype=np.float64)

    w_ee = float(params.get("w_ee", 1.2))
    w_ei = float(params.get("w_ei", 1.0))
    w_ie = float(params.get("w_ie", -1.4))
    w_ii = float(params.get("w_ii", -0.6))

    for (e, i) in [(0, 1), (2, 3), (4, 5)]:
        W[e, e] += w_ee
        W[i, e] += w_ei
        W[e, i] += w_ie
        W[i, i] += w_ii

    w_ff = float(params.get("w_ff", 0.8))
    w_fb = float(params.get("w_fb", 0.6))
    w_sd = float(params.get("w_sd", 0.5))

    W[0, 2] += w_ff
    W[1, 2] += 0.5 * w_ff

    W[0, 4] += w_fb
    W[1, 4] += 0.3 * w_fb

    W[4, 0] += w_sd
    W[5, 0] += 0.3 * w_sd

    inp_to = np.array([0.0, 0.0, 1.0, 0.2, 0.1, 0.0], dtype=np.float64)

    v = 0.01 * rng.normal(size=(6,), scale=1.0).astype(np.float64)

    out = np.zeros(n_int, dtype=np.float64)
    for k in range(n_int):
        r = _sigmoid(v, gain=2.0, bias=0.0)
        drive = inp_to * u[k]

        dv = np.zeros_like(v)
        dv[0] = (-v[0] + g * (W[0] @ r) + drive[0]) / tau_e
        dv[2] = (-v[2] + g * (W[2] @ r) + drive[2]) / tau_e
        dv[4] = (-v[4] + g * (W[4] @ r) + drive[4]) / tau_e

        dv[1] = (-v[1] + g * (W[1] @ r) + drive[1]) / tau_i
        dv[3] = (-v[3] + g * (W[3] @ r) + drive[3]) / tau_i
        dv[5] = (-v[5] + g * (W[5] @ r) + drive[5]) / tau_i

        v += dt * dv

        # LFP-like readout
        out[k] = (v[0] + 0.8 * v[4] + 0.5 * v[2]) - (0.6 * v[1] + 0.4 * v[5] + 0.3 * v[3])

    return out


def simulate_eeg(
    params: Dict[str, float],
    fs: int = 250,
    duration: float = 10.0,
    n_channels: int = 16,
    seed: Optional[int] = None,
    bandpass: Optional[Tuple[float, float]] = (0.5, 40.0),
    stim_onset: float = 2.0,
    stim_sigma: float = 0.05,
    n_sources: int = 3,
    leadfield: Optional[np.ndarray] = None,
    sensor_noise_std: float = 2.0,
    n_trials: int = 10,
    input_noise_std: float = 0.2,
    internal_fs: int = 1000,
    baseline_correct: bool = True,
    baseline_window: Optional[Tuple[float, float]] = None,
    warmup_sec: float = 0.0,
    downsample_method: str = "slice",
    uV_scale: float = 100.0,
    return_trials: bool = False,
    return_sources: bool = False,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    CMC-like EEG simulator (drop-in alternative).
    """
    rng = np.random.default_rng(seed)

    if internal_fs % fs != 0:
        raise ValueError("internal_fs must be an integer multiple of fs.")
    ds = internal_fs // fs

    n_out = int(np.round(duration * fs))
    n_warm_out = int(np.round(warmup_sec * fs))
    total_out = n_warm_out + n_out
    total_duration = total_out / float(fs)
    stim_onset_total = stim_onset + (n_warm_out / float(fs))

    n_int_total = total_out * ds
    n_warm_int = n_warm_out * ds

    if leadfield is None:
        lf = rng.normal(size=(n_channels, n_sources)).astype(np.float32)
        lf /= (np.linalg.norm(lf, axis=0, keepdims=True) + 1e-9)
        leadfield = lf
    else:
        leadfield = np.asarray(leadfield, dtype=np.float32)
        if leadfield.shape != (n_channels, n_sources):
            raise ValueError(f"leadfield must have shape {(n_channels, n_sources)}, got {leadfield.shape}")

    eeg_trials = [] if return_trials else None
    sources_trials = [] if return_sources else None
    eeg_acc = np.zeros((n_channels, n_out), dtype=np.float64)

    for _tr in range(int(n_trials)):
        src = np.zeros((n_sources, n_int_total), dtype=np.float64)
        for s in range(n_sources):
            s_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            src_s = _cmc_one_source(
                params=params,
                internal_fs=internal_fs,
                duration=total_duration,
                stim_onset=stim_onset_total,
                stim_sigma=stim_sigma,
                input_noise_std=input_noise_std,
                seed=s_seed,
            )
            src[s] = _match_len_1d(src_s, n_int_total)

        if n_warm_int > 0:
            src_use = src[:, n_warm_int : n_warm_int + n_out * ds]
        else:
            src_use = src[:, : n_out * ds]

        if downsample_method == "slice":
            src_ds = src_use[:, ::ds]
        elif downsample_method == "poly":
            src_ds = resample_poly(src_use, up=1, down=ds, axis=1)
        else:
            raise ValueError("downsample_method must be 'slice' or 'poly'")

        src_ds = src_ds[:, :n_out]

        eeg = (leadfield @ src_ds) * float(uV_scale)
        eeg += rng.normal(0.0, sensor_noise_std, size=eeg.shape)
        eeg_acc += eeg

        if return_trials and eeg_trials is not None:
            eeg_trials.append(eeg.astype(np.float32))
        if return_sources and sources_trials is not None:
            sources_trials.append(src_ds.astype(np.float32))

    eeg_avg = (eeg_acc / float(n_trials)).astype(np.float64)

    if baseline_correct:
        bl = _compute_baseline_indices(fs=fs, n_out=n_out, stim_onset=stim_onset, baseline_window=baseline_window)
        eeg_avg -= eeg_avg[:, bl].mean(axis=1, keepdims=True)

    if bandpass is not None:
        lo, hi = float(bandpass[0]), float(bandpass[1])
        nyq = 0.5 * fs
        lo_n = max(1e-6, lo / nyq)
        hi_n = min(0.999, hi / nyq)
        if lo_n < hi_n:
            b, a = butter(4, [lo_n, hi_n], btype="band")
            eeg_avg = filtfilt(b, a, eeg_avg, axis=1)

    eeg_avg = eeg_avg.astype(np.float32)

    if not return_trials and not return_sources:
        return eeg_avg

    out: Dict[str, np.ndarray] = {"eeg": eeg_avg}
    if return_trials and eeg_trials is not None:
        out["eeg_trials"] = np.stack(eeg_trials, axis=0).astype(np.float32)
    if return_sources and sources_trials is not None:
        out["sources"] = np.stack(sources_trials, axis=0).astype(np.float32)
    return out
