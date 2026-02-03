#simulate/simulator.py
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly

LOGGER = logging.getLogger(__name__)

JR_REQUIRED_KEYS = ("A", "B", "a", "b", "C", "p0", "stim_amp")


def _validate_jr_params(params: Dict[str, float]) -> None:
    missing = [k for k in JR_REQUIRED_KEYS if k not in params]
    if missing:
        raise KeyError(
            f"Jansen–Rit params missing keys: {missing}. "
            f"Required keys: {list(JR_REQUIRED_KEYS)}"
        )


def _sigmoid_jr(v: np.ndarray, e0: float = 2.5, v0: float = 6.0, r: float = 0.56) -> np.ndarray:
    """
    Jansen–Rit sigmoid firing-rate function with overflow-safe clipping.
    """
    v = np.asarray(v, dtype=np.float64)
    x = r * (v0 - v)
    x = np.clip(x, -60.0, 60.0)
    return (2.0 * e0) / (1.0 + np.exp(x))


def _match_len_1d(x: np.ndarray, n: int) -> np.ndarray:
    """
    Ensure a 1D array has length n via truncation or edge padding.
    This avoids rare rounding mismatches between duration*fs and integer lengths.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.shape[0] == n:
        return x
    if x.shape[0] > n:
        return x[:n]
    pad = n - x.shape[0]
    return np.pad(x, (0, pad), mode="edge")


def _jr_one_source(
    params: Dict[str, float],
    internal_fs: int,
    duration: float,
    stim_onset: float,
    stim_sigma: float,
    input_noise_std: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate a single Jansen–Rit cortical column (one source).

    Returns pyramidal proxy v_pyr(t) = y1 - y2 at internal_fs.
    """
    _validate_jr_params(params)
    rng = np.random.default_rng(seed)

    A = float(params["A"])
    B = float(params["B"])
    a = float(params["a"])
    b = float(params["b"])
    C = float(params["C"])
    p0 = float(params["p0"])
    stim_amp = float(params["stim_amp"])

    if internal_fs <= 0:
        raise ValueError(f"internal_fs must be positive, got {internal_fs}")
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")
    if stim_sigma <= 0:
        raise ValueError(f"stim_sigma must be positive, got {stim_sigma}")
    if input_noise_std < 0:
        raise ValueError(f"input_noise_std must be >= 0, got {input_noise_std}")

    # Canonical JR connectivity proportions
    C1 = 1.0 * C
    C2 = 0.8 * C
    C3 = 0.25 * C
    C4 = 0.25 * C

    dt = 1.0 / float(internal_fs)
    n_int = int(np.round(duration * internal_fs))
    if n_int <= 2:
        raise ValueError(f"Duration too small: duration={duration} internal_fs={internal_fs}")

    t = np.arange(n_int, dtype=np.float64) * dt

    stim = stim_amp * np.exp(-0.5 * ((t - stim_onset) / (stim_sigma + 1e-12)) ** 2)
    p = p0 + stim + rng.normal(0.0, input_noise_std, size=n_int)
    p = np.maximum(p, 0.0)

    # States: y0,y1,y2 and derivatives y3,y4,y5
    y0 = 0.01 * rng.normal()
    y1 = 0.01 * rng.normal()
    y2 = 0.01 * rng.normal()
    y3 = 0.0
    y4 = 0.0
    y5 = 0.0

    out = np.zeros(n_int, dtype=np.float64)

    # Forward Euler (stable for your dt with internal_fs=1000 and your parameter ranges)
    for i in range(n_int):
        v_pyr = y1 - y2
        S_pyr = _sigmoid_jr(v_pyr)
        S_y0 = _sigmoid_jr(C1 * y0)

        dy0 = y3
        dy3 = A * a * S_pyr - 2.0 * a * y3 - (a**2) * y0

        dy1 = y4
        dy4 = A * a * (p[i] + C2 * S_y0) - 2.0 * a * y4 - (a**2) * y1

        dy2 = y5
        dy5 = B * b * (C4 * _sigmoid_jr(C3 * y0)) - 2.0 * b * y5 - (b**2) * y2

        y0 += dt * dy0
        y3 += dt * dy3
        y1 += dt * dy1
        y4 += dt * dy4
        y2 += dt * dy2
        y5 += dt * dy5

        out[i] = v_pyr

    return out


def _compute_baseline_indices(
    fs: int,
    n_out: int,
    stim_onset: float,
    baseline_window: Optional[Tuple[float, float]],
) -> slice:
    """
    Baseline indices for baseline correction.

    baseline_window:
      - None: use [0, stim_onset) (legacy behavior)
      - (tmin, tmax): relative to stim_onset, i.e. [stim_onset+tmin, stim_onset+tmax)
        Example: (-0.2, 0.0) means last 200ms pre-stim.
    """
    if baseline_window is None:
        end = int(max(1, np.round(stim_onset * fs)))
        end = min(end, n_out)
        return slice(0, end)

    tmin_rel, tmax_rel = float(baseline_window[0]), float(baseline_window[1])
    if tmax_rel <= tmin_rel:
        raise ValueError(f"Invalid baseline_window={baseline_window}. Need tmax > tmin.")

    start_t = stim_onset + tmin_rel
    end_t = stim_onset + tmax_rel
    start = int(np.round(start_t * fs))
    end = int(np.round(end_t * fs))
    start = max(0, min(start, n_out))
    end = max(0, min(end, n_out))

    if end <= start:
        raise ValueError(
            f"Baseline window outside range: baseline_window={baseline_window}, stim_onset={stim_onset}, "
            f"fs={fs}, n_out={n_out} -> start={start}, end={end}"
        )
    return slice(start, end)


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
    input_noise_std: float = 2.0,
    internal_fs: int = 1000,
    baseline_correct: bool = True,
    baseline_window: Optional[Tuple[float, float]] = None,
    warmup_sec: float = 0.0,
    downsample_method: str = "slice",
    uV_scale: float = 1000.0,
    return_trials: bool = False,
    return_sources: bool = False,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Jansen–Rit ERP EEG forward simulator.

    - No per-trial standardization (important for identifiability).
    - Optional warm-up to avoid init transients contaminating baseline.
    - Optional baseline window relative to stimulus (ERP-standard).
    - Optional polyphase downsampling to reduce aliasing.

    Returns EEG (channels, time) unless return_trials/return_sources True.
    """
    _validate_jr_params(params)

    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")
    if n_channels <= 0 or n_sources <= 0:
        raise ValueError(f"n_channels and n_sources must be positive, got {n_channels}, {n_sources}")
    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive, got {n_trials}")
    if warmup_sec < 0:
        raise ValueError(f"warmup_sec must be >= 0, got {warmup_sec}")
    if sensor_noise_std < 0:
        raise ValueError(f"sensor_noise_std must be >= 0, got {sensor_noise_std}")

    rng = np.random.default_rng(seed)

    if internal_fs % fs != 0:
        raise ValueError("internal_fs must be an integer multiple of fs (e.g., 1000 and 250).")
    ds = internal_fs // fs

    n_out = int(np.round(duration * fs))
    if n_out <= 2:
        raise ValueError(f"duration too small: duration={duration}, fs={fs} -> n_out={n_out}")

    n_warm_out = int(np.round(warmup_sec * fs))
    total_out = n_warm_out + n_out
    total_duration = total_out / float(fs)

    stim_onset_total = stim_onset + (n_warm_out / float(fs))

    # Exact internal length to align with output grid
    n_int_total = total_out * ds
    n_warm_int = n_warm_out * ds

    # Leadfield
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
            src_s = _jr_one_source(
                params=params,
                internal_fs=internal_fs,
                duration=total_duration,
                stim_onset=stim_onset_total,
                stim_sigma=stim_sigma,
                input_noise_std=input_noise_std,
                seed=s_seed,
            )
            src[s] = _match_len_1d(src_s, n_int_total)

        # Discard warm-up if requested
        if n_warm_int > 0:
            src_use = src[:, n_warm_int : n_warm_int + n_out * ds]
        else:
            src_use = src[:, : n_out * ds]

        # Downsample sources to sensor fs
        if downsample_method == "slice":
            src_ds = src_use[:, ::ds]
        elif downsample_method == "poly":
            src_ds = resample_poly(src_use, up=1, down=ds, axis=1)
        else:
            raise ValueError("downsample_method must be 'slice' or 'poly'")

        # Ensure exact length n_out
        if src_ds.shape[1] > n_out:
            src_ds = src_ds[:, :n_out]
        elif src_ds.shape[1] < n_out:
            src_ds = np.pad(src_ds, ((0, 0), (0, n_out - src_ds.shape[1])), mode="edge")

        eeg = (leadfield @ src_ds) * float(uV_scale)
        eeg += rng.normal(0.0, sensor_noise_std, size=eeg.shape)

        eeg_acc += eeg

        if return_trials and eeg_trials is not None:
            eeg_trials.append(eeg.astype(np.float32))
        if return_sources and sources_trials is not None:
            sources_trials.append(src_ds.astype(np.float32))

    eeg_avg = (eeg_acc / float(n_trials)).astype(np.float64)

    # Baseline correction (ERP standard)
    if baseline_correct:
        bl_slice = _compute_baseline_indices(
            fs=fs,
            n_out=n_out,
            stim_onset=stim_onset,
            baseline_window=baseline_window,
        )
        eeg_avg -= eeg_avg[:, bl_slice].mean(axis=1, keepdims=True)

    # Bandpass (optional)
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
