# data/generate_dataset.py
import os
import sys
import numpy as np
import h5py
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from simulate.simulator import simulate_eeg

save_path = os.path.join(os.path.dirname(__file__), "synthetic_eeg_dataset.h5")

# ---- Simulation settings ----
num_samples = 1000
fs = 250
duration = 10.0
n_channels = 16
n_sources = 3
bandpass = (0.5, 40.0)

# ERP settings
stim_onset = 2.0
stim_sigma = 0.05
n_trials = 10
sensor_noise_std = 2.0
input_noise_std = 2.0

# ---- Parameter priors (uniform) ----
PRIORS = {
    "A":        (2.0, 5.0),
    "B":        (12.0, 30.0),
    "a":        (60.0, 140.0),
    "b":        (30.0, 90.0),
    "C":        (90.0, 180.0),
    "p0":       (80.0, 140.0),
    "stim_amp": (10.0, 200.0),
}

param_names = list(PRIORS.keys())
P = len(param_names)

rng = np.random.default_rng(seed=42)
n_samples_time = int(fs * duration)

print("Generating synthetic Jansen–Rit ERP EEG dataset...")
print("Saving to:", save_path)

if os.path.exists(save_path):
    os.remove(save_path)

# Fixed leadfield for the entire dataset
leadfield_seed = 1234
rng_lf = np.random.default_rng(seed=leadfield_seed)
leadfield = rng_lf.normal(size=(n_channels, n_sources)).astype(np.float32)
leadfield /= (np.linalg.norm(leadfield, axis=0, keepdims=True) + 1e-9)

with h5py.File(save_path, "w") as f:
    eeg_ds = f.create_dataset(
        "EEG",
        shape=(num_samples, n_channels, n_samples_time),
        dtype=np.float32,
        compression="gzip",
        compression_opts=4,
        chunks=(1, n_channels, n_samples_time),
    )
    params_ds = f.create_dataset(
        "params",
        shape=(num_samples, P),
        dtype=np.float32,
        compression="gzip",
        compression_opts=4,
        chunks=(min(1024, num_samples), P),
    )
    seeds_ds = f.create_dataset(
        "seeds",
        shape=(num_samples,),
        dtype=np.uint32,
        compression="gzip",
        compression_opts=4,
        chunks=(min(8192, num_samples),),
    )

    f.create_dataset("leadfield", data=leadfield, dtype=np.float32)

    # metadata
    f.attrs["fs"] = fs
    f.attrs["duration"] = duration
    f.attrs["n_channels"] = n_channels
    f.attrs["n_sources"] = n_sources
    f.attrs["bandpass"] = bandpass
    f.attrs["stim_onset"] = stim_onset
    f.attrs["stim_sigma"] = stim_sigma
    f.attrs["n_trials"] = n_trials
    f.attrs["sensor_noise_std"] = sensor_noise_std
    f.attrs["input_noise_std"] = input_noise_std
    f.attrs["param_names"] = np.array(param_names, dtype="S")

    prior_grp = f.create_group("priors")
    for name, (low, high) in PRIORS.items():
        prior_grp.attrs[name] = (low, high)

    for i in tqdm(range(num_samples), desc="Simulating EEG", unit="sample"):
        params = {name: rng.uniform(low, high) for name, (low, high) in PRIORS.items()}

        seed_i = np.uint32(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        eeg = simulate_eeg(
            params=params,
            fs=fs,
            duration=duration,
            n_channels=n_channels,
            seed=int(seed_i),
            bandpass=bandpass,
            stim_onset=stim_onset,
            stim_sigma=stim_sigma,
            n_sources=n_sources,
            leadfield=leadfield,
            sensor_noise_std=sensor_noise_std,
            n_trials=n_trials,
            input_noise_std=input_noise_std,
        )

        eeg_ds[i] = eeg
        params_ds[i] = np.array([params[k] for k in param_names], dtype=np.float32)
        seeds_ds[i] = seed_i

print("Done.")
print("Saved:", save_path)
