# data/generate_dataset.py
from __future__ import annotations

import logging
import os
import sys
from typing import Dict, Tuple

import h5py
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from simulate.simulator import simulate_eeg  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("data.generate_dataset")

SAVE_PATH = os.path.join(os.path.dirname(__file__), "synthetic_eeg_dataset.h5")

# ---- Simulation settings (defaults preserved) ----
NUM_SAMPLES = 10_000
FS = 250
DURATION = 10.0
N_CHANNELS = 16
N_SOURCES = 3
BANDPASS = (0.5, 40.0)

# ERP settings
STIM_ONSET = 2.0
STIM_SIGMA = 0.05
N_TRIALS = 10
SENSOR_NOISE_STD = 2.0
INPUT_NOISE_STD = 2.0

# ---- Parameter priors (uniform) ----
PRIORS: Dict[str, Tuple[float, float]] = {
    "A": (2.0, 5.0),
    "B": (12.0, 30.0),
    "a": (60.0, 140.0),
    "b": (30.0, 90.0),
    "C": (90.0, 180.0),
    "p0": (80.0, 140.0),
    "stim_amp": (10.0, 200.0),
}


def _make_fixed_leadfield(
    n_channels: int,
    n_sources: int,
    seed: int,
) -> np.ndarray:
    """Create a fixed random leadfield (column-normalized)."""
    rng = np.random.default_rng(seed=seed)
    lf = rng.normal(size=(n_channels, n_sources)).astype(np.float32)
    lf /= (np.linalg.norm(lf, axis=0, keepdims=True) + 1e-9)
    return lf


def main() -> None:
    param_names = list(PRIORS.keys())
    p = len(param_names)

    # IMPORTANT: store these for reproducibility
    param_rng_seed = 42
    leadfield_seed = 1234

    rng = np.random.default_rng(seed=param_rng_seed)
    n_samples_time = int(FS * DURATION)

    LOGGER.info("Generating synthetic Jansen–Rit ERP EEG dataset...")
    LOGGER.info("Saving to: %s", SAVE_PATH)

    tmp_path = SAVE_PATH + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    # Fixed leadfield for the entire dataset
    leadfield = _make_fixed_leadfield(
        n_channels=N_CHANNELS,
        n_sources=N_SOURCES,
        seed=leadfield_seed,
    )

    try:
        with h5py.File(tmp_path, "w") as f:
            eeg_ds = f.create_dataset(
                "EEG",
                shape=(NUM_SAMPLES, N_CHANNELS, n_samples_time),
                dtype=np.float32,
                compression="gzip",
                compression_opts=4,
                chunks=(1, N_CHANNELS, n_samples_time),
            )
            params_ds = f.create_dataset(
                "params",
                shape=(NUM_SAMPLES, p),
                dtype=np.float32,
                compression="gzip",
                compression_opts=4,
                chunks=(min(1024, NUM_SAMPLES), p),
            )
            seeds_ds = f.create_dataset(
                "seeds",
                shape=(NUM_SAMPLES,),
                dtype=np.uint32,
                compression="gzip",
                compression_opts=4,
                chunks=(min(8192, NUM_SAMPLES),),
            )

            f.create_dataset("leadfield", data=leadfield, dtype=np.float32)

            # ---- Metadata (expanded but backwards compatible) ----
            f.attrs["forward_model"] = np.array("jansen_rit", dtype="S")
            f.attrs["fs"] = FS
            f.attrs["duration"] = DURATION
            f.attrs["n_channels"] = N_CHANNELS
            f.attrs["n_sources"] = N_SOURCES
            f.attrs["bandpass"] = BANDPASS
            f.attrs["stim_onset"] = STIM_ONSET
            f.attrs["stim_sigma"] = STIM_SIGMA
            f.attrs["n_trials"] = N_TRIALS
            f.attrs["sensor_noise_std"] = SENSOR_NOISE_STD
            f.attrs["input_noise_std"] = INPUT_NOISE_STD
            f.attrs["param_names"] = np.array(param_names, dtype="S")

            # reproducibility
            f.attrs["param_rng_seed"] = int(param_rng_seed)
            f.attrs["leadfield_seed"] = int(leadfield_seed)

            prior_grp = f.create_group("priors")
            for name, (low, high) in PRIORS.items():
                prior_grp.attrs[name] = (float(low), float(high))

            # ---- Generate ----
            for i in tqdm(range(NUM_SAMPLES), desc="Simulating EEG", unit="sample"):
                params = {name: float(rng.uniform(lo, hi)) for name, (lo, hi) in PRIORS.items()}

                # per-sample seed for sensor/input noise streams inside simulate_eeg()
                seed_i = np.uint32(
                    rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32)
                )

                eeg = simulate_eeg(
                    params=params,
                    fs=FS,
                    duration=DURATION,
                    n_channels=N_CHANNELS,
                    seed=int(seed_i),
                    bandpass=BANDPASS,
                    stim_onset=STIM_ONSET,
                    stim_sigma=STIM_SIGMA,
                    n_sources=N_SOURCES,
                    leadfield=leadfield,
                    sensor_noise_std=SENSOR_NOISE_STD,
                    n_trials=N_TRIALS,
                    input_noise_std=INPUT_NOISE_STD,
                )

                eeg_ds[i] = eeg
                params_ds[i] = np.array([params[k] for k in param_names], dtype=np.float32)
                seeds_ds[i] = seed_i

        # Atomic replace
        if os.path.exists(SAVE_PATH):
            os.remove(SAVE_PATH)
        os.replace(tmp_path, SAVE_PATH)

    except Exception:
        # Clean up tmp file on error
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    LOGGER.info("Done.")
    LOGGER.info("Saved: %s", SAVE_PATH)


if __name__ == "__main__":
    main()
