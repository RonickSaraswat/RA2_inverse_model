import os
import sys
import numpy as np
import h5py
from tqdm import tqdm


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from simulate.simulator import simulate_eeg


save_path = os.path.join(os.path.dirname(__file__), "synthetic_eeg_dataset.h5")
num_samples = 1000  # (can be changed according to requirement)

eeg_data = []
params_data = []

print(" Generating synthetic EEG dataset...")

for _ in tqdm(range(num_samples), desc="Generating EEG data"):
    params = {'a': 0.1, 'b': 0.2, 'd': 2, 'couple': 1}
    eeg = simulate_eeg(params)
    eeg_data.append(eeg)
    params_data.append(list(params.values()))

# Save to HDF5
with h5py.File(save_path, "w") as f:
    f.create_dataset("EEG", data=np.array(eeg_data))
    f.create_dataset("params", data=np.array(params_data))

print(f" Synthetic EEG dataset saved to: {save_path}")
