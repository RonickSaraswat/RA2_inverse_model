# data/prepare_training_data.py
import os
import sys
import time
import numpy as np
import h5py
from tqdm import tqdm  # for progress bar

# --- Ensure project root is on sys.path ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
# ------------------------------------------

from features.feature_extraction import extract_features

DATA_FILE = os.path.join(BASE_DIR, "data", "synthetic_eeg_dataset.h5")
OUT_DIR = os.path.join(BASE_DIR, "data_out")
os.makedirs(OUT_DIR, exist_ok=True)

print("Preparing training data...")

# Load EEG and parameters
with h5py.File(DATA_FILE, "r") as f:
    print("Available keys:", list(f.keys()))
    eeg = f["EEG"][:]          # shape (N, channels, time)
    params = f["params"][:]    # shape (N, P)
print("Loaded data:", eeg.shape, params.shape)

N = eeg.shape[0]
feat_list = []

start_time = time.time()  # start timer

# Progress bar + feature extraction
print(f"\nExtracting features for {N} samples...")
for i in tqdm(range(N), desc="Processing samples", unit="sample"):
    feats = extract_features(eeg[i], fs=250, bands=[(1,4),(4,8),(8,13),(13,30),(30,45)], ar_order=5)
    feat_list.append(feats)

# Stack features and save
X = np.vstack(feat_list).astype(np.float32)
y = np.array(params, dtype=np.float32)

elapsed = time.time() - start_time  # stop timer

print("\nFeature matrix shape:", X.shape, "Params shape:", y.shape)
np.save(os.path.join(OUT_DIR, "features.npy"), X)
np.save(os.path.join(OUT_DIR, "params.npy"), y)
print(f"\n Saved .npy files to: {OUT_DIR}")
print(f" Total time taken: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
