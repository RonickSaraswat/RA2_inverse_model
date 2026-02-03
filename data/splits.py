# data/splits.py
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy types (for debug payload)."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)


def _update_hasher_with_array(h: "hashlib._Hash", arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr)
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.tobytes(order="C"))


def compute_data_fingerprint(data_out_dir: str) -> Tuple[str, Dict[str, Any]]:
    """
    Compute a stable, content-based fingerprint of prepared training data.

    Design goals:
      - Catch mismatches when features/params/meta actually change.
      - Avoid false mismatches from mtimes/file copy.
      - Keep it fast: we do NOT hash the entire features.npy.
    """
    features_path = os.path.join(data_out_dir, "features.npy")
    params_path = os.path.join(data_out_dir, "params.npy")
    tfr_meta_path = os.path.join(data_out_dir, "tfr_meta.npz")
    param_meta_path = os.path.join(data_out_dir, "param_meta.npz")

    if not os.path.exists(features_path) or not os.path.exists(params_path):
        raise FileNotFoundError(
            "Prepared data not found. Expected:\n"
            f"  - {features_path}\n"
            f"  - {params_path}\n"
            "Run: python data/prepare_training_data.py"
        )

    X = np.load(features_path, mmap_mode="r")
    y = np.load(params_path, mmap_mode="r")

    N = int(X.shape[0])
    if N <= 0:
        raise ValueError("features.npy has zero samples.")

    # Select a few deterministic indices to hash
    idxs = sorted(set([0, N // 4, N // 2, (3 * N) // 4, N - 1]))

    h = hashlib.sha256()

    # Shapes/dtypes
    h.update(str(tuple(X.shape)).encode("utf-8"))
    h.update(str(X.dtype).encode("utf-8"))
    h.update(str(tuple(y.shape)).encode("utf-8"))
    h.update(str(y.dtype).encode("utf-8"))

    # Hash a few samples (content-based, fast)
    for i in idxs:
        _update_hasher_with_array(h, np.asarray(X[i], dtype=np.float32))
        _update_hasher_with_array(h, np.asarray(y[i], dtype=np.float32))

    payload: Dict[str, Any] = {
        "features_shape": tuple(X.shape),
        "params_shape": tuple(y.shape),
        "features_dtype": str(X.dtype),
        "params_dtype": str(y.dtype),
        "sample_indices_hashed": idxs,
    }

    # Include stable tokenization/meta config
    if os.path.exists(tfr_meta_path):
        meta = np.load(tfr_meta_path)
        keep_keys = [
            "fs",
            "decim",
            "fmin",
            "fmax",
            "n_freqs",
            "stim_onset",
            "pre_sec",
            "post_sec",
            "freq_patch",
            "time_patch",
            "n_time_patches",
            "n_freq_patches",
            "n_tokens",
            "n_tokens_erp",
            "n_tokens_tfr",
            "time_patch_centers",
            "freq_patch_centers",
        ]
        meta_payload = {}
        for k in keep_keys:
            if k in meta.files:
                v = meta[k]
                meta_payload[k] = v
                _update_hasher_with_array(h, np.asarray(v))
        payload["tfr_meta"] = meta_payload
    else:
        payload["tfr_meta"] = None

    # Include parameter bounds/names
    if os.path.exists(param_meta_path):
        pm = np.load(param_meta_path)
        payload["param_names"] = pm["param_names"]
        payload["prior_low"] = pm["prior_low"]
        payload["prior_high"] = pm["prior_high"]

        _update_hasher_with_array(h, np.asarray(pm["param_names"]))
        _update_hasher_with_array(h, np.asarray(pm["prior_low"]))
        _update_hasher_with_array(h, np.asarray(pm["prior_high"]))
    else:
        payload["param_meta"] = None

    sha = h.hexdigest()
    return sha, payload


def create_splits(
    n_samples: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Deterministic train/val/test index splits."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    frac_sum = float(train_frac + val_frac + test_frac)
    if abs(frac_sum - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {frac_sum}.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples).astype(np.int64)

    n_train = int(np.floor(train_frac * n_samples))
    n_val = int(np.floor(val_frac * n_samples))
    n_test = n_samples - n_train - n_val

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    if len(test_idx) != n_test:
        raise RuntimeError("Internal split sizing error.")

    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    if len(np.unique(all_idx)) != n_samples:
        raise RuntimeError("Split indices are not a disjoint partition of [0..N).")

    return {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}


def save_splits(
    splits_path: str,
    splits: Dict[str, np.ndarray],
    *,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    fingerprint_sha256: str,
    fingerprint_payload: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(splits_path), exist_ok=True)

    payload_json = json.dumps(
        fingerprint_payload, sort_keys=True, default=_json_default
    )

    np.savez(
        splits_path,
        train_idx=splits["train_idx"].astype(np.int64),
        val_idx=splits["val_idx"].astype(np.int64),
        test_idx=splits["test_idx"].astype(np.int64),
        split_seed=int(seed),
        train_frac=float(train_frac),
        val_frac=float(val_frac),
        test_frac=float(test_frac),
        created_unix=float(time.time()),
        fingerprint_sha256=np.array(fingerprint_sha256, dtype="S"),
        fingerprint_payload_json=np.array(payload_json, dtype="S"),
    )


def load_splits(splits_path: str) -> Dict[str, Any]:
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"Splits file not found: {splits_path}")

    z = np.load(splits_path, allow_pickle=False)
    out: Dict[str, Any] = {
        "train_idx": z["train_idx"].astype(np.int64),
        "val_idx": z["val_idx"].astype(np.int64),
        "test_idx": z["test_idx"].astype(np.int64),
        "split_seed": int(z["split_seed"]),
        "train_frac": float(z["train_frac"]),
        "val_frac": float(z["val_frac"]),
        "test_frac": float(z["test_frac"]),
        "created_unix": float(z["created_unix"]),
        "fingerprint_sha256": z["fingerprint_sha256"].tobytes().decode("utf-8"),
        "fingerprint_payload_json": z["fingerprint_payload_json"].tobytes().decode("utf-8"),
    }
    return out


def ensure_splits(
    data_out_dir: str,
    splits_path: Optional[str] = None,
    *,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Ensure a persistent split exists and matches current prepared data fingerprint.

    - If missing: create and save.
    - If exists: validate fingerprint and return.
    - If mismatch:
        - overwrite=False: raise (paper-clean safety)
        - overwrite=True: back up old splits and create new.
    """
    if splits_path is None:
        splits_path = os.path.join(data_out_dir, "splits.npz")

    fingerprint_sha, fingerprint_payload = compute_data_fingerprint(data_out_dir)

    if os.path.exists(splits_path) and not overwrite:
        loaded = load_splits(splits_path)
        if loaded["fingerprint_sha256"] != fingerprint_sha:
            msg = (
                "splits.npz fingerprint mismatch.\n"
                f"  splits file: {splits_path}\n"
                f"  splits fingerprint: {loaded['fingerprint_sha256']}\n"
                f"  current fingerprint: {fingerprint_sha}\n\n"
                "This usually means features/params/meta were regenerated but splits.npz was not.\n"
                "Fix options:\n"
                "  - delete the existing splits file and rerun, OR\n"
                "  - run: python data/make_splits.py --overwrite\n"
            )
            raise RuntimeError(msg)
        return loaded

    # If overwriting, back up the old file for audit trail
    if os.path.exists(splits_path) and overwrite:
        ts = time.strftime("%Y%m%d-%H%M%S")
        backup_path = splits_path.replace(".npz", f".bak_{ts}.npz")
        shutil.copy2(splits_path, backup_path)
        LOGGER.info("Backed up existing splits to: %s", backup_path)

    # Create new splits
    X = np.load(os.path.join(data_out_dir, "features.npy"), mmap_mode="r")
    n_samples = int(X.shape[0])

    splits = create_splits(
        n_samples=n_samples,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )

    save_splits(
        splits_path,
        splits,
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        fingerprint_sha256=fingerprint_sha,
        fingerprint_payload=fingerprint_payload,
    )
    LOGGER.info("Created splits: %s", splits_path)
    return load_splits(splits_path)
