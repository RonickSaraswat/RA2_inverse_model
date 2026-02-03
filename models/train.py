#models/train.py
import os
import sys
import pickle
import logging
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

DATA_OUT = os.path.join(BASE_DIR, "data_out")
MODELS_OUT = os.path.join(BASE_DIR, "models_out")
os.makedirs(MODELS_OUT, exist_ok=True)

from data.splits import ensure_splits  # noqa: E402
from models.bi_lstm_model import build_bi_lstm_model  # noqa: E402
from models.param_transforms import theta_to_z  # noqa: E402


def _import_tensorflow_or_die():
    try:
        import tensorflow as tf  # type: ignore
        return tf
    except ModuleNotFoundError as e:
        msg = (
            "\nERROR: TensorFlow is not installed in the Python environment running this script.\n\n"
            f"Python executable:\n  {sys.executable}\n\n"
            "Fix (recommended, reproducible):\n"
            "  python -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  python -m pip install -U pip\n"
            "  python -m pip install -r requirements.txt\n\n"
            "If you are on Apple Silicon and want GPU acceleration (optional):\n"
            "  xcode-select --install\n"
            "  python -m pip install tensorflow-metal\n\n"
            "Then rerun:\n"
            "  python models/train.py\n"
        )
        raise SystemExit(msg) from e


@dataclass
class TrainConfig:
    # Repro
    seed: int = 42

    # Optimization
    batch_size: int = 64
    epochs: int = 200
    warmup_epochs: int = 20

    learning_rate: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 1e-5
    clipnorm: float = 1.0

    dropout_rate: float = 0.10
    augment_std: float = 0.01  # small, after scaling

    # Model
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    ff_dim: int = 256

    # Splits
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15


def main() -> None:
    tf = _import_tensorflow_or_die()
    from tensorflow.keras.callbacks import (  # noqa: WPS433
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
        CSVLogger,
    )
    from tensorflow.keras.optimizers import Adam  # noqa: WPS433

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("train")

    cfg = TrainConfig()

    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    logger.info("Loading features/params from %s", DATA_OUT)
    X_mem = np.load(os.path.join(DATA_OUT, "features.npy"), mmap_mode="r")      # (N, tokens, feat)
    y_theta = np.load(os.path.join(DATA_OUT, "params.npy"), mmap_mode="r")      # (N, P)

    meta = np.load(os.path.join(DATA_OUT, "tfr_meta.npz"))
    param_meta = np.load(os.path.join(DATA_OUT, "param_meta.npz"))

    param_names = [x.decode("utf-8") for x in param_meta["param_names"]]
    low = param_meta["prior_low"].astype(np.float32)
    high = param_meta["prior_high"].astype(np.float32)

    N, n_tokens, feature_dim = X_mem.shape
    P = int(y_theta.shape[1])

    n_time_patches = int(meta["n_time_patches"])
    n_freq_patches = int(meta["n_freq_patches"])
    n_tokens_erp = int(meta["n_tokens_erp"]) if "n_tokens_erp" in meta.files else int(n_time_patches)

    logger.info("X shape=%s | y shape=%s", X_mem.shape, y_theta.shape)
    logger.info("Params=%s", param_names)
    logger.info(
        "Tokens=%d feat_dim=%d | TF grid=%dx%d | ERP tokens=%d",
        n_tokens, feature_dim, n_time_patches, n_freq_patches, n_tokens_erp,
    )

    # Locked splits
    splits = ensure_splits(
        data_out_dir=DATA_OUT,
        seed=cfg.seed,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        overwrite=False,
    )
    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]
    test_idx = splits["test_idx"]
    logger.info("Split sizes: train=%d val=%d test=%d", len(train_idx), len(val_idx), len(test_idx))

    # Target transform (z-space)
    y_z = theta_to_z(np.asarray(y_theta, dtype=np.float32), low, high)  # (N, P)

    # Load train/val arrays into memory (fast + stable training)
    logger.info("Loading train/val arrays into RAM...")
    X_train = np.asarray(X_mem[train_idx], dtype=np.float32)  # (Ntr, tokens, feat)
    X_val = np.asarray(X_mem[val_idx], dtype=np.float32)      # (Nva, tokens, feat)
    y_train = np.asarray(y_z[train_idx], dtype=np.float32)
    y_val = np.asarray(y_z[val_idx], dtype=np.float32)

    # Fit scaler on training only
    logger.info("Fitting StandardScaler on training only...")
    scaler = StandardScaler()
    flat_train = X_train.reshape(-1, feature_dim)
    scaler.fit(flat_train)

    # Apply scaling in-place (no leakage: scaler fitted on train only)
    mu = scaler.mean_.astype(np.float32)
    sd = scaler.scale_.astype(np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)

    X_train -= mu[None, None, :]
    X_train /= sd[None, None, :]
    X_val -= mu[None, None, :]
    X_val /= sd[None, None, :]

    # Build tf.data
    def augment(x, y):
        if cfg.augment_std <= 0:
            return x, y
        noise = tf.random.normal(tf.shape(x), stddev=cfg.augment_std, dtype=x.dtype)
        return x + noise, y

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(buffer_size=min(8192, len(X_train)), seed=cfg.seed, reshuffle_each_iteration=True)
        .batch(cfg.batch_size)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Losses
    def mse_mu_only(y_true, y_pred):
        mu_z = y_pred[:, :P]
        return tf.reduce_mean(tf.reduce_sum(tf.square(y_true - mu_z), axis=1))

    def gaussian_nll_z(y_true, y_pred):
        mu_z = y_pred[:, :P]
        logvar_z = y_pred[:, P:]
        logvar_z = tf.clip_by_value(logvar_z, -10.0, 10.0)
        inv_var = tf.exp(-logvar_z)
        nll = 0.5 * (inv_var * tf.square(y_true - mu_z) + logvar_z)
        return tf.reduce_mean(tf.reduce_sum(nll, axis=1))

    # Build model
    model = build_bi_lstm_model(
        n_tokens=n_tokens,
        feature_dim=feature_dim,
        n_params=P,
        n_time_patches=n_time_patches,
        n_freq_patches=n_freq_patches,
        n_tokens_erp=n_tokens_erp,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        ff_dim=cfg.ff_dim,
        dropout_rate=cfg.dropout_rate,
        return_attention=False,
    )
    model.summary()

    # Optimizer
    try:
        from tensorflow.keras.optimizers import AdamW  # noqa: WPS433

        opt = AdamW(
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            clipnorm=cfg.clipnorm,
        )
        logger.info("Using AdamW.")
    except Exception:
        opt = Adam(learning_rate=cfg.learning_rate, clipnorm=cfg.clipnorm)
        logger.info("Using Adam (AdamW unavailable).")

    # Callbacks
    best_path = os.path.join(MODELS_OUT, "jr_paramtoken_inverse_model_best.keras")
    final_path = os.path.join(MODELS_OUT, "jr_paramtoken_inverse_model.keras")

    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=cfg.min_lr,
            verbose=1,
        ),
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1),
        ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True, verbose=1),
        CSVLogger(os.path.join(MODELS_OUT, "train_log.csv")),
    ]

    # ---------------- Phase 1: mean warmup (forces learning signal) ----------------
    logger.info("Phase 1/%d: warmup_epochs=%d with MSE(mu) in z-space", 2, cfg.warmup_epochs)
    model.compile(optimizer=opt, loss=mse_mu_only)
    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.warmup_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # ---------------- Phase 2: full Bayesian NLL ----------------
    logger.info("Phase 2/%d: NLL training for up to %d epochs", 2, cfg.epochs)
    model.compile(optimizer=opt, loss=gaussian_nll_z)
    hist2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model (best weights already restored by EarlyStopping)
    model.save(final_path)
    logger.info("Saved model: %s", final_path)
    logger.info("Best checkpoint: %s", best_path)

    # Save scaler
    with open(os.path.join(MODELS_OUT, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Save history merged
    history = {"phase1": hist1.history, "phase2": hist2.history}
    with open(os.path.join(MODELS_OUT, "training_history.pkl"), "wb") as f:
        pickle.dump(history, f)

    # Save bounds + config
    np.savez(
        os.path.join(MODELS_OUT, "param_bounds.npz"),
        param_names=np.array(param_names, dtype="S"),
        prior_low=low,
        prior_high=high,
    )

    np.savez(
        os.path.join(MODELS_OUT, "model_config.npz"),
        n_tokens=int(n_tokens),
        feature_dim=int(feature_dim),
        n_params=int(P),
        n_time_patches=int(n_time_patches),
        n_freq_patches=int(n_freq_patches),
        n_tokens_erp=int(n_tokens_erp),
        d_model=int(cfg.d_model),
        num_layers=int(cfg.num_layers),
        num_heads=int(cfg.num_heads),
        ff_dim=int(cfg.ff_dim),
        dropout_rate=float(cfg.dropout_rate),
        seed=int(cfg.seed),
    )

    np.savez(os.path.join(MODELS_OUT, "train_config.npz"), **asdict(cfg))
    logger.info("Saved scaler/history/bounds/model_config/train_config to %s", MODELS_OUT)


if __name__ == "__main__":
    main()
