# eval/evaluate.py
import os
import sys
import argparse
import pickle
import logging
from typing import Any, Dict, Optional

import numpy as np

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

DATA_OUT = os.path.join(BASE_DIR, "data_out")
MODELS_OUT = os.path.join(BASE_DIR, "models_out")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Local imports (after sys.path)
from data.splits import ensure_splits  # noqa: E402
from models.param_transforms import (  # noqa: E402
    sample_theta_from_gaussian_z,
    theta_to_z,
)

EPS = 1e-8


def _import_tf_or_die():
    try:
        import tensorflow as tf  # type: ignore
        return tf
    except ModuleNotFoundError as e:
        msg = (
            "\nERROR: TensorFlow is not installed in the environment running eval/evaluate.py\n\n"
            f"Python executable:\n  {sys.executable}\n\n"
            "Fix:\n"
            "  python -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  python -m pip install -U pip\n"
            "  python -m pip install -r requirements.txt\n\n"
            "Then rerun:\n"
            "  python eval/evaluate.py\n"
        )
        raise SystemExit(msg) from e


def _get_custom_objects_safe() -> Dict[str, Any]:
    """
    If your model uses custom layers, loading may require custom_objects.
    We try to import get_custom_objects() from models/bi_lstm_model.py.
    """
    try:
        from models.bi_lstm_model import get_custom_objects  # noqa: E402
        return get_custom_objects()
    except Exception:
        return {}


def _load_model_robust(model_path: str):
    """
    Robust Keras loader across TF/Keras versions and custom layers.
    """
    _ = _import_tf_or_die()
    from tensorflow.keras.models import load_model  # noqa: WPS433

    # Ensure custom layer definitions are imported/registered
    try:
        import models.bi_lstm_model  # noqa: F401,E402
    except Exception:
        pass

    custom_objects = _get_custom_objects_safe()

    # Keras 3 sometimes requires safe_mode=False if custom layers exist
    try:
        return load_model(model_path, compile=False, custom_objects=custom_objects, safe_mode=False)
    except TypeError:
        return load_model(model_path, compile=False, custom_objects=custom_objects)


def _scale_X(X_batch: np.ndarray, scaler, n_tokens: int, feature_dim: int) -> np.ndarray:
    flat = X_batch.reshape(-1, feature_dim)
    flat_s = scaler.transform(flat).astype(np.float32)
    return flat_s.reshape(X_batch.shape[0], n_tokens, feature_dim)


def _gaussian_nll_z_np(z_true: np.ndarray, mu_z: np.ndarray, logvar_z: np.ndarray) -> np.ndarray:
    """
    Per-sample diagonal Gaussian NLL in z-space (matches training loss up to a constant).
    """
    logvar_z = np.clip(logvar_z, -10.0, 10.0)
    inv_var = np.exp(-logvar_z)
    nll = 0.5 * (inv_var * (z_true - mu_z) ** 2 + logvar_z)
    return np.sum(nll, axis=1)


def _maybe_plot_attention(
    *,
    model,
    scaler,
    X_memmap,
    test_idx: np.ndarray,
    n_tokens: int,
    feature_dim: int,
    P: int,
    n_time: int,
    n_freq: int,
    n_tokens_erp: int,
    time_centers: np.ndarray,
    freq_centers: np.ndarray,
    param_names: list,
    viz: int,
) -> None:
    """
    Optional attention visualization (ERP and TFR relevance heatmaps).

    Requires build_bi_lstm_model(..., return_attention=True) to output (pred, scores)
    where scores ~ (B, H, P, tokens).
    """
    # Safe matplotlib import for headless runs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    from models.bi_lstm_model import build_bi_lstm_model  # noqa: E402

    # Try to match training config if available
    cfg_path = os.path.join(MODELS_OUT, "model_config.npz")
    if os.path.exists(cfg_path):
        cfg = np.load(cfg_path)
        d_model = int(cfg.get("d_model", 128))
        num_layers = int(cfg.get("num_layers", 4))
        num_heads = int(cfg.get("num_heads", 4))
        ff_dim = int(cfg.get("ff_dim", 256))
        dropout_rate = float(cfg.get("dropout_rate", 0.15))
    else:
        d_model, num_layers, num_heads, ff_dim, dropout_rate = 128, 4, 4, 256, 0.15

    attn_model = build_bi_lstm_model(
        n_tokens=n_tokens,
        feature_dim=feature_dim,
        n_params=P,
        n_time_patches=n_time,
        n_freq_patches=n_freq,
        n_tokens_erp=n_tokens_erp,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout_rate=dropout_rate,
        return_attention=True,
    )
    attn_model.set_weights(model.get_weights())

    viz = int(min(viz, len(test_idx)))
    viz_ids = np.asarray(test_idx[:viz], dtype=np.int64)

    xb = np.asarray(X_memmap[viz_ids], dtype=np.float32)
    xb = _scale_X(xb, scaler, n_tokens=n_tokens, feature_dim=feature_dim)

    # scores: (B, H, P, tokens)
    _, scores = attn_model.predict(xb, verbose=0)
    scores = scores.mean(axis=1).astype(np.float32)  # (B, P, tokens)

    for bi in range(viz):
        for pi, nm in enumerate(param_names):
            w = scores[bi, pi]  # (tokens,)

            w_erp = w[:n_tokens_erp]  # (n_time,)
            w_tfr = w[n_tokens_erp:].reshape(n_time, n_freq).T  # (n_freq, n_time)

            plt.figure(figsize=(7, 2.5))
            plt.plot(time_centers, w_erp)
            plt.xlabel("Time (s) rel stim")
            plt.ylabel("Attention")
            plt.title(f"ERP relevance — sample {int(viz_ids[bi])} — {nm}")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"attn_erp_s{bi}_{nm}.png"), dpi=200)
            plt.close()

            plt.figure(figsize=(8, 3))
            plt.imshow(
                w_tfr,
                aspect="auto",
                origin="lower",
                extent=[float(time_centers[0]), float(time_centers[-1]), float(freq_centers[0]), float(freq_centers[-1])],
            )
            plt.colorbar(label="Attention")
            plt.xlabel("Time (s) rel stim")
            plt.ylabel("Frequency (Hz)")
            plt.title(f"TFR relevance — sample {int(viz_ids[bi])} — {nm}")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"attn_tfr_s{bi}_{nm}.png"), dpi=200)
            plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JR inverse model on locked test split.")
    parser.add_argument("--chunk", type=int, default=128, help="Prediction chunk size.")
    parser.add_argument("--mc-samples", type=int, default=200, help="Posterior samples per example.")
    parser.add_argument("--attention", action="store_true", help="Also generate attention relevance plots.")
    parser.add_argument("--viz", type=int, default=5, help="How many test examples to visualize for attention.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("evaluate")

    _ = _import_tf_or_die()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # ---- Load prepared data ----
    X = np.load(os.path.join(DATA_OUT, "features.npy"), mmap_mode="r")      # (N, tokens, feat)
    y_theta = np.load(os.path.join(DATA_OUT, "params.npy"), mmap_mode="r")  # (N, P)

    meta = np.load(os.path.join(DATA_OUT, "tfr_meta.npz"))
    n_time = int(meta["n_time_patches"])
    n_freq = int(meta["n_freq_patches"])
    n_tokens_erp = int(meta["n_tokens_erp"]) if "n_tokens_erp" in meta.files else n_time
    time_centers = meta["time_patch_centers"].astype(np.float32)
    freq_centers = meta["freq_patch_centers"].astype(np.float32)

    N, n_tokens, feature_dim = X.shape
    P = int(y_theta.shape[1])

    # ---- Locked split (paper-clean) ----
    splits = ensure_splits(
        data_out_dir=DATA_OUT,
        seed=42,
        train_frac=0.70,
        val_frac=0.15,
        test_frac=0.15,
        overwrite=False,
    )
    test_idx = np.asarray(splits["test_idx"], dtype=np.int64)
    y_test = np.asarray(y_theta[test_idx], dtype=np.float32)
    logger.info("Loaded splits: test=%d", len(test_idx))

    # ---- Load model + scaler + bounds ----
    model_path = os.path.join(MODELS_OUT, "jr_paramtoken_inverse_model.keras")
    model = _load_model_robust(model_path)

    with open(os.path.join(MODELS_OUT, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    bounds = np.load(os.path.join(MODELS_OUT, "param_bounds.npz"))
    param_names = [x.decode("utf-8") for x in bounds["param_names"]]
    prior_low = bounds["prior_low"].astype(np.float32)
    prior_high = bounds["prior_high"].astype(np.float32)

    # ---- Predict in chunks ----
    preds = []
    chunk = int(max(1, args.chunk))
    for start in range(0, len(test_idx), chunk):
        sl = test_idx[start:start + chunk]
        xb = np.asarray(X[sl], dtype=np.float32)
        xb = _scale_X(xb, scaler, n_tokens=n_tokens, feature_dim=feature_dim)
        yp = model.predict(xb, verbose=0)
        preds.append(yp.astype(np.float32))
    pred = np.concatenate(preds, axis=0)

    mu_z = pred[:, :P]
    logvar_z = np.clip(pred[:, P:], -10.0, 10.0)

    # ---- NLL(z) ----
    z_true = theta_to_z(y_test, prior_low, prior_high)
    nll_z = _gaussian_nll_z_np(z_true=z_true, mu_z=mu_z, logvar_z=logvar_z)
    logger.info("=== Test NLL(z) mean ± std === %.3f ± %.3f", float(nll_z.mean()), float(nll_z.std()))

    # ---- Posterior samples in theta-space ----
    theta_samps = sample_theta_from_gaussian_z(
        mu_z,
        logvar_z,
        prior_low,
        prior_high,
        n_samples=int(args.mc_samples),
        seed=0,
    )
    theta_mean = theta_samps.mean(axis=0)
    theta_std = theta_samps.std(axis=0)

    # ---- Metrics ----
    abs_err = np.abs(theta_mean - y_test)
    rel_err = abs_err / (np.abs(y_test) + EPS) * 100.0

    rel_mean = rel_err.mean(axis=0)
    rel_std = rel_err.std(axis=0)
    rel_med = np.median(rel_err, axis=0)
    rel_q05 = np.quantile(rel_err, 0.05, axis=0)
    rel_q95 = np.quantile(rel_err, 0.95, axis=0)

    snr_db = 20.0 * np.log10(
        np.linalg.norm(y_test, axis=0) / (np.linalg.norm(y_test - theta_mean, axis=0) + EPS)
    )
    rmse = np.sqrt(np.mean((theta_mean - y_test) ** 2, axis=0))

    lo_q = np.quantile(theta_samps, 0.05, axis=0)
    hi_q = np.quantile(theta_samps, 0.95, axis=0)
    coverage = ((y_test >= lo_q) & (y_test <= hi_q)).mean(axis=0)

    prior_std = (prior_high - prior_low) / np.sqrt(12.0)
    contr = theta_std.mean(axis=0) / (prior_std + EPS)

    # ---- Print summary ----
    print("\n=== Test Relative Error (%) [mean ± std] ===")
    for i, nm in enumerate(param_names):
        print(f"{nm:8s}: {rel_mean[i]:.2f}% ± {rel_std[i]:.2f}%")

    print("\n=== Test Relative Error (%) [median, 5–95%] ===")
    for i, nm in enumerate(param_names):
        print(f"{nm:8s}: {rel_med[i]:.2f}%  ( {rel_q05[i]:.2f}% – {rel_q95[i]:.2f}% )")

    print("\n=== Parameter-space SNR proxy (dB) ===")
    for i, nm in enumerate(param_names):
        print(f"{nm:8s}: {snr_db[i]:.2f} dB")

    print("\n=== Test RMSE (theta units) ===")
    for i, nm in enumerate(param_names):
        print(f"{nm:8s}: {rmse[i]:.4f}")

    print("\n=== 90% Interval Coverage ===")
    for i, nm in enumerate(param_names):
        print(f"{nm:8s}: {coverage[i]*100:.1f}%")

    print("\n=== Posterior Contraction (avg post std / prior std) ===")
    for i, nm in enumerate(param_names):
        print(f"{nm:8s}: {contr[i]:.3f}")

    # ---- Save outputs (single source of truth) ----
    np.savez(
        os.path.join(PLOTS_DIR, "eval_test_outputs.npz"),
        test_idx=test_idx.astype(np.int64),
        y_test=y_test.astype(np.float32),
        mu_z=mu_z.astype(np.float32),
        logvar_z=logvar_z.astype(np.float32),
        nll_z=nll_z.astype(np.float32),
        theta_mean=theta_mean.astype(np.float32),
        theta_std=theta_std.astype(np.float32),
        rel_err=rel_err.astype(np.float32),
        coverage90=coverage.astype(np.float32),
        contraction=contr.astype(np.float32),
        snr_db=snr_db.astype(np.float32),
        rmse=rmse.astype(np.float32),
        param_names=np.array(param_names, dtype="S"),
        prior_low=prior_low.astype(np.float32),
        prior_high=prior_high.astype(np.float32),
        n_time=int(n_time),
        n_freq=int(n_freq),
        n_tokens=int(n_tokens),
        n_tokens_erp=int(n_tokens_erp),
        time_patch_centers=time_centers.astype(np.float32),
        freq_patch_centers=freq_centers.astype(np.float32),
    )

    # Backward-compatible arrays used by older plotting scripts
    np.save(os.path.join(PLOTS_DIR, "theta_mean.npy"), theta_mean)
    np.save(os.path.join(PLOTS_DIR, "theta_std.npy"), theta_std)
    np.save(os.path.join(PLOTS_DIR, "rel_err_mean.npy"), rel_mean)
    np.save(os.path.join(PLOTS_DIR, "rel_err_median.npy"), rel_med)
    np.save(os.path.join(PLOTS_DIR, "snr_db.npy"), snr_db)
    np.save(os.path.join(PLOTS_DIR, "coverage90.npy"), coverage)
    np.save(os.path.join(PLOTS_DIR, "contraction.npy"), contr)
    np.save(os.path.join(PLOTS_DIR, "nll_z_test.npy"), nll_z)

    logger.info("Saved evaluation outputs to: %s", PLOTS_DIR)

    # ---- Optional attention plots ----
    if args.attention:
        try:
            _maybe_plot_attention(
                model=model,
                scaler=scaler,
                X_memmap=X,
                test_idx=test_idx,
                n_tokens=n_tokens,
                feature_dim=feature_dim,
                P=P,
                n_time=n_time,
                n_freq=n_freq,
                n_tokens_erp=n_tokens_erp,
                time_centers=time_centers,
                freq_centers=freq_centers,
                param_names=param_names,
                viz=int(args.viz),
            )
            logger.info("Saved attention plots to: %s", PLOTS_DIR)
        except Exception as e:
            logger.warning("Attention plotting skipped due to error: %r", e)


if __name__ == "__main__":
    main()
