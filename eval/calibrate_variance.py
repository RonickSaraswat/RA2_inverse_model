# eval/calibrate_variance.py
"""
Post-hoc variance calibration to reduce overconfidence.

Fits per-parameter scale factors s_p on the VALIDATION split so that
credible intervals match nominal coverage (multi-level by default).

We calibrate in z-space (Gaussian) because your model outputs N(mu_z, diag(var_z)).
Scaling std by s_p keeps the posterior family Gaussian and preserves log-prob math.

Saves:
  models_out/variance_calibration.npz

References:
  - Guo et al., 2017, "On Calibration of Modern Neural Networks" (temperature scaling idea)
  - Kuleshov et al., 2018, "Accurate Uncertainties for Deep Learning Using Calibrated Regression"
  - Lakshminarayanan et al., 2017, "Deep Ensembles" (often improves calibration further)
"""

from __future__ import annotations

import argparse
import os
import sys
import pickle
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

# Reduce TF verbosity (must be set before importing tensorflow)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

DATA_OUT_DEFAULT = os.path.join(BASE_DIR, "data_out")
MODELS_OUT_DEFAULT = os.path.join(BASE_DIR, "models_out")

from data.splits import ensure_splits  # noqa: E402
from models.param_transforms import theta_to_z, z_to_theta  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("calibrate_variance")


def _import_tf_or_die():
    try:
        import tensorflow as tf  # type: ignore
        return tf
    except ModuleNotFoundError as e:
        raise SystemExit(
            "TensorFlow not installed. Activate your .venv and install requirements."
        ) from e


def get_custom_objects_safe() -> Dict[str, Any]:
    try:
        from models.bi_lstm_model import get_custom_objects  # noqa: E402
        return get_custom_objects()
    except Exception:
        return {}


def load_trained_model(model_path: str):
    _ = _import_tf_or_die()
    from tensorflow.keras.models import load_model  # noqa: WPS433

    # Ensure custom layers get registered
    try:
        import models.bi_lstm_model  # noqa: F401
    except Exception:
        pass

    custom_objects = get_custom_objects_safe()
    try:
        return load_model(model_path, compile=False, custom_objects=custom_objects, safe_mode=False)
    except TypeError:
        return load_model(model_path, compile=False, custom_objects=custom_objects)


def choose_model_path(models_out: str) -> str:
    best = os.path.join(models_out, "jr_paramtoken_inverse_model_best.keras")
    final = os.path.join(models_out, "jr_paramtoken_inverse_model.keras")
    if os.path.exists(best):
        return best
    if os.path.exists(final):
        return final
    raise FileNotFoundError(f"Could not find model at:\n  {best}\n  {final}")


def scale_X(X_batch: np.ndarray, scaler, n_tokens: int, feature_dim: int) -> np.ndarray:
    flat = X_batch.reshape(-1, feature_dim)
    flat_s = scaler.transform(flat).astype(np.float32)
    return flat_s.reshape(X_batch.shape[0], n_tokens, feature_dim)


def predict_mu_logvar(
    model,
    xb: np.ndarray,
    P: int,
    mc_dropout: int = 0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns mu_z, logvar_z (both float32) for a batch.
    If mc_dropout > 1: uses dropout-at-inference (epistemic) and returns total variance.
    """
    tf = _import_tf_or_die()
    if mc_dropout is None:
        mc_dropout = 0
    mc_dropout = int(mc_dropout)

    if mc_dropout <= 1:
        yp = model.predict(xb, verbose=0).astype(np.float32)
        mu_z = yp[:, :P]
        logvar_z = yp[:, P:]
        logvar_z = np.clip(logvar_z, -10.0, 10.0).astype(np.float32)
        return mu_z, logvar_z

    tf.random.set_seed(int(seed))

    sum_mu = np.zeros((xb.shape[0], P), dtype=np.float64)
    sum_mu2 = np.zeros((xb.shape[0], P), dtype=np.float64)
    sum_var = np.zeros((xb.shape[0], P), dtype=np.float64)

    xb_tf = tf.convert_to_tensor(xb, dtype=tf.float32)
    for _ in range(mc_dropout):
        yp = model(xb_tf, training=True).numpy().astype(np.float32)  # (B, 2P)
        mu = yp[:, :P].astype(np.float64)
        logvar = np.clip(yp[:, P:], -10.0, 10.0).astype(np.float64)
        var = np.exp(logvar)

        sum_mu += mu
        sum_mu2 += mu * mu
        sum_var += var

    m = float(mc_dropout)
    mu_bar = sum_mu / m
    ale = sum_var / m
    epi = (sum_mu2 / m) - (mu_bar * mu_bar)
    var_total = np.maximum(ale + epi, 1e-12)

    mu_z = mu_bar.astype(np.float32)
    logvar_z = np.log(var_total).astype(np.float32)
    return mu_z, logvar_z


def coverage_theta_interval(
    theta_true: np.ndarray,
    mu_z: np.ndarray,
    logvar_z: np.ndarray,
    low: float,
    high: float,
    level: float,
    scale: float,
) -> float:
    """
    Compute empirical coverage of the nominal central interval at 'level'
    after scaling std by 'scale' for one parameter.
    """
    theta_true = np.asarray(theta_true, dtype=np.float64)
    mu_z = np.asarray(mu_z, dtype=np.float64)
    logvar_z = np.asarray(logvar_z, dtype=np.float64)

    std = np.exp(0.5 * np.clip(logvar_z, -10.0, 10.0)) * float(scale)
    q = float(norm.ppf(0.5 + 0.5 * float(level)))  # central interval
    z_lo = mu_z - q * std
    z_hi = mu_z + q * std

    th_lo = z_to_theta(z_lo, low, high).astype(np.float64)
    th_hi = z_to_theta(z_hi, low, high).astype(np.float64)

    inside = (theta_true >= th_lo) & (theta_true <= th_hi)
    return float(np.mean(inside))


def fit_scale_per_param(
    theta_true: np.ndarray,
    mu_z: np.ndarray,
    logvar_z: np.ndarray,
    low: float,
    high: float,
    levels: List[float],
    grid_lo: float = 0.25,
    grid_hi: float = 8.0,
    grid_n: int = 81,
) -> float:
    """
    Fit scale s >= 0 by minimizing squared coverage error across multiple levels:
      argmin_s sum_k (cov(s, level_k) - level_k)^2

    Uses a log-spaced grid + a local refinement grid.
    """
    levels = [float(x) for x in levels]
    scales = np.exp(np.linspace(np.log(grid_lo), np.log(grid_hi), int(grid_n)))

    def obj(s: float) -> float:
        return float(
            np.sum([(coverage_theta_interval(theta_true, mu_z, logvar_z, low, high, lv, s) - lv) ** 2 for lv in levels])
        )

    best_s = float(scales[0])
    best_obj = obj(best_s)

    for s in scales[1:]:
        o = obj(float(s))
        if o < best_obj:
            best_obj = o
            best_s = float(s)

    # local refinement around best_s
    ref_lo = max(grid_lo, best_s / 2.0)
    ref_hi = min(grid_hi, best_s * 2.0)
    ref_scales = np.exp(np.linspace(np.log(ref_lo), np.log(ref_hi), 41))

    for s in ref_scales:
        o = obj(float(s))
        if o < best_obj:
            best_obj = o
            best_s = float(s)

    return float(best_s)


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate posterior variance scales to reduce overconfidence.")
    ap.add_argument("--data-out", type=str, default=DATA_OUT_DEFAULT)
    ap.add_argument("--models-out", type=str, default=MODELS_OUT_DEFAULT)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--levels", type=float, nargs="+", default=[0.5, 0.8, 0.9],
                    help="Nominal central interval levels to match on validation.")
    ap.add_argument("--mc-dropout", type=int, default=0,
                    help="If >1, calibrate using MC-dropout predictive variance (slower). Must match evaluation.")
    ap.add_argument("--seed", type=int, default=0, help="Seed used only for MC-dropout randomness.")
    args = ap.parse_args()

    data_out = args.data_out
    models_out = args.models_out

    X = np.load(os.path.join(data_out, "features.npy"), mmap_mode="r")
    y_theta = np.load(os.path.join(data_out, "params.npy"), mmap_mode="r")

    N, n_tokens, feature_dim = X.shape
    P = int(y_theta.shape[1])

    splits = ensure_splits(
        data_out_dir=data_out,
        seed=42,
        train_frac=0.70,
        val_frac=0.15,
        test_frac=0.15,
        overwrite=False,
    )
    val_idx = np.asarray(splits["val_idx"], dtype=np.int64)
    y_val = np.asarray(y_theta[val_idx], dtype=np.float32)

    bounds = np.load(os.path.join(models_out, "param_bounds.npz"))
    param_names = [x.decode("utf-8") for x in bounds["param_names"]]
    low = bounds["prior_low"].astype(np.float32)
    high = bounds["prior_high"].astype(np.float32)

    with open(os.path.join(models_out, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    model_path = choose_model_path(models_out)
    LOGGER.info("Loading model for calibration: %s", model_path)
    model = load_trained_model(model_path)

    # Predict on validation split
    mu_all = np.zeros((len(val_idx), P), dtype=np.float32)
    lv_all = np.zeros((len(val_idx), P), dtype=np.float32)

    chunk = int(args.chunk)
    for start in range(0, len(val_idx), chunk):
        sl = val_idx[start : start + chunk]
        xb = np.asarray(X[sl], dtype=np.float32)
        xb = scale_X(xb, scaler, n_tokens=n_tokens, feature_dim=feature_dim)

        mu_z, logvar_z = predict_mu_logvar(model, xb, P=P, mc_dropout=int(args.mc_dropout), seed=int(args.seed))

        mu_all[start : start + len(sl)] = mu_z
        lv_all[start : start + len(sl)] = logvar_z

    LOGGER.info("Fitting per-parameter variance scales on validation (N=%d)", len(val_idx))
    var_scale = np.ones((P,), dtype=np.float32)

    cov_before = np.zeros((P, len(args.levels)), dtype=np.float64)
    cov_after = np.zeros((P, len(args.levels)), dtype=np.float64)

    for p in range(P):
        th = y_val[:, p]
        mu = mu_all[:, p]
        lv = lv_all[:, p]

        # baseline coverages at s=1
        for j, lv_nom in enumerate(args.levels):
            cov_before[p, j] = coverage_theta_interval(th, mu, lv, float(low[p]), float(high[p]), float(lv_nom), scale=1.0)

        s = fit_scale_per_param(
            theta_true=th,
            mu_z=mu,
            logvar_z=lv,
            low=float(low[p]),
            high=float(high[p]),
            levels=list(args.levels),
        )
        var_scale[p] = float(s)

        # after
        for j, lv_nom in enumerate(args.levels):
            cov_after[p, j] = coverage_theta_interval(th, mu, lv, float(low[p]), float(high[p]), float(lv_nom), scale=float(s))

        LOGGER.info(
            "%-8s | scale=%.3f | cov@%s before=%s after=%s",
            param_names[p],
            float(s),
            ",".join([f"{x:.2f}" for x in args.levels]),
            ",".join([f"{x:.3f}" for x in cov_before[p]]),
            ",".join([f"{x:.3f}" for x in cov_after[p]]),
        )

    out_path = os.path.join(models_out, "variance_calibration.npz")
    np.savez(
        out_path,
        param_names=np.array(param_names, dtype="S"),
        var_scale=var_scale.astype(np.float32),
        logvar_shift=(2.0 * np.log(np.maximum(var_scale, 1e-8))).astype(np.float32),
        levels=np.array(args.levels, dtype=np.float32),
        cov_before=cov_before.astype(np.float64),
        cov_after=cov_after.astype(np.float64),
        mc_dropout=int(args.mc_dropout),
        seed=int(args.seed),
    )
    LOGGER.info("Saved variance calibration to: %s", out_path)
    LOGGER.info("Next: run evaluation with --apply-calibration to use these scales on TEST.")


if __name__ == "__main__":
    main()
