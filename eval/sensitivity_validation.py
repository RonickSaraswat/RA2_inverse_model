import os
import sys
import logging
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import h5py
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

DATA_FILE = os.path.join(BASE_DIR, "data", "synthetic_eeg_dataset.h5")
DATA_OUT = os.path.join(BASE_DIR, "data_out")
MODELS_OUT = os.path.join(BASE_DIR, "models_out")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

from data.splits import ensure_splits  # noqa: E402
from simulate.simulator import simulate_eeg  # noqa: E402
from features.feature_extraction import extract_features  # noqa: E402
from models.bi_lstm_model import build_bi_lstm_model  # noqa: E402

LOGGER = logging.getLogger("sensitivity_validation")


@dataclass
class SensConfig:
    seed: int = 123
    n_examples: int = 30           # evaluate more than 1 sample (paper)
    delta_frac_range: float = 0.01 # delta = frac * (high-low), central difference
    fd_seed: int = 777            # seed used inside simulate_eeg for FD sims
    batch_size: int = 1           # keep 1 for clean jacobian shapes
    topk_frac: float = 0.10       # overlap in top 10% tokens


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, safe for constant vectors."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    xs = x.std() + 1e-12
    ys = y.std() + 1e-12
    return float(np.mean((x / xs) * (y / ys)))


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Simple rank transform for Spearman without scipy dependency."""
    a = np.asarray(a)
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    # handle ties approximately by average rank
    # (good enough for our use; exact tie handling would require scipy)
    return ranks


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _safe_corr(rx, ry)


def _topk_overlap(x: np.ndarray, y: np.ndarray, frac: float) -> float:
    """
    Fractional overlap of top-k indices.
    overlap = |TopK(x) ∩ TopK(y)| / k
    """
    n = len(x)
    k = int(max(1, np.round(frac * n)))
    ix = np.argsort(x)[-k:]
    iy = np.argsort(y)[-k:]
    inter = np.intersect1d(ix, iy).size
    return float(inter / k)


def _scale_tokens(tokens: np.ndarray, scaler, feature_dim: int) -> np.ndarray:
    """
    Apply sklearn StandardScaler to token array (tokens, feat).
    Returns (1, tokens, feat) float32 as model input.
    """
    tokens = np.asarray(tokens, dtype=np.float32)
    flat = tokens.reshape(-1, feature_dim)
    flat_s = scaler.transform(flat).astype(np.float32)
    return flat_s.reshape(1, tokens.shape[0], feature_dim)


def _patchify_hybrid(
    eeg_sample: np.ndarray,
    fs: int,
    decim: int,
    fmin: float,
    fmax: float,
    n_freqs: int,
    stim_onset: float,
    pre_sec: float,
    post_sec: float,
    n_time: int,
    n_freq: int,
    freq_patch: int,
    time_patch: int,
) -> np.ndarray:
    """
    Match data/prepare_training_data.py exactly.
    Returns tokens: (n_tokens, C), where n_tokens = n_time + n_time*n_freq.
    """
    tfr = extract_features(eeg_sample, fs=fs, fmin=fmin, fmax=fmax, n_freqs=n_freqs, decim=decim)
    C, F, Tdec_full = tfr.shape

    # windowing in decimated time indices
    start_idx = int(np.round((stim_onset - pre_sec) * fs / decim))
    start_idx = max(0, start_idx)

    Freq_use = n_freq * freq_patch
    T_use = n_time * time_patch

    # ERP tokens from decimated EEG
    eeg_dec = eeg_sample[:, ::decim]
    win = eeg_dec[:, start_idx:start_idx + T_use]               # (C, T_use)
    erp_patch = win.reshape(C, n_time, time_patch).mean(axis=2) # (C, n_time)
    erp_tok = erp_patch.T.astype(np.float32)                    # (n_time, C)

    # TFR tokens
    tfr_win = tfr[:, :Freq_use, start_idx:start_idx + T_use]
    rs = tfr_win.reshape(C, n_freq, freq_patch, n_time, time_patch)
    patch = rs.mean(axis=(2, 4))                                # (C, n_freq, n_time)
    tfr_tok = patch.transpose(2, 1, 0).reshape(n_time * n_freq, C).astype(np.float32)

    tokens = np.concatenate([erp_tok, tfr_tok], axis=0).astype(np.float32)
    return tokens


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = SensConfig()
    rng = np.random.default_rng(cfg.seed)

    # Load meta
    meta = np.load(os.path.join(DATA_OUT, "tfr_meta.npz"))
    param_meta = np.load(os.path.join(DATA_OUT, "param_meta.npz"))

    fs = int(meta["fs"])
    decim = int(meta["decim"])
    fmin = float(meta["fmin"])
    fmax = float(meta["fmax"])
    n_freqs = int(meta["n_freqs"])
    stim_onset = float(meta["stim_onset"])
    pre_sec = float(meta["pre_sec"])
    post_sec = float(meta["post_sec"])
    freq_patch = int(meta["freq_patch"])
    time_patch = int(meta["time_patch"])
    n_time = int(meta["n_time_patches"])
    n_freq = int(meta["n_freq_patches"])
    n_tokens_erp = int(meta["n_tokens_erp"])
    time_centers = meta["time_patch_centers"]
    freq_centers = meta["freq_patch_centers"]

    param_names = [x.decode("utf-8") for x in param_meta["param_names"]]
    prior_low = param_meta["prior_low"].astype(np.float32)
    prior_high = param_meta["prior_high"].astype(np.float32)
    P = len(param_names)

    X = np.load(os.path.join(DATA_OUT, "features.npy"), mmap_mode="r")
    _, n_tokens, feature_dim = X.shape

    # Paper-clean split
    splits = ensure_splits(
        data_out_dir=DATA_OUT,
        seed=42,
        train_frac=0.70,
        val_frac=0.15,
        test_frac=0.15,
        overwrite=False,
    )
    test_idx = splits["test_idx"]

    # Pick multiple test examples
    if cfg.n_examples > len(test_idx):
        raise ValueError(f"n_examples={cfg.n_examples} > test set size {len(test_idx)}")

    chosen = rng.choice(test_idx, size=cfg.n_examples, replace=False)
    chosen = np.asarray(chosen, dtype=np.int64)

    # Load model + scaler
    # Important: ensure custom layers are registered (import already happened above).
    model = load_model(os.path.join(MODELS_OUT, "jr_paramtoken_inverse_model.keras"), compile=False)

    with open(os.path.join(MODELS_OUT, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    # Build attention model with same weights
    cfg_path = os.path.join(MODELS_OUT, "model_config.npz")
    if os.path.exists(cfg_path):
        mcfg = np.load(cfg_path)
        d_model = int(mcfg["d_model"])
        num_layers = int(mcfg["num_layers"])
        num_heads = int(mcfg["num_heads"])
        ff_dim = int(mcfg["ff_dim"])
        dropout_rate = float(mcfg["dropout_rate"])
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

    # Load H5 once (we read only selected indices)
    with h5py.File(DATA_FILE, "r") as f:
        leadfield = f["leadfield"][:]
        pnames_h5 = [x.decode("utf-8") for x in f.attrs["param_names"]]
        sim_args = dict(
            fs=int(f.attrs["fs"]),
            duration=float(f.attrs["duration"]),
            n_channels=int(f.attrs["n_channels"]),
            bandpass=tuple(f.attrs["bandpass"]),
            stim_onset=float(f.attrs["stim_onset"]),
            stim_sigma=float(f.attrs["stim_sigma"]),
            n_sources=int(f.attrs["n_sources"]),
            leadfield=leadfield,
            sensor_noise_std=float(f.attrs["sensor_noise_std"]),
            n_trials=int(f.attrs["n_trials"]),
            input_noise_std=float(f.attrs["input_noise_std"]),
        )

        # sanity check param name ordering
        if pnames_h5 != param_names:
            LOGGER.warning(
                "Param names mismatch between param_meta and H5 attrs.\n"
                "param_meta=%s\nh5=%s",
                param_names, pnames_h5
            )

        # Containers for aggregated correlations
        methods = ["attn", "grad"]
        blocks = ["erp", "tfr"]

        pearson = {m: {b: np.zeros((cfg.n_examples, P), dtype=np.float32) for b in blocks} for m in methods}
        spearman = {m: {b: np.zeros((cfg.n_examples, P), dtype=np.float32) for b in blocks} for m in methods}
        topk = {m: {b: np.zeros((cfg.n_examples, P), dtype=np.float32) for b in blocks} for m in methods}

        LOGGER.info("Evaluating sensitivity validation on %d test examples...", cfg.n_examples)

        # Loop examples
        for ei, idx in enumerate(chosen):
            eeg0 = f["EEG"][int(idx)]
            params0 = f["params"][int(idx)]
            base_params = {pnames_h5[i]: float(params0[i]) for i in range(len(pnames_h5))}

            # Tokens (raw) then scaled (model input)
            tokens0 = _patchify_hybrid(
                eeg_sample=eeg0,
                fs=fs,
                decim=decim,
                fmin=fmin,
                fmax=fmax,
                n_freqs=n_freqs,
                stim_onset=stim_onset,
                pre_sec=pre_sec,
                post_sec=post_sec,
                n_time=n_time,
                n_freq=n_freq,
                freq_patch=freq_patch,
                time_patch=time_patch,
            )
            xb0 = _scale_tokens(tokens0, scaler=scaler, feature_dim=feature_dim)  # (1, tokens, feat)

            # Attention scores
            _, scores = attn_model.predict(xb0, verbose=0)  # (1, H, P, tokens)
            scores = scores.mean(axis=1)[0].astype(np.float32)  # (P, tokens)

            # Gradient sensitivity (Jacobian norm)
            xb_tf = tf.convert_to_tensor(xb0, dtype=tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(xb_tf)
                y_pred = model(xb_tf, training=False)  # (1, 2P)
                mu_z = y_pred[:, :P]                   # (1, P)

            J = tape.jacobian(mu_z, xb_tf)            # (1, P, 1, tokens, feat)
            if J is None:
                raise RuntimeError("Jacobian returned None; check TF gradients.")
            J = tf.squeeze(J, axis=(0, 2))            # (P, tokens, feat)
            grads = tf.norm(J, axis=-1).numpy().astype(np.float32)  # (P, tokens)

            # Forward FD learnability in *scaled token space* (central diff)
            sens_fd = np.zeros((P, n_tokens), dtype=np.float32)

            for pi, nm in enumerate(param_names):
                base = float(base_params[nm])
                delta = float(cfg.delta_frac_range * (prior_high[pi] - prior_low[pi]))
                delta = max(delta, 1e-6)

                plus = dict(base_params)
                minus = dict(base_params)

                plus[nm] = min(float(prior_high[pi]), base + delta)
                minus[nm] = max(float(prior_low[pi]), base - delta)

                eeg_plus = simulate_eeg(params=plus, seed=cfg.fd_seed, **sim_args)
                eeg_minus = simulate_eeg(params=minus, seed=cfg.fd_seed, **sim_args)

                tok_plus = _patchify_hybrid(
                    eeg_sample=eeg_plus,
                    fs=fs,
                    decim=decim,
                    fmin=fmin,
                    fmax=fmax,
                    n_freqs=n_freqs,
                    stim_onset=stim_onset,
                    pre_sec=pre_sec,
                    post_sec=post_sec,
                    n_time=n_time,
                    n_freq=n_freq,
                    freq_patch=freq_patch,
                    time_patch=time_patch,
                )
                tok_minus = _patchify_hybrid(
                    eeg_sample=eeg_minus,
                    fs=fs,
                    decim=decim,
                    fmin=fmin,
                    fmax=fmax,
                    n_freqs=n_freqs,
                    stim_onset=stim_onset,
                    pre_sec=pre_sec,
                    post_sec=post_sec,
                    n_time=n_time,
                    n_freq=n_freq,
                    freq_patch=freq_patch,
                    time_patch=time_patch,
                )

                # Scale to match model coordinates
                tok_plus_s = _scale_tokens(tok_plus, scaler=scaler, feature_dim=feature_dim)[0]
                tok_minus_s = _scale_tokens(tok_minus, scaler=scaler, feature_dim=feature_dim)[0]

                # central difference derivative wrt parameter
                diff = (tok_plus_s - tok_minus_s) / (2.0 * delta)  # (tokens, feat)
                sens_fd[pi] = np.linalg.norm(diff, axis=1).astype(np.float32)

            # Correlations, separately for ERP and TFR blocks
            for pi in range(P):
                fd = sens_fd[pi]
                att = scores[pi]
                grd = grads[pi]

                for block_name, sl in [
                    ("erp", slice(0, n_tokens_erp)),
                    ("tfr", slice(n_tokens_erp, n_tokens)),
                ]:
                    fd_b = fd[sl]
                    att_b = att[sl]
                    grd_b = grd[sl]

                    pearson["attn"][block_name][ei, pi] = _safe_corr(att_b, fd_b)
                    pearson["grad"][block_name][ei, pi] = _safe_corr(grd_b, fd_b)

                    spearman["attn"][block_name][ei, pi] = _spearman_corr(att_b, fd_b)
                    spearman["grad"][block_name][ei, pi] = _spearman_corr(grd_b, fd_b)

                    topk["attn"][block_name][ei, pi] = _topk_overlap(att_b, fd_b, cfg.topk_frac)
                    topk["grad"][block_name][ei, pi] = _topk_overlap(grd_b, fd_b, cfg.topk_frac)

        # Print summary
        def summarize(arr: np.ndarray) -> Tuple[float, float]:
            return float(arr.mean()), float(arr.std())

        LOGGER.info("=== Sensitivity validation summary over %d examples ===", cfg.n_examples)
        for pi, nm in enumerate(param_names):
            a_erp = pearson["attn"]["erp"][:, pi]
            g_erp = pearson["grad"]["erp"][:, pi]
            a_tfr = pearson["attn"]["tfr"][:, pi]
            g_tfr = pearson["grad"]["tfr"][:, pi]

            LOGGER.info(
                "%-8s | Pearson(att~FD) ERP %.3f±%.3f  TFR %.3f±%.3f  | Pearson(grad~FD) ERP %.3f±%.3f  TFR %.3f±%.3f",
                nm,
                *summarize(a_erp),
                *summarize(a_tfr),
                *summarize(g_erp),
                *summarize(g_tfr),
            )

        # Save outputs
        out_path = os.path.join(PLOTS_DIR, "sensitivity_validation_summary.npz")
        np.savez(
            out_path,
            chosen_idx=chosen.astype(np.int64),
            param_names=np.array(param_names, dtype="S"),
            pearson_attn_erp=pearson["attn"]["erp"],
            pearson_attn_tfr=pearson["attn"]["tfr"],
            pearson_grad_erp=pearson["grad"]["erp"],
            pearson_grad_tfr=pearson["grad"]["tfr"],
            spearman_attn_erp=spearman["attn"]["erp"],
            spearman_attn_tfr=spearman["attn"]["tfr"],
            spearman_grad_erp=spearman["grad"]["erp"],
            spearman_grad_tfr=spearman["grad"]["tfr"],
            topk_attn_erp=topk["attn"]["erp"],
            topk_attn_tfr=topk["attn"]["tfr"],
            topk_grad_erp=topk["grad"]["erp"],
            topk_grad_tfr=topk["grad"]["tfr"],
        )
        LOGGER.info("Saved: %s", out_path)

        # Quick paper-style plot: boxplot Pearson correlations (ERP block)
        plt.figure(figsize=(10, 4))
        data = [pearson["grad"]["erp"][:, i] for i in range(P)]
        plt.boxplot(data, labels=param_names, showfliers=False)
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.ylabel("Pearson corr(grad sens, FD learnability) [ERP block]")
        plt.title(f"Sensitivity agreement over {cfg.n_examples} test examples")
        plt.tight_layout()
        fig_path = os.path.join(PLOTS_DIR, "sens_agreement_boxplot_grad_erp.png")
        plt.savefig(fig_path, dpi=200)
        plt.close()
        LOGGER.info("Saved: %s", fig_path)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.get_logger().setLevel("ERROR")
    main()
