# baselines/run_sbi_baselines.py
from __future__ import annotations

import argparse
import os
import sys
import time
import logging
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

DATA_OUT = os.path.join(BASE_DIR, "data_out")

from data.splits import ensure_splits  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
LOGGER = logging.getLogger("sbi_baselines")


def _import_sbi_or_die():
    try:
        import torch
        import sbi
        from sbi.inference import SNPE, SNLE, SNRE
        from sbi.utils import BoxUniform
        return torch, sbi, SNPE, SNLE, SNRE, BoxUniform
    except Exception as e:
        raise SystemExit(
            "Missing sbi/torch. Install in the active environment:\n"
            "  python -m pip install torch sbi\n"
        ) from e


def _try_get_nn_builders():
    """
    Newer sbi provides posterior_nn/likelihood_nn/classifier_nn in sbi.neural_nets.
    Some older versions have them in sbi.utils.get_nn_models.
    """
    try:
        from sbi.neural_nets import posterior_nn, likelihood_nn, classifier_nn
        return posterior_nn, likelihood_nn, classifier_nn
    except Exception:
        from sbi.utils.get_nn_models import posterior_nn, likelihood_nn, classifier_nn
        return posterior_nn, likelihood_nn, classifier_nn


def _mlp_embedding(torch, x_dim: int, hidden: int = 256):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(x_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
    )


def _uniform_prior_std(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return (high - low) / np.sqrt(12.0)


def main():
    torch, sbi, SNPE, SNLE, SNRE, BoxUniform = _import_sbi_or_die()
    posterior_nn, likelihood_nn, classifier_nn = _try_get_nn_builders()

    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["snpe", "snle", "snre"], default="snpe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--density", choices=["maf", "nsf"], default="maf")
    ap.add_argument("--outdir", type=str, default=os.path.join(BASE_DIR, "baselines_out"))
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--num-post-samples", type=int, default=200)
    ap.add_argument("--max-test", type=int, default=-1, help="Use -1 for full test set.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    LOGGER.info("Using python=%s", sys.executable)
    LOGGER.info("Using torch=%s, sbi=%s", getattr(torch, "__version__", "unknown"), getattr(sbi, "__version__", "unknown"))

    # Load data
    X = np.load(os.path.join(DATA_OUT, "features.npy"), mmap_mode="r")  # (N, tokens, C)
    y = np.load(os.path.join(DATA_OUT, "params.npy"), mmap_mode="r")    # (N, P)

    bounds_path = os.path.join(BASE_DIR, "models_out", "param_bounds.npz")
    if not os.path.exists(bounds_path):
        raise FileNotFoundError(
            f"Missing {bounds_path}. Create it via:\n"
            f"  cp models_out/main_seed0/param_bounds.npz models_out/param_bounds.npz"
        )
    bounds = np.load(bounds_path)
    low = bounds["prior_low"].astype(np.float32)
    high = bounds["prior_high"].astype(np.float32)

    splits = ensure_splits(DATA_OUT, seed=args.split_seed, train_frac=0.70, val_frac=0.15, test_frac=0.15, overwrite=False)
    train_idx, test_idx = splits["train_idx"], splits["test_idx"]

    if args.max_test is not None and args.max_test > 0:
        test_idx = test_idx[: args.max_test]

    # Flatten tokens -> x vector
    X_train = np.asarray(X[train_idx], dtype=np.float32).reshape(len(train_idx), -1)
    X_test = np.asarray(X[test_idx], dtype=np.float32).reshape(len(test_idx), -1)
    y_train = np.asarray(y[train_idx], dtype=np.float32)
    y_test = np.asarray(y[test_idx], dtype=np.float32)

    # Standardize x using train stats
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-6
    X_train = (X_train - mu) / sd
    X_test = (X_test - mu) / sd

    x_dim = X_train.shape[1]
    P = y_train.shape[1]

    prior = BoxUniform(low=torch.as_tensor(low), high=torch.as_tensor(high))

    embedding_net = _mlp_embedding(torch, x_dim=x_dim, hidden=args.hidden)

    theta_train = torch.as_tensor(y_train, dtype=torch.float32)
    x_train = torch.as_tensor(X_train, dtype=torch.float32)

    # --- Build inference object (compatible with new/old sbi) ---
    if args.method == "snpe":
        # New API: pass a build function from posterior_nn
        try:
            inference = SNPE(prior=prior, density_estimator=args.density, embedding_net=embedding_net)  # old API
        except TypeError:
            build_fun = posterior_nn(model=args.density, embedding_net=embedding_net)
            inference = SNPE(prior=prior, density_estimator=build_fun)

    elif args.method == "snle":
        try:
            inference = SNLE(prior=prior, density_estimator=args.density, embedding_net=embedding_net)  # old API
        except TypeError:
            build_fun = likelihood_nn(model=args.density, embedding_net=embedding_net)
            inference = SNLE(prior=prior, density_estimator=build_fun)

    else:  # snre
        try:
            inference = SNRE(prior=prior, classifier="mlp", embedding_net=embedding_net)  # old API
        except TypeError:
            build_fun = classifier_nn(model="mlp", embedding_net=embedding_net)
            inference = SNRE(prior=prior, classifier=build_fun)

    LOGGER.info("Training %s baseline on %d sims...", args.method, len(train_idx))
    t0 = time.time()
    inference = inference.append_simulations(theta_train, x_train)
    density_estimator = inference.train(validation_fraction=0.1)
    t1 = time.time()
    LOGGER.info("Training done in %.2f min", (t1 - t0) / 60.0)

    posterior = inference.build_posterior(density_estimator)

    # Evaluate on test
    x0 = torch.as_tensor(X_test, dtype=torch.float32)
    theta_true = torch.as_tensor(y_test, dtype=torch.float32)

    # NLL(theta)
    with torch.no_grad():
        logp = []
        for i in range(x0.shape[0]):
            lp = posterior.log_prob(theta_true[i], x=x0[i])
            logp.append(float(lp.cpu().numpy()))
        logp = np.array(logp, dtype=np.float32)
    nll_theta = float((-logp).mean())

    # Posterior samples
    S = int(args.num_post_samples)

    # NOTE: SNRE posterior sampling can be slow (often MCMC). Consider --max-test 200 for SNRE.
    samps = []
    with torch.no_grad():
        for i in range(x0.shape[0]):
            s = posterior.sample((S,), x=x0[i]).cpu().numpy().astype(np.float32)  # (S,P)
            samps.append(s)
    theta_samps = np.stack(samps, axis=1)  # (S,N,P)

    theta_mean = theta_samps.mean(axis=0)
    lo = np.quantile(theta_samps, 0.05, axis=0)
    hi = np.quantile(theta_samps, 0.95, axis=0)
    coverage90 = ((y_test >= lo) & (y_test <= hi)).mean(axis=0).astype(np.float32)

    post_std = theta_samps.std(axis=0).mean(axis=0)
    contraction = (post_std / (_uniform_prior_std(low, high) + 1e-12)).astype(np.float32)

    sbc_ranks = np.sum(theta_samps < y_test[None, :, :], axis=0).astype(np.int32)  # (N,P) in [0,S]

    out_path = os.path.join(args.outdir, f"{args.method}_{args.density}_seed{args.seed}.npz")
    np.savez(
        out_path,
        method=args.method,
        density=args.density,
        seed=args.seed,
        split_seed=args.split_seed,
        y_test=y_test,
        theta_mean=theta_mean,
        theta_samps=theta_samps,        # enables SBC plotting from samples if desired
        sbc_ranks=sbc_ranks,            # enables SBC plotting without samples
        n_post_samples=S,
        nll_theta=nll_theta,
        coverage90=coverage90,
        contraction=contraction,
        prior_low=low,
        prior_high=high,
    )
    LOGGER.info("Saved baseline results: %s", out_path)
    print("nll_theta:", nll_theta)
    print("coverage90:", coverage90)
    print("contraction:", contraction)


if __name__ == "__main__":
    main()
