# models/param_transforms.py
import numpy as np


def theta_to_z(theta, low, high, eps=1e-6):
    """
    Map bounded theta in [low, high] to unbounded z via logit transform.
    theta: (..., P)
    low/high: (P,)
    """
    theta = np.asarray(theta, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    u = (theta - low) / (high - low + 1e-12)
    u = np.clip(u, eps, 1.0 - eps)
    z = np.log(u) - np.log(1.0 - u)
    return z.astype(np.float32)


def z_to_theta(z, low, high):
    z = np.asarray(z, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    u = 1.0 / (1.0 + np.exp(-z))
    theta = low + u * (high - low)
    return theta.astype(np.float32)


def sample_theta_from_gaussian_z(mu_z, logvar_z, low, high, n_samples=200, seed=0):
    """
    Draw samples from diagonal Gaussian in z-space, map to theta-space.
    mu_z/logvar_z: (N, P)
    returns: (S, N, P)
    """
    rng = np.random.default_rng(seed)
    mu_z = np.asarray(mu_z, dtype=np.float32)
    logvar_z = np.asarray(logvar_z, dtype=np.float32)
    std = np.exp(0.5 * np.clip(logvar_z, -10.0, 10.0)).astype(np.float32)

    S = int(n_samples)
    eps = rng.normal(size=(S,) + mu_z.shape).astype(np.float32)
    z = mu_z[None, :, :] + eps * std[None, :, :]
    return z_to_theta(z, low, high)
