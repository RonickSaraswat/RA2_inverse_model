# features/feature_extraction.py
import numpy as np
from scipy.signal import welch, coherence
from scipy.linalg import lstsq

def bandpower(sig, fs=250, bands=None, nperseg=256):
    """
    sig: (n_channels, n_samples)
    bands: list of (low, high) tuples
    returns: array (n_channels * len(bands))
    """
    if bands is None:
        bands = [(1,4),(4,8),(8,13),(13,30),(30,45)]
    chs = []
    for ch in range(sig.shape[0]):
        f, pxx = welch(sig[ch], fs=fs, nperseg=min(nperseg, sig.shape[1]))
        ch_band = []
        for low, high in bands:
            idx = np.logical_and(f >= low, f <= high)
            ch_band.append(np.trapz(pxx[idx], f[idx]))
        chs.append(ch_band)
    return np.array(chs).reshape(-1)  # flatten

def sample_entropy(x, m=2, r=None):
    """
    sample entropy implementation.
    x: 1D array
    m: embedding dimension
    r: tolerance (if None use 0.2*std)
    """
    x = np.asarray(x)
    N = len(x)
    if r is None:
        r = 0.2 * np.std(x)
    def _phi(m):
        count = 0
        for i in range(N - m):
            xi = x[i:i+m]
            for j in range(i+1, N - m + 1):
                xj = x[j:j+m]
                if np.max(np.abs(xi - xj)) <= r:
                    count += 1
        return count
    try:
        return -np.log(_phi(m+1) / (_phi(m) + 1e-12) + 1e-12)
    except Exception:
        return np.nan

def connectivity_matrix(sig):
    """
    Pearson correlation-based connectivity flattened upper triangle.
    sig: (n_channels, n_samples)
    returns flattened vector length n_channels*(n_channels-1)/2
    """
    C = np.corrcoef(sig)
    # take upper triangle excluding diag
    triu_idx = np.triu_indices_from(C, k=1)
    return C[triu_idx]

def ar_coeffs(sig, order=5):
    """
    Fit AR model per channel via Yule-Walker (approx via least squares)
    returns flattened coefficients (n_channels * order)
    """
    n_ch, n_samples = sig.shape
    coeffs = []
    for ch in range(n_ch):
        x = sig[ch]
        # construct lagged matrix
        if n_samples <= order:
            coeffs.extend([0.0]*order)
            continue
        X = np.column_stack([x[order - i - 1: n_samples - i -1] for i in range(order)])
        y = x[order:]
        # least squares
        try:
            a, *_ = lstsq(X, y)
            # if lstsq returns col vector, flatten
            coeffs.extend(a.flatten().tolist())
        except Exception:
            coeffs.extend([0.0]*order)
    return np.array(coeffs)

def extract_features(eeg, fs=250, bands=None, ar_order=5):
    """
    eeg: (n_channels, n_samples)
    returns 1D feature vector
    """
    feats = []
    # per-channel mean/std
    feats.extend(eeg.mean(axis=1).tolist())
    feats.extend(eeg.std(axis=1).tolist())
    # bandpower per channel
    feats.extend(bandpower(eeg, fs=fs, bands=bands).tolist())
    # sample entropy per channel
    for ch in range(eeg.shape[0]):
        feats.append(sample_entropy(eeg[ch]))
    # AR coefficients
    feats.extend(ar_coeffs(eeg, order=ar_order).tolist())
    # connectivity (pearson)
    feats.extend(connectivity_matrix(eeg).tolist())
    return np.array(feats, dtype=np.float32)
