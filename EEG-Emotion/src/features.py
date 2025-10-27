"""features.py — Compute PSD and bandpowers."""
from typing import Dict, Tuple
import numpy as np
from scipy.signal import welch

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

def psd_welch(trial: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    # trial shape: (channels, samples)
    f, pxx = welch(trial, fs=fs, nperseg=min(256, trial.shape[-1]))
    return f, pxx  # pxx shape: (channels, freqs)

def bandpower(f: np.ndarray, pxx: np.ndarray, lo: float, hi: float) -> np.ndarray:
    idx = (f >= lo) & (f < hi)
    return np.trapz(pxx[..., idx], f[idx], axis=-1)

def extract_bandpowers(trial: np.ndarray, fs: float) -> Dict[str, np.ndarray]:
    f, pxx = psd_welch(trial, fs)
    feats = {}
    total = bandpower(f, pxx, 1, 45) + 1e-12
    for name, (lo, hi) in BANDS.items():
        bp = bandpower(f, pxx, lo, hi)
        feats[f"{name}_abs"] = bp
        feats[f"{name}_rel"] = bp / total
    return feats
