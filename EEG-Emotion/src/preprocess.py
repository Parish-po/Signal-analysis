"""preprocess.py — Filtering, baseline, and standardization."""
from typing import Tuple
import numpy as np
from scipy.signal import butter, filtfilt

def bandpass(data: np.ndarray, fs: float, lo: float = 0.5, hi: float = 45.0, order: int = 4) -> np.ndarray:
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype='band')
    return filtfilt(b, a, data, axis=-1)

def baseline_correct(data: np.ndarray, baseline_idx: slice) -> np.ndarray:
    base = data[..., baseline_idx].mean(axis=-1, keepdims=True)
    return data - base

def preprocess_eeg(eeg: np.ndarray, fs: float, baseline_idx: slice | None = None) -> np.ndarray:
    x = bandpass(eeg, fs)
    if baseline_idx is not None:
        x = baseline_correct(x, baseline_idx)
    return x
