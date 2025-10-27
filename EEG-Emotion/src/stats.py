"""stats.py — t-tests and effect sizes."""
from typing import Dict, Tuple
import numpy as np
from scipy.stats import ttest_ind

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    s = (((nx-1)*vx + (ny-1)*vy) / (nx + ny - 2)) ** 0.5
    return (x.mean() - y.mean()) / (s + 1e-12)

def ttest_group_diff(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    t, p = ttest_ind(x, y, equal_var=False)
    return {"t": float(t), "p": float(p), "d": float(cohens_d(x, y))}
