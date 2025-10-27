"""models.py — Optional ML classifier (e.g., high/low valence)."""
from typing import Tuple, Dict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

def binarize_scores(y: np.ndarray, threshold: float | None = None) -> np.ndarray:
    thr = threshold if threshold is not None else float(np.median(y))
    return (y >= thr).astype(int)

def fit_simple_classifier(X: np.ndarray, y_continuous: np.ndarray) -> Dict[str, float]:
    y = binarize_scores(y_continuous)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))])
    pipe.fit(Xtr, ytr)
    yhat = pipe.predict(Xte)
    return {
        "acc": float(accuracy_score(yte, yhat)),
        "f1": float(f1_score(yte, yhat))
    }
