#!/usr/bin/env python
"""Train a simple classifier and save metrics."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from src.models import fit_simple_classifier

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features_csv', type=Path, required=True)
    ap.add_argument('--out_dir', type=Path, required=True)
    ap.add_argument('--target', choices=['valence','arousal'], default='valence')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.features_csv)
    feat_cols = [c for c in df.columns if any(b in c for b in ['delta','theta','alpha','beta','gamma']) and c.endswith('_rel')]
    X = df[feat_cols].values
    y = df[args.target].values
    metrics = fit_simple_classifier(X, y)
    out = args.out_dir / 'model_metrics.txt'
    with open(out, 'w', encoding='utf-8') as f:
        for k,v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"Wrote {out}")

if __name__ == '__main__':
    main()
