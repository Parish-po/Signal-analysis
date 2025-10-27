#!/usr/bin/env python
"""Compute group-difference stats and save tables."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from src.stats import ttest_group_diff

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features_csv', type=Path, required=True)
    ap.add_argument('--out_dir', type=Path, required=True)
    ap.add_argument('--target', choices=['valence','arousal'], default='valence')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.features_csv)

    thr = df[args.target].median()
    df['label'] = (df[args.target] >= thr).astype(int)

    metrics = []
    feat_cols = [c for c in df.columns if any(b in c for b in ['delta','theta','alpha','beta','gamma']) and c.endswith('_rel')]
    for c in feat_cols:
        x = df.loc[df['label']==0, c].values
        y = df.loc[df['label']==1, c].values
        res = ttest_group_diff(x, y)
        metrics.append({'feature': c, **res})

    out_csv = args.out_dir / 'ttest_results.csv'
    pd.DataFrame(metrics).to_csv(out_csv, index=False)
    print(f'Wrote {out_csv}')

if __name__ == '__main__':
    main()
