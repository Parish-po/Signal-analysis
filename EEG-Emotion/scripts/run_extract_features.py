#!/usr/bin/env python
"""Extract features from DREAMER EEG and save CSV."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.io_utils import find_mat_files, load_dreamer_mat, extract_trials
from src.preprocess import preprocess_eeg
from src.features import extract_bandpowers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--out_dir', type=Path, required=True)
    ap.add_argument('--fs', type=float, default=128.0, help='Sampling rate (Hz)')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mat_path in find_mat_files(args.data_dir):
        mat = load_dreamer_mat(mat_path)
        eeg, labels = extract_trials(mat)
        for i in range(eeg.shape[0]):
            trial = preprocess_eeg(eeg[i], fs=args.fs)
            feats = extract_bandpowers(trial, fs=args.fs)
            row = {'file': mat_path.name, 'trial': i}
            row.update({k: feats[k].mean() for k in feats})  # avg over channels
            row.update({'valence': labels['valence'][i], 'arousal': labels['arousal'][i]})
            rows.append(row)
    df = pd.DataFrame(rows)
    out_csv = args.out_dir / 'features.csv'
    df.to_csv(out_csv, index=False)
    print(f'Wrote {out_csv}')

if __name__ == '__main__':
    main()
