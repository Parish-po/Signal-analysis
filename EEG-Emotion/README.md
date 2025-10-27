# EEG-Emotion

A compact pipeline for exploring emotion-related patterns in EEG using the **DREAMER** dataset.

## Structure
- `data/`: place raw `.mat` DREAMER files here (not committed).
- `src/`: core library code (I/O, preprocessing, features, stats, models, viz).
- `notebooks/`: exploratory notebooks.
- `results/`: generated features, figures, and tables.
- `reports/`: one-pager and (optional) slides.
- `scripts/`: CLI scripts to run steps headlessly.
- `environment.yml`: conda environment for reproducibility.

## Quickstart
```bash
conda env create -f environment.yml
conda activate eeg-emotion
# Explore
jupyter lab
# Or run headless
python scripts/run_extract_features.py --data_dir data --out_dir results/features
python scripts/run_stats.py --features_csv results/features/features.csv --out_dir results/tables
python scripts/run_train_model.py --features_csv results/features/features.csv --out_dir results/tables
```
