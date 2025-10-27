# EEG-Emotion — One Pager

**Goal:** Test whether EEG spectral features (bandpowers) differ across affective states (valence/arousal) in the DREAMER dataset.

**Data:** DREAMER (.mat), subjects × trials × channels.

**Pipeline:** I/O → preprocessing (filter, baseline) → PSD/bandpowers → stats (t-tests, effect sizes) → optional ML (classify high/low valence) → visualization.

**Key Outputs:** 
- `results/features/*.csv`: per-trial features
- `results/tables/*.csv`: stats & model metrics
- `results/figures/*.png`: plots
