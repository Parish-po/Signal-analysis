import scipy.io as sio
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# -----------------------------------------------------
# 🧩 1. Define the EEG filtering functions
# -----------------------------------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def bandpass_filter_eeg(eeg_matrix, fs=128, lowcut=1.0, highcut=50.0, order=4):
    eeg_matrix = np.array(eeg_matrix, dtype=np.float32)
    if eeg_matrix.shape[0] < eeg_matrix.shape[1]:
        eeg_matrix = eeg_matrix.T  # ensure (samples, channels)
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    filtered = filtfilt(b, a, eeg_matrix, axis=0)
    return filtered


# -----------------------------------------------------
# ⚙️ 2. Load and preprocess DREAMER dataset
# -----------------------------------------------------
def load_dreamer_data(mat_path):
    """Load DREAMER.mat and extract EEG, Valence, Arousal per trial."""
    data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    dreamer = data["DREAMER"]

    # handle versions with or without .Data
    subjects = getattr(dreamer, "Data", dreamer)
    rows = []

    for subj_id, subj in enumerate(subjects, start=1):
        eeg_obj = getattr(subj, "EEG", None)
        if eeg_obj is None:
            continue

        # EEG has 'stimuli' and 'baseline' lists
        for vid_id, (stim, base) in enumerate(zip(eeg_obj.stimuli, eeg_obj.baseline), start=1):
            stim = np.array(stim)
            base = np.array(base)

            # filter both
            stim_filt = bandpass_filter_eeg(stim)
            base_filt = bandpass_filter_eeg(base)

            val = np.atleast_1d(getattr(subj, "ScoreValence", [np.nan]))[vid_id-1]
            aro = np.atleast_1d(getattr(subj, "ScoreArousal", [np.nan]))[vid_id-1]

            rows.append({
                "subject_id": subj_id,
                "video_id": vid_id,
                "valence": float(val),
                "arousal": float(aro),
                "baseline": base_filt,
                "stimuli": stim_filt
            })

    df = pd.DataFrame(rows)
    df.to_pickle("../data/processed/DREAMER_filtered.pkl")  # save full objects safely
    print(f"✅ Loaded and filtered {len(df)} trials from {len(subjects)} subjects.")
    return df
