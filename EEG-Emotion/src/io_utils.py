"""io_utils.py — Load DREAMER .mat files and extract trials."""
from typing import Dict, Any, List, Tuple
from pathlib import Path
import scipy.io as sio
import numpy as np

def load_dreamer_mat(path: Path) -> Dict[str, Any]:
    """Load a DREAMER .mat file and return its dict structure."""
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    return mat

def find_mat_files(data_dir: Path) -> List[Path]:
    return sorted([p for p in data_dir.glob('**/*.mat')])

def extract_trials(mat_dict: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return (eeg, labels) shaped as needed by downstream steps.
    Adapt this to match DREAMER's exact structure in your files.
    """
    # Placeholder: synth shape (trials, channels, samples)
    eeg = np.zeros((10, 32, 750))  # replace with real extraction
    labels = {
        "valence": np.random.randint(1, 10, size=(10,)),
        "arousal": np.random.randint(1, 10, size=(10,)),
    }
    return eeg, labels
