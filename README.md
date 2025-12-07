# DREAMER Emotion Alignment: EEG-Based Subject Identification

## 📋 Overview

This project investigates **trait stability in EEG signals** during emotional video stimuli using the DREAMER dataset. The research demonstrates that while traditional linear approaches (subtraction-based reactivity) fail to capture stable individual differences, **non-linear machine learning models can successfully identify subjects** from their EEG signatures with significantly above-chance accuracy.

### Key Findings

- **Subject Identification Accuracy**: Random Forest classifier achieved **significantly above chance-level** accuracy in identifying subjects from EEG features
- **Trait vs. State Distinction**: The study reveals that emotional EEG responses are primarily **state-dependent** rather than trait-like
- **Methodological Insight**: Traditional reactivity measures (Stimulus - Baseline) fail because they attempt to remove the very trait variance needed for identification

---

## 🎯 Research Motivation

**Central Question**: Can we identify individuals from their EEG responses during emotional stimulation?

**Why This Matters**:
- Understanding individual differences in emotional processing
- Developing personalized emotion recognition systems
- Distinguishing trait-like (stable) vs. state-like (variable) EEG characteristics

---

## 📊 Dataset: DREAMER

The **DREAMER** (Database for Emotion Analysis using Multimodal signals) dataset contains:

- **Participants**: 23 subjects
- **Stimuli**: 18 film clips designed to elicit different emotions
- **Recordings**: 
  - **Baseline**: Pre-stimulus resting EEG (3-4 seconds)
  - **Stimulus**: EEG during video viewing (~60-90 seconds)
- **Channels**: 14 EEG channels (Emotiv EPOC headset)
  - Frontal: AF3, AF4, F3, F4, F7, F8
  - Central: FC5, FC6
  - Temporal: T7, T8
  - Parietal: P7, P8
  - Occipital: O1, O2
- **Sampling Rate**: 128 Hz
- **Self-reported Labels**: Valence and Arousal ratings (1-5 scale)

---

## 🔬 Methodology

### 1. Data Preprocessing

**Signal Filtering** ([preprocessing.py](src/preprocessing.py)):
- Butterworth bandpass filter (1-50 Hz)
- Removes DC offset and high-frequency noise
- Applied to both baseline and stimulus periods

```python
# Filter specifications
- Order: 4
- Low cutoff: 1 Hz (removes DC drift)
- High cutoff: 50 Hz (removes line noise and EMG artifacts)
```

### 2. Feature Extraction

Two complementary approaches were implemented:

#### A. Concatenation Features ([02_feature_extraction.ipynb](notebooks/02_feature_extraction.ipynb))

Features extracted **separately** for baseline and stimulus periods:

**Time-Domain Features**:
- **RMS (Root Mean Square)**: Overall signal power per channel
  - Captures signal intensity

**Frequency-Domain Features** (Welch's method):
- **Theta Band (4-8 Hz)**: Attention, working memory
- **Alpha Band (8-13 Hz)**: Relaxation vs. arousal
- **Beta Band (13-30 Hz)**: Alertness, mental effort

**Asymmetry Features**:
- **FAA (Frontal Alpha Asymmetry)**: 
  ```
  FAA = ln(Right_Alpha) - ln(Left_Alpha)
  ```
  - Calculated using frontal electrode pairs: AF3/AF4, F7/F8, F3/F4
  - Right frontal: withdrawal emotions (fear, sadness)
  - Left frontal: approach emotions (happiness, excitement)

**Total Features**: 
- Base_RMS (14 channels) + Stim_RMS (14 channels)
- Base_Theta/Alpha/Beta (14 channels × 3 bands) + Stim_Theta/Alpha/Beta (14 channels × 3 bands)
- Base_FAA + Stim_FAA
- **~130 features total**

#### B. Reactivity Features ([02_feature_extraction_reactivity.ipynb](notebooks/02_feature_extraction_reactivity.ipynb))

Features computed as **difference scores** (Stimulus - Baseline):

```python
ΔRMS = RMS_stimulus - RMS_baseline
ΔTheta = Theta_stimulus - Theta_baseline
ΔAlpha = Alpha_stimulus - Alpha_baseline
ΔBeta = Beta_stimulus - Beta_baseline
ΔFAA = FAA_stimulus - FAA_baseline
ΔCorr_F3F4 = Corr(F3,F4)_stimulus - Corr(F3,F4)_baseline
ΔCorr_F7F8 = Corr(F7,F8)_stimulus - Corr(F7,F8)_baseline
```

**Rationale**: Traditional psychophysiology approach assumes reactivity captures individual response patterns.

---

### 3. Statistical Analysis

**Variance Decomposition** ([03_analysis_correlations.ipynb](notebooks/03_analysis_correlations.ipynb)):

Used **Mixed Linear Models** to decompose variance:

```python
Feature ~ 1 + (1|subject_id) + (1|video_id) + residual
```

**Components**:
- **Subject Variance**: Stable individual differences (trait)
- **Video Variance**: Stimulus-specific effects
- **Residual Variance**: Measurement error and state fluctuations

**ICC (Intraclass Correlation Coefficient)**:
```
ICC = Var(Subject) / [Var(Subject) + Var(Video) + Var(Residual)]
```

- ICC ≈ 1: High trait stability (good for identification)
- ICC ≈ 0: High state variability (poor for identification)

---

### 4. Machine Learning Classification

**Objective**: Identify subjects from EEG features

**Model**: Random Forest Classifier
- 100 trees
- 5-fold stratified cross-validation
- StandardScaler for feature normalization

**Evaluation**:
- **Accuracy**: Proportion of correctly identified subjects
- **Chance Level**: 1/23 ≈ 4.35% (for 23 subjects)

---

## 📈 Results

### Key Findings

#### 1. **Concatenation Approach Succeeds**
- **Model Accuracy**: Significantly above chance level
- **Why**: Preserves trait variance by keeping baseline and stimulus features separate
- **Interpretation**: Subject-specific EEG signatures are retained

#### 2. **Reactivity Approach Fails**
- **Model Accuracy**: Near chance level
- **Why**: Subtraction removes the trait variance critical for identification
- **ICC Analysis**: Reactivity features show very low ICC values

#### 3. **Variance Components** ([variance_components_concat.csv](outputs/models/variance_components_concat.csv))

| Feature Type | Subject Variance | Residual Variance | ICC |
|-------------|------------------|-------------------|-----|
| Base_Alpha_AF3 | Moderate | Low | **High** |
| Stim_Alpha_AF3 | Moderate | Moderate | **Moderate** |
| ΔAlpha_AF3 | Very Low | High | **Very Low** |

**Interpretation**: 
- Baseline features are most stable across videos
- Stimulus features show moderate stability
- Reactivity (Δ) features have minimal trait variance

---

## 🗂️ Project Structure

```
DREAMER-EmotionAlignment/
├── README.md                          # This file
├── data/
│   ├── raw/
│   │   └── DREAMER.mat               # Original MATLAB dataset
│   └── processed/
│       ├── DREAMER.csv               # Basic processed data
│       ├── eeg_features_concat.csv   # Concatenation features
│       └── eeg_features_reactivity.csv # Reactivity features
├── notebooks/
│   ├── 01_explore_dataset.ipynb      # Data exploration & visualization
│   ├── 02_feature_extraction.ipynb   # Concatenation feature extraction
│   ├── 02_feature_extraction_reactivity.ipynb # Reactivity features
│   ├── 03_analysis_correlations.ipynb # Statistical analysis & ML
│   └── 04_plot.ipynb                 # Results visualization
├── src/
│   ├── preprocessing.py              # Signal filtering & loading
│   └── feature_extraction.py         # Feature computation utilities
├── outputs/
│   ├── models/
│   │   ├── ml_comparison_concat.csv  # ML results (concatenation)
│   │   ├── ml_results_reactivity.csv # ML results (reactivity)
│   │   └── variance_components_concat.csv # ICC analysis
│   ├── plots/
│   │   ├── icc_plot.png              # Trait stability visualization
│   │   └── model_comparison.png      # ML performance comparison
│   └── paper/
│       └── Paper.pdf                 # Research paper
└── requirements.txt                  # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook/Lab

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd DREAMER-EmotionAlignment
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download DREAMER dataset**:
   - Place `DREAMER.mat` in `data/raw/`
   - Available from: [DREAMER Dataset Source]

---

## 📝 Usage

### Running the Analysis Pipeline

Execute notebooks in order:

```bash
jupyter notebook
```

1. **[01_explore_dataset.ipynb](notebooks/01_explore_dataset.ipynb)**: 
   - Load and visualize raw EEG data
   - Inspect power spectra
   - Understand dataset structure

2. **[02_feature_extraction.ipynb](notebooks/02_feature_extraction.ipynb)**:
   - Extract concatenation features
   - Generates: `eeg_features_concat.csv`

3. **[02_feature_extraction_reactivity.ipynb](notebooks/02_feature_extraction_reactivity.ipynb)**:
   - Extract reactivity features
   - Generates: `eeg_features_reactivity.csv`

4. **[03_analysis_correlations.ipynb](notebooks/03_analysis_correlations.ipynb)**:
   - Run statistical analyses
   - Train machine learning models
   - Calculate ICC and variance components

5. **[04_plot.ipynb](notebooks/04_plot.ipynb)**:
   - Generate publication-quality figures

---

## 📊 Key Visualizations

### 1. Trait Stability (ICC) Plot
![ICC Plot](outputs/plots/icc_plot.png)

Shows ICC values comparing:
- Baseline features (high stability)
- Stimulus features (moderate stability)
- Reactivity features (low stability)

### 2. Model Performance Comparison
![Model Comparison](outputs/plots/model_comparison.png)

Demonstrates:
- Concatenation model: Significantly above chance
- Reactivity model: Near chance level
- Red dashed line: Chance level baseline

---

## 🔑 Key Insights

### Why Subtraction Fails

**Mathematical Perspective**:
```
X_stim = T (trait) + S (state) + ε (noise)
X_base = T (trait) + ε (noise)

Reactivity = X_stim - X_base = S (state) + ε 
```

**Result**: Subtraction removes the trait (T) we need for identification!

### Why Concatenation Succeeds

**Representation**:
```
Features = [X_base, X_stim]
         = [T + ε₁, T + S + ε₂]
```

**Result**: Trait (T) is preserved in the baseline features, allowing identification.

---

## 🧪 Technical Details

### Signal Processing

- **Welch's Method**: For power spectral density estimation
  - `nperseg = 256` samples (2 seconds at 128 Hz)
  - Reduces spectral variance through averaging

### Machine Learning

- **Feature Scaling**: StandardScaler (zero mean, unit variance)
- **Cross-Validation**: 5-fold stratified (maintains class balance)
- **Random Forest Parameters**:
  - `n_estimators = 100`
  - `random_state = 42` (reproducibility)

### Statistical Testing

- **Mixed Linear Models**: `statsmodels.mixedlm`
- **Formula**: `Feature ~ 1 + (1|subject_id) + (1|video_id)`
- **Estimation**: REML (Restricted Maximum Likelihood)

---

## 📚 Dependencies

Main libraries:
- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **scipy**: Signal processing (filtering, FFT, Welch)
- **scikit-learn**: Machine learning (Random Forest, cross-validation)
- **statsmodels**: Mixed linear models, ICC calculation
- **matplotlib/seaborn**: Visualization

See [requirements.txt](requirements.txt) for complete list.

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional feature extraction methods
- Deep learning approaches
- Cross-dataset validation
- Real-time emotion recognition

---

## 📄 Citation

If you use this code or findings, please cite:

```bibtex
@article{dreamer_emotion_alignment,
  title={Trait Stability in EEG-Based Emotion Recognition: A Machine Learning Approach},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2024}
}
```

---

## 📧 Contact

For questions or collaboration:
- **Email**: [your.email@example.com]
- **Issues**: [GitHub Issues]

---

## 🙏 Acknowledgments

- **DREAMER Dataset**: Katsigiannis, S., & Ramzan, N. (2017). DREAMER: A Database for Emotion Recognition through EEG and ECG Signals from Wireless Low-cost Off-the-shelf Devices. *IEEE Journal of Biomedical and Health Informatics*.
- **Python Scientific Stack**: NumPy, SciPy, pandas, scikit-learn communities

---

## 📜 License

[Specify License - MIT, Apache 2.0, etc.]

---

## 🔬 Future Directions

- **Deep Learning**: CNN/RNN architectures for temporal patterns
- **Transfer Learning**: Pre-trained models from larger EEG datasets
- **Multimodal Fusion**: Combine EEG with ECG, facial expressions
- **Real-time Implementation**: Online emotion recognition system
- **Clinical Applications**: Mental health assessment, ADHD diagnostics

---

**Last Updated**: December 2024
