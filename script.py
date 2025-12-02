import os

def create_dreamer_project_structure(base_dir="DREAMER-EmotionAlignment"):
    """
    Creates the full folder + subfolder + template file structure
    for the DREAMER EEG project (Physiological vs. Self-Perceived Emotion).
    """

    # Define folders
    folders = [
        f"{base_dir}/data/raw",
        f"{base_dir}/data/processed",
        f"{base_dir}/notebooks",
        f"{base_dir}/src",
        f"{base_dir}/outputs/plots",
        f"{base_dir}/outputs/tables"
    ]

    # Create folders
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    # Create placeholder files
    files = {
        f"{base_dir}/README.md": "# DREAMER-EmotionAlignment\n\n"
                                 "Analyzing how EEG signals reflect self-perceived emotions "
                                 "using the DREAMER dataset.\n\n"
                                 "## Structure\n"
                                 "- data/: raw and processed EEG data\n"
                                 "- notebooks/: analysis notebooks\n"
                                 "- src/: Python modules for preprocessing and analysis\n"
                                 "- outputs/: plots and reports\n",
        f"{base_dir}/data/.gitignore": "# Ignore large files\n*.mat\n*.npy\n*.csv\n",
        f"{base_dir}/notebooks/01_explore_dataset.ipynb": "",
        f"{base_dir}/notebooks/02_feature_extraction.ipynb": "",
        f"{base_dir}/notebooks/03_analysis_correlations.ipynb": "",
        f"{base_dir}/notebooks/04_visualization_and_report.ipynb": "",
        f"{base_dir}/src/preprocessing.py": '"""\nEEG preprocessing functions\n"""\n',
        f"{base_dir}/src/feature_extraction.py": '"""\nFeature extraction (band power, asymmetry, etc.)\n"""\n',
        f"{base_dir}/src/analysis.py": '"""\nCorrelation and regression analysis functions\n"""\n',
    }

    for filepath, content in files.items():
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    print(f"✅ Project structure created successfully at: {os.path.abspath(base_dir)}")

# Run the function
if __name__ == "__main__":
    create_dreamer_project_structure()
