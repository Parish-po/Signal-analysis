"""
Feature extraction (band power, asymmetry, etc.)
"""
def extract_variance_components(model):
    random_effects = model.random_effects
    vc = model.vcomp
    
    return {
        "var_subject": vc["subject_id"],
        "var_video": vc["video_id"],
        "var_residual": model.scale
    }
