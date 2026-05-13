import pandas as pd
from pycaret.classification import setup, compare_models

def run_benchmarking(df):
    """
    Sets up the experiment with 50-feature selection to reduce noise.
    """
    experiment = setup(
        data=df,
        target='class',
        fold_strategy='groupkfold',
        fold_groups='id',
        ignore_features=['id'],
        feature_selection=True,
        n_features_to_select=50, # Reducing 750+ columns to 50 best
        feature_selection_estimator='rf',
        session_id=42,
        verbose=True
    )
    return experiment

def compare_models_clinical():
    """
    Returns the Top 3 real models sorted by F1-Score.
    """
    # Exclude dummy to prevent the model from just guessing the majority class
    top_3_models = compare_models(
        sort='F1', 
        n_select=3, 
        exclude=['lightgbm', 'dummy']
    )
    return top_3_models