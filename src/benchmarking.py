import pandas as pd
from pycaret.classification import setup, compare_models

def run_benchmarking(df):
    """
    Sets up the experiment with 50-feature selection to eliminate noise.
    """
    experiment = setup(
        data=df,
        target='class',
        fold_strategy='groupkfold',
        fold_groups='id',
        ignore_features=['id'],
        feature_selection=True,
        n_features_to_select=50, 
        feature_selection_estimator='rf',
        session_id=42,
        verbose=True
    )
    return experiment

def compare_models_clinical():
    """
    Returns the Top 3 models that natively support soft-voting probabilities.
    Excludes 'ridge', 'svm', and 'dummy' to prevent blending metric crashes.
    """
    top_3_models = compare_models(
        sort='F1', 
        n_select=3, 
        exclude=['lightgbm', 'dummy', 'ridge', 'svm'] # <-- EXCLUDED RIDGE AND SVM HERE
    )
    return top_3_models