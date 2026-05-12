import pandas as pd
from pycaret.classification import setup, compare_models, get_config

def run_benchmarking(df):
    experiment = setup(
        data=df,
        target='class',
        fold_strategy='groupkfold',
        fold_groups='id',
        ignore_features=['id'],
        feature_selection=True,
        n_features_to_select=0.2,
        feature_selection_estimator='rf',  # <--- THIS IS THE FIX
        session_id=42,
        verbose= True

    )
    return experiment


def compare_models_clinical():
    """
    Benchmarks all available baseline models in PyCaret.
    Sorts the leaderboard prioritizing F1-score for balanced clinical metrics.
    """
    print("Benchmarking available models... (This will take a bit longer)")

    # We exclude 'lightgbm' because it is structurally incompatible
    # with tiny datasets and spams the terminal with split warnings.
    best_model = compare_models(
        sort='F1',
        exclude=['lightgbm']  # <--- THIS IS THE FIX
    )

    return best_model