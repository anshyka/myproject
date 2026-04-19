import pandas as pd
from pycaret.classification import setup, compare_models, get_config

def run_benchmarking(df: pd.DataFrame):
    """
    Initializes the PyCaret setup with strict clinical constraints.
    Enforces GroupKFold to prevent patient-level data leakage.
    """
    experiment = setup(
        data=df,
        target='status',
        fold_strategy='groupkfold',
        fold_groups='name',
        ignore_features=['name'],  # <--- THIS IS THE FIX
        session_id=42,
        verbose=False
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