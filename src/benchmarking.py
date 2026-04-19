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
    print("Benchmarking models... (This may take a moment)")
    best_model = compare_models(
        include=['rf', 'gbc', 'svm', 'lr'],
        sort='F1'  # Changed from 'Recall'
    )
    return best_model