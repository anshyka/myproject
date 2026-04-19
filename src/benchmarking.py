import pandas as pd
from pycaret.classification import setup, compare_models, get_config

def run_benchmarking(df: pd.DataFrame):
    """
    Initializes the PyCaret setup with strict clinical constraints.
    Enforces GroupKFold to prevent patient-level data leakage.
    """
    # Initialize the PyCaret environment
    # fold_strategy and fold_groups are the critical parameters here
    experiment = setup(
        data=df,
        target='status',
        fold_strategy='groupkfold',
        fold_groups='name',      # Tells PyCaret which column holds the patient IDs
        session_id=42,           # For reproducibility
        verbose=False            # Keeps terminal output clean during automated runs
    )
    return experiment

def compare_models_clinical():
    """
    Benchmarks specific clinical baseline models.
    Sorts the leaderboard prioritizing Recall to minimize False Negatives.
    """
    # rf = Random Forest, xgboost = Extreme Gradient Boosting
    # svm = Support Vector Machine, lr = Logistic Regression
    print("Benchmarking models... (This may take a moment)")
    best_model = compare_models(
        include=['rf', 'xgboost', 'svm', 'lr'],
        sort='Recall'
    )
    return best_model