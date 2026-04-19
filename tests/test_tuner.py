import pytest
import pandas as pd
import numpy as np
from src.benchmarking import run_benchmarking
from src.tuner import fine_tune_recall
from pycaret.classification import create_model, pull


@pytest.fixture
def tuning_data():
    """Generates enough synthetic data to survive a 5-fold cross-validation."""
    np.random.seed(42)
    # 20 patients, 5 records each = 100 rows
    names = [f"P{i}" for i in range(1, 21)] * 5
    return pd.DataFrame({
        'name': names,
        'Jitter(%)': np.random.uniform(0.001, 0.05, 100),
        'Shimmer': np.random.uniform(0.01, 0.1, 100),
        'status': np.random.choice([0, 1], 100)
    })


def test_tune_recall_improvement(tuning_data):
    """
    Verifies that the Optuna tuner (with choose_better=True)
    never degrades the recall of the baseline model.
    """
    run_benchmarking(tuning_data)

    # Train a simple, fast baseline model for the test
    baseline_model = create_model('lr', verbose=False)

    # Extract baseline recall from PyCaret's metric grid
    metrics = pull()
    baseline_recall = metrics.loc['Mean', 'Recall']

    # Run our tuner
    tuned_model = fine_tune_recall(baseline_model)

    # Extract tuned recall
    tuned_metrics = pull()
    tuned_recall = tuned_metrics.loc['Mean', 'Recall']

    assert tuned_recall >= baseline_recall, "CRITICAL FAILURE: Tuning degraded Recall!"