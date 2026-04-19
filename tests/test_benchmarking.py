import pytest
import pandas as pd
from src.benchmarking import run_benchmarking
from pycaret.classification import get_config


@pytest.fixture
def dummy_pycaret_data():
    """Provides a synthetic DataFrame with distinct patient groups."""
    return pd.DataFrame({
        'name': ['P1', 'P1', 'P2', 'P2', 'P3', 'P3', 'P4', 'P4'],
        'Jitter(%)': [0.01, 0.02, 0.015, 0.025, 0.011, 0.021, 0.005, 0.006],
        'Shimmer': [0.03, 0.04, 0.035, 0.045, 0.031, 0.041, 0.01, 0.015],
        'status': [1, 1, 0, 0, 1, 1, 0, 0]
    })


def test_pycaret_group_setup(dummy_pycaret_data):
    """
    Verifies that PyCaret's internal configuration correctly
    registers the 'name' column as the fold_groups parameter.
    """
    # Run the setup
    run_benchmarking(dummy_pycaret_data)

    # Retrieve PyCaret's internal configuration state (updated for PyCaret 3.x)
    configured_fold_groups = get_config('fold_groups_param')

    # Assert that PyCaret correctly extracted the 'name' column as a Series
    assert configured_fold_groups is not None, "CRITICAL FAILURE: fold_groups_param is None"
    assert configured_fold_groups.name == 'name', \
        f"CRITICAL FAILURE: PyCaret grouped by {configured_fold_groups.name} instead of 'name'"