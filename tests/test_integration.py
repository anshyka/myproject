import os
import joblib
import pytest
import pandas as pd
import numpy as np
from src.preprocessing import ParkinsonsDataLoader
from src.benchmarking import run_benchmarking, compare_models_clinical
from src.tuner import fine_tune_recall


@pytest.fixture
def integration_data():
    """Generates synthetic clinical data for pipeline integration testing."""
    np.random.seed(42)
    names = [f"P{i}" for i in range(1, 11)] * 5  # 10 patients, 5 records each

    df = pd.DataFrame({
        'name': names,
        'MDVP:Fo(Hz)': np.random.uniform(100, 200, 50),
        'MDVP:Jitter(%)': np.random.uniform(0.001, 0.02, 50),
        'status': np.random.choice([0, 1], 50)
    })

    # Save it temporarily for the dataloader to pick up
    os.makedirs('data/temp', exist_ok=True)
    df.to_csv('data/temp/dummy_data.csv', index=False)
    return 'data/temp/dummy_data.csv'


def test_full_pipeline_execution(integration_data):
    """Verifies that the entire pipeline runs and successfully exports a valid artifact."""

    # Run the core logic
    loader = ParkinsonsDataLoader()
    df = loader.load_data(integration_data)
    run_benchmarking(df)

    best_model = compare_models_clinical()
    tuned_model = fine_tune_recall(best_model)

    # Export
    export_path = 'artifacts/models/test_model.joblib'
    os.makedirs('artifacts/models', exist_ok=True)

    artifact = {
        'model_pipeline': tuned_model,
        'feature_names': ['MDVP:Fo(Hz)', 'MDVP:Jitter(%)']
    }
    joblib.dump(artifact, export_path)

    # Assertions
    assert os.path.exists(export_path), "CRITICAL FAILURE: joblib artifact was not created!"

    # Test if it can be loaded
    loaded_artifact = joblib.load(export_path)
    assert 'model_pipeline' in loaded_artifact, "Export missing model pipeline!"
    assert 'feature_names' in loaded_artifact, "Export missing feature names!"

    # Cleanup temp files
    os.remove(integration_data)
    os.remove(export_path)