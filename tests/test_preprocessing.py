import pytest
import pandas as pd
from src.preprocessing import ParkinsonsDataLoader


@pytest.fixture
def dummy_clinical_data():
    """Provides a synthetic DataFrame for testing."""
    return pd.DataFrame({
        'name': ['phon_R01_S01_1', 'phon_R01_S01_2', 'phon_R01_S02_1'],
        'MDVP:Fo(Hz)': [119.992, 122.400, 116.682],
        'MDVP:Jitter(%)': [0.00784, 0.00968, 0.01050],
        'status': [1, 1, 0]
    })


def test_missing_target_column():
    """Test that a missing 'status' column raises an error."""
    df_missing_target = pd.DataFrame({'name': ['S01'], 'MDVP:Fo(Hz)': [120.0]})
    loader = ParkinsonsDataLoader()

    with pytest.raises(ValueError, match="Missing target column 'status'"):
        loader.validate_schema(df_missing_target)


def test_missing_group_column():
    """Test that a missing 'name' column raises an error."""
    df_missing_group = pd.DataFrame({'status': [1], 'MDVP:Fo(Hz)': [120.0]})
    loader = ParkinsonsDataLoader()

    with pytest.raises(ValueError, match="Missing group column 'name'"):
        loader.validate_schema(df_missing_group)


def test_feature_isolation(dummy_clinical_data):
    """
    CRITICAL: Ensure the 'name' column is extracted for grouping
    but entirely removed from the feature matrix X to prevent data leakage.
    """
    loader = ParkinsonsDataLoader()
    X, y, groups = loader.get_feature_arrays(dummy_clinical_data)

    assert 'name' not in X.columns, "CRITICAL FAILURE: Group ID 'name' leaked into features (X)!"
    assert 'status' not in X.columns, "CRITICAL FAILURE: Target 'status' leaked into features (X)!"

    # Check that X only contains the 2 acoustic features
    assert len(X.columns) == 2
    assert list(groups) == ['phon_R01_S01_1', 'phon_R01_S01_2', 'phon_R01_S02_1']


from src.preprocessing import DataSplitter, PreProcessor


def test_patient_leakage():
    """
    CRITICAL: Verifies that patient IDs in the train set NEVER
    intersect with patient IDs in the validation set.
    """
    # Create synthetic dataset with overlapping patient IDs
    X = pd.DataFrame({'feat1': range(10), 'feat2': range(10)})
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    # Patient A has 5 records, Patient B has 3, Patient C has 2
    groups = pd.Series(['Patient_A'] * 5 + ['Patient_B'] * 3 + ['Patient_C'] * 2)

    splitter = DataSplitter(n_splits=2)

    for train_idx, val_idx in splitter.gkf.split(X, y, groups):
        train_groups = set(groups.iloc[train_idx])
        val_groups = set(groups.iloc[val_idx])

        # The intersection of groups must be completely empty
        intersection = train_groups.intersection(val_groups)
        assert len(intersection) == 0, f"LEAKAGE DETECTED! Overlapping patients: {intersection}"


def test_preprocessor_scaling():
    """Verifies that the PreProcessor maintains DataFrame structure and scales correctly."""
    X_train = pd.DataFrame({'feat1': [1, 2, 3], 'feat2': [40, 50, 60]})
    X_val = pd.DataFrame({'feat1': [4], 'feat2': [70]})

    preprocessor = PreProcessor()

    # Fit and transform train
    X_train_scaled = preprocessor.fit_transform(X_train)
    assert X_train_scaled.shape == X_train.shape

    # Just transform val (no fitting)
    X_val_scaled = preprocessor.transform(X_val)
    assert X_val_scaled.shape == X_val.shape

    # Check that it returns pandas DataFrames, not numpy arrays
    assert isinstance(X_train_scaled, pd.DataFrame)
    assert isinstance(X_val_scaled, pd.DataFrame)