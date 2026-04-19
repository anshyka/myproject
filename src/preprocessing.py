import pandas as pd
from typing import Tuple


class ParkinsonsDataLoader:
    """Handles loading and validation of the Parkinson's acoustic dataset."""

    def __init__(self, target_col: str = 'status', group_col: str = 'name'):
        self.target_col = target_col
        self.group_col = group_col

    def load_data(self, path: str) -> pd.DataFrame:
        """Loads the dataset from a CSV filepath."""
        return pd.read_csv(path)

    def validate_schema(self, df: pd.DataFrame) -> None:
        """
        Ensures essential clinical columns are present.
        Raises ValueError if 'status' or 'name' are missing.
        """
        if self.target_col not in df.columns:
            raise ValueError(f"Schema Error: Missing target column '{self.target_col}'")
        if self.group_col not in df.columns:
            raise ValueError(f"Schema Error: Missing group column '{self.group_col}'")

    def get_feature_arrays(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Separates features (X), target (y), and groups.
        Critically ensures group IDs do not leak into the feature matrix.
        """
        self.validate_schema(df)

        y = df[self.target_col]
        groups = df[self.group_col]

        # Isolate features by dropping the target and the group ID
        X = df.drop(columns=[self.target_col, self.group_col])

        return X, y, groups


from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from typing import Generator


class DataSplitter:
    """Handles leakage-proof cross-validation using GroupKFold."""

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self.gkf = GroupKFold(n_splits=self.n_splits)

    def split(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> Generator:
        """Yields train and validation splits ensuring no patient group overlap."""
        for train_idx, val_idx in self.gkf.split(X, y, groups):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            yield X_train, X_val, y_train, y_val


class PreProcessor:
    """Wraps StandardScaler to prevent data leakage during scaling."""

    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Fits the scaler on training data and transforms it."""
        scaled_data = self.scaler.fit_transform(X_train)
        return pd.DataFrame(scaled_data, columns=X_train.columns, index=X_train.index)

    def transform(self, X_val: pd.DataFrame) -> pd.DataFrame:
        """Transforms validation data using the ALREADY FITTED scaler."""
        scaled_data = self.scaler.transform(X_val)
        return pd.DataFrame(scaled_data, columns=X_val.columns, index=X_val.index)