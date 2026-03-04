"""Module for loading and preparing the California Housing dataset."""

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def load_data() -> pd.DataFrame:
    """Load the California Housing dataset from scikit-learn.

    Returns:
        pd.DataFrame: DataFrame with feature columns and a 'target' column
                      representing the median house value (MedHouseVal).
    """
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()
    df.rename(columns={"MedHouseVal": "target"}, inplace=True)
    return df


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    """Split the DataFrame into training and test sets.

    Args:
        df: DataFrame containing features and the 'target' column.
        test_size: Fraction of data to use for testing.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) as DataFrames/Series.
    """
    X = df.drop(columns=["target"])
    y = df["target"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
