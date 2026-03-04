"""Module for data preprocessing."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_preprocessor() -> Pipeline:
    """Build a preprocessing pipeline with standard scaling.

    Returns:
        sklearn.pipeline.Pipeline: Preprocessing pipeline.
    """
    return Pipeline(steps=[("scaler", StandardScaler())])


def preprocess_data(X_train, X_test, preprocessor=None):
    """Fit preprocessor on training data and transform both train and test sets.

    Args:
        X_train: Training feature set.
        X_test: Test feature set.
        preprocessor: An optional fitted or unfitted sklearn preprocessor.
                      If None, a new preprocessor is created via
                      :func:`create_preprocessor`.

    Returns:
        Tuple of (X_train_processed, X_test_processed, preprocessor).
    """
    if preprocessor is None:
        preprocessor = create_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    return X_train_processed, X_test_processed, preprocessor
