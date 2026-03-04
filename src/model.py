"""Module for model training, evaluation, and persistence."""

import os
from typing import Dict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_model(X_train, y_train) -> RandomForestRegressor:
    """Train a RandomForestRegressor on the provided data.

    Args:
        X_train: Training feature set.
        y_train: Training target values.

    Returns:
        Fitted RandomForestRegressor model.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    """Evaluate a trained model on test data.

    Args:
        model: Fitted sklearn estimator.
        X_test: Test feature set.
        y_test: True target values for the test set.

    Returns:
        Dictionary with keys 'MAE', 'MSE', 'RMSE', and 'R2'.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def save_model(model, filepath: str = "models/model.joblib") -> None:
    """Persist a trained model to disk using joblib.

    Args:
        model: Fitted sklearn estimator to save.
        filepath: Destination file path (directory will be created if absent).
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: str = "models/model.joblib"):
    """Load a previously saved model from disk.

    Args:
        filepath: Path to the serialised model file.

    Returns:
        The deserialized sklearn estimator.
    """
    return joblib.load(filepath)
