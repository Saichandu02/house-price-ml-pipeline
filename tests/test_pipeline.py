"""Unit tests for the house price ML pipeline."""

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_data, split_data
from src.preprocessing import create_preprocessor, preprocess_data
from src.model import evaluate_model, load_model, save_model, train_model

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]

_RNG = np.random.RandomState(0)
_N = 500  # small synthetic dataset used across all tests


def _make_fake_housing():
    """Return a Bunch-like object that mimics fetch_california_housing(as_frame=True)."""
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=_N, n_features=len(FEATURE_COLS), noise=10.0, random_state=0)
    # Scale target to a realistic range (0–5)
    y = (y - y.min()) / (y.max() - y.min()) * 5
    df = pd.DataFrame(X, columns=FEATURE_COLS)
    df["MedHouseVal"] = y
    return SimpleNamespace(frame=df)


@pytest.fixture
def sample_df():
    """Small synthetic DataFrame with the California Housing schema."""
    with patch("src.data_loader.fetch_california_housing", return_value=_make_fake_housing()):
        return load_data()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def test_load_data_shape(sample_df):
    """DataFrame should have _N rows and 9 columns (8 features + target)."""
    assert sample_df.shape == (_N, 9)


def test_load_data_columns(sample_df):
    """DataFrame should contain all expected feature columns and 'target'."""
    for col in FEATURE_COLS:
        assert col in sample_df.columns
    assert "target" in sample_df.columns


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------

def test_split_data_proportions(sample_df):
    """Test split produces ~80/20 train/test proportion."""
    X_train, X_test, y_train, y_test = split_data(sample_df)
    total = len(X_train) + len(X_test)
    assert total == len(sample_df)
    assert abs(len(X_test) / total - 0.2) < 0.02


def test_split_data_no_overlap(sample_df):
    """Train and test sets must not share index values."""
    X_train, X_test, _, _ = split_data(sample_df)
    assert len(set(X_train.index) & set(X_test.index)) == 0


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def test_preprocess_data_shape(sample_df):
    """Preprocessed arrays should retain the same number of samples/features."""
    X_train, X_test, _, _ = split_data(sample_df)
    X_train_p, X_test_p, _ = preprocess_data(X_train, X_test)
    assert X_train_p.shape == X_train.shape
    assert X_test_p.shape == X_test.shape


def test_preprocess_data_mean_near_zero(sample_df):
    """After StandardScaler the training set should have near-zero column means."""
    X_train, X_test, _, _ = split_data(sample_df)
    X_train_p, _, _ = preprocess_data(X_train, X_test)
    assert np.allclose(X_train_p.mean(axis=0), 0, atol=1e-6)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def test_train_model_returns_fitted_model(sample_df):
    """train_model should return a fitted RandomForestRegressor."""
    from sklearn.ensemble import RandomForestRegressor

    X_train, X_test, y_train, _ = split_data(sample_df)
    X_train_p, _, _ = preprocess_data(X_train, X_test)
    model = train_model(X_train_p, y_train)
    assert isinstance(model, RandomForestRegressor)
    # A fitted model exposes estimators_
    assert len(model.estimators_) == 100


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def test_evaluate_model_keys(sample_df):
    """evaluate_model should return a dict with MAE, MSE, RMSE, and R2 keys."""
    X_train, X_test, y_train, y_test = split_data(sample_df)
    X_train_p, X_test_p, _ = preprocess_data(X_train, X_test)
    model = train_model(X_train_p, y_train)
    metrics = evaluate_model(model, X_test_p, y_test)
    assert set(metrics.keys()) == {"MAE", "MSE", "RMSE", "R2"}


def test_evaluate_model_values_reasonable(sample_df):
    """R² on the test set should be a positive number."""
    X_train, X_test, y_train, y_test = split_data(sample_df)
    X_train_p, X_test_p, _ = preprocess_data(X_train, X_test)
    model = train_model(X_train_p, y_train)
    metrics = evaluate_model(model, X_test_p, y_test)
    assert metrics["R2"] > 0


# ---------------------------------------------------------------------------
# Model save / load round-trip
# ---------------------------------------------------------------------------

def test_save_load_model_roundtrip(sample_df):
    """A saved model should produce identical predictions after reloading."""
    X_train, X_test, y_train, y_test = split_data(sample_df)
    X_train_p, X_test_p, _ = preprocess_data(X_train, X_test)
    model = train_model(X_train_p, y_train)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "model.joblib")
        save_model(model, filepath)
        loaded = load_model(filepath)

    preds_original = model.predict(X_test_p)
    preds_loaded = loaded.predict(X_test_p)
    np.testing.assert_array_equal(preds_original, preds_loaded)
