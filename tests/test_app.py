"""Tests for the interactive Streamlit web app and supporting modules."""

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_data, split_data
from src.model import train_model
from src.preprocessing import preprocess_data
from src.visualize_interactive import (
    plot_feature_importance_interactive,
    plot_predictions_vs_actual_interactive,
    plot_residuals_interactive,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures (reuse the same synthetic data approach as test_pipeline)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]

_N = 200


def _make_fake_housing():
    from sklearn.datasets import make_regression
    X, y = make_regression(
        n_samples=_N, n_features=len(FEATURE_COLS), noise=10.0, random_state=0
    )
    y = (y - y.min()) / (y.max() - y.min()) * 5
    df = pd.DataFrame(X, columns=FEATURE_COLS)
    df["MedHouseVal"] = y
    return SimpleNamespace(frame=df)


@pytest.fixture
def trained_artifacts():
    """Return a dict containing model, preprocessor, metrics, etc."""
    with patch(
        "src.data_loader.fetch_california_housing",
        return_value=_make_fake_housing(),
    ):
        df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_p, X_test_p, preprocessor = preprocess_data(X_train, X_test)
    model = train_model(X_train_p, y_train)
    y_pred = model.predict(X_test_p)
    return {
        "model": model,
        "preprocessor": preprocessor,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "feature_names": list(X_train.columns),
    }


# ---------------------------------------------------------------------------
# Interactive visualization tests
# ---------------------------------------------------------------------------

class TestFeatureImportanceInteractive:
    def test_returns_figure(self, trained_artifacts):
        fig = plot_feature_importance_interactive(
            trained_artifacts["model"],
            trained_artifacts["feature_names"],
        )
        assert fig is not None
        assert hasattr(fig, "to_json")

    def test_has_correct_title(self, trained_artifacts):
        fig = plot_feature_importance_interactive(
            trained_artifacts["model"],
            trained_artifacts["feature_names"],
        )
        assert fig.layout.title.text == "Feature Importances"


class TestPredictionsVsActualInteractive:
    def test_returns_figure(self, trained_artifacts):
        fig = plot_predictions_vs_actual_interactive(
            trained_artifacts["y_test"],
            trained_artifacts["y_pred"],
        )
        assert fig is not None
        assert hasattr(fig, "to_json")

    def test_has_two_traces(self, trained_artifacts):
        fig = plot_predictions_vs_actual_interactive(
            trained_artifacts["y_test"],
            trained_artifacts["y_pred"],
        )
        # One scatter trace + one ideal-line trace
        assert len(fig.data) == 2


class TestResidualsInteractive:
    def test_returns_figure(self, trained_artifacts):
        fig = plot_residuals_interactive(
            trained_artifacts["y_test"],
            trained_artifacts["y_pred"],
        )
        assert fig is not None
        assert hasattr(fig, "to_json")

    def test_has_correct_title(self, trained_artifacts):
        fig = plot_residuals_interactive(
            trained_artifacts["y_test"],
            trained_artifacts["y_pred"],
        )
        assert fig.layout.title.text == "Residuals Distribution"


# ---------------------------------------------------------------------------
# Prediction logic tests (mirrors what the Streamlit app does)
# ---------------------------------------------------------------------------

class TestPredictionFlow:
    """Test the prediction flow as used in the Streamlit app."""

    def test_single_prediction(self, trained_artifacts):
        """User input → scale → predict should return a single float."""
        model = trained_artifacts["model"]
        preprocessor = trained_artifacts["preprocessor"]

        user_input = {col: 0.0 for col in FEATURE_COLS}
        input_df = pd.DataFrame([user_input])
        input_scaled = preprocessor.transform(input_df)
        prediction = model.predict(input_scaled)

        assert prediction.shape == (1,)
        assert isinstance(float(prediction[0]), float)

    def test_batch_prediction(self, trained_artifacts):
        """Multiple inputs should return the same number of predictions."""
        model = trained_artifacts["model"]
        preprocessor = trained_artifacts["preprocessor"]

        rows = [{col: float(i) for col in FEATURE_COLS} for i in range(5)]
        input_df = pd.DataFrame(rows)
        input_scaled = preprocessor.transform(input_df)
        predictions = model.predict(input_scaled)

        assert predictions.shape == (5,)
