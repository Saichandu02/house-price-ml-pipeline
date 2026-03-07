"""Module for generating interactive Plotly visualizations for the web app."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def plot_feature_importance_interactive(model, feature_names):
    """Return an interactive Plotly bar chart of feature importances.

    Args:
        model: Fitted model with a ``feature_importances_`` attribute.
        feature_names: Ordered list of feature name strings.

    Returns:
        plotly.graph_objects.Figure
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig = px.bar(
        x=sorted_importances,
        y=sorted_names,
        orientation="h",
        labels={"x": "Importance", "y": "Feature"},
        title="Feature Importances",
        color=sorted_importances,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        yaxis_title=None,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def plot_predictions_vs_actual_interactive(y_test, y_pred):
    """Return an interactive Plotly scatter of predicted vs actual values.

    Args:
        y_test: True target values.
        y_pred: Model-predicted target values.

    Returns:
        plotly.graph_objects.Figure
    """
    y_test_arr = np.asarray(y_test)
    y_pred_arr = np.asarray(y_pred)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=y_test_arr,
            y=y_pred_arr,
            mode="markers",
            marker=dict(size=4, opacity=0.4, color="#636EFA"),
            name="Predictions",
        )
    )
    min_val = min(float(y_test_arr.min()), float(y_pred_arr.min()))
    max_val = max(float(y_test_arr.max()), float(y_pred_arr.max()))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="red", dash="dash", width=2),
            name="Ideal",
        )
    )
    fig.update_layout(
        title="Predictions vs Actual",
        xaxis_title="Actual",
        yaxis_title="Predicted",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def plot_residuals_interactive(y_test, y_pred):
    """Return an interactive Plotly histogram of residuals.

    Args:
        y_test: True target values.
        y_pred: Model-predicted target values.

    Returns:
        plotly.graph_objects.Figure
    """
    residuals = np.asarray(y_test) - np.asarray(y_pred)

    fig = px.histogram(
        x=residuals,
        nbins=50,
        labels={"x": "Residual", "count": "Count"},
        title="Residuals Distribution",
        marginal="rug",
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=2)
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig
