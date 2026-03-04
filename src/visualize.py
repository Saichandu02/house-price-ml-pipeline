"""Module for generating and saving visualizations of model results."""

import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_feature_importance(
    model,
    feature_names,
    save_path: str = "outputs/feature_importance.png",
) -> None:
    """Plot feature importances from a tree-based model.

    Args:
        model: Fitted model with a ``feature_importances_`` attribute.
        feature_names: Ordered list of feature name strings.
        save_path: File path where the plot will be saved.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(sorted_names)), sorted_importances)
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=45, ha="right")
    ax.set_title("Feature Importances")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_predictions_vs_actual(
    y_test,
    y_pred,
    save_path: str = "outputs/predictions_vs_actual.png",
) -> None:
    """Scatter plot of predicted values against actual values.

    Args:
        y_test: True target values.
        y_pred: Model-predicted target values.
        save_path: File path where the plot will be saved.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.3, s=10)
    min_val = min(float(np.min(y_test)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_test)), float(np.max(y_pred)))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5)
    ax.set_title("Predictions vs Actual")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_residuals(
    y_test,
    y_pred,
    save_path: str = "outputs/residuals.png",
) -> None:
    """Plot the distribution of prediction residuals.

    Args:
        y_test: True target values.
        y_pred: Model-predicted target values.
        save_path: File path where the plot will be saved.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    residuals = np.array(y_test) - np.array(y_pred)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, kde=True, ax=ax, bins=50)
    ax.axvline(x=0, color="r", linestyle="--", linewidth=1.5)
    ax.set_title("Residuals Distribution")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
