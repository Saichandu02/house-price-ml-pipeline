"""Main orchestrator for the house price ML pipeline."""

from src.data_loader import load_data, split_data
from src.preprocessing import preprocess_data
from src.model import train_model, evaluate_model, save_model
from src.visualize import (
    plot_feature_importance,
    plot_predictions_vs_actual,
    plot_residuals,
)


def run_pipeline() -> None:
    """Execute the end-to-end house price prediction pipeline.

    Steps:
        1. Load the California Housing dataset.
        2. Split into training and test sets.
        3. Preprocess features (standard scaling).
        4. Train a RandomForestRegressor.
        5. Evaluate the model and print metrics.
        6. Save the trained model to disk.
        7. Generate and save visualization plots.
    """
    # 1. Load data
    print("Loading data...")
    df = load_data()

    # 2. Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)

    # 3. Preprocess data
    print("Preprocessing data...")
    X_train_proc, X_test_proc, _ = preprocess_data(X_train, X_test)

    # 4. Train model
    print("Training model...")
    model = train_model(X_train_proc, y_train)

    # 5. Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test_proc, y_test)
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # 6. Save model
    print("Saving model...")
    save_model(model)

    # 7. Generate visualizations
    print("Generating visualizations...")
    y_pred = model.predict(X_test_proc)
    feature_names = list(X_train.columns)
    plot_feature_importance(model, feature_names)
    plot_predictions_vs_actual(y_test, y_pred)
    plot_residuals(y_test, y_pred)

    print("Pipeline complete.")


if __name__ == "__main__":
    run_pipeline()
