# House Price ML Pipeline

A fully working end-to-end machine learning pipeline for predicting California house prices, built with Python and scikit-learn.

## Project Structure

```
house-price-ml-pipeline/
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Dataset loading and train/test splitting
│   ├── preprocessing.py    # Feature scaling via StandardScaler
│   ├── model.py            # Model training, evaluation, and persistence
│   ├── visualize.py        # Visualization helpers
│   └── pipeline.py         # End-to-end orchestrator
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py    # pytest unit tests
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

The pipeline uses the **California Housing dataset** from `sklearn.datasets.fetch_california_housing`.  
It contains 20,640 samples with 8 numeric features (median income, house age, average rooms, average bedrooms, population, average occupancy, latitude, longitude) and a target representing the **median house value** in hundreds of thousands of dollars.  
No external data files are needed — the dataset is fetched programmatically.

## ML Model

A **Random Forest Regressor** (`n_estimators=100, random_state=42`) is used because:
- It handles non-linear relationships between features and target well.
- It is robust to outliers and does not require extensive hyperparameter tuning to achieve good results.
- Feature importances are directly available, making the model interpretable.

## Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run the full pipeline

```bash
python -m src.pipeline
```

This will:
1. Load the California Housing dataset.
2. Split it into training (80%) and test (20%) sets.
3. Scale features with `StandardScaler`.
4. Train a `RandomForestRegressor`.
5. Print MAE, MSE, RMSE, and R² metrics.
6. Save the model to `models/model.joblib`.
7. Save three plots to `outputs/`:
   - `feature_importance.png`
   - `predictions_vs_actual.png`
   - `residuals.png`

### Run the tests

```bash
pytest tests/
```
