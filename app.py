"""Interactive Streamlit web app for the House Price ML Pipeline."""

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import load_data, split_data
from src.model import evaluate_model, train_model
from src.preprocessing import create_preprocessor, preprocess_data
from src.visualize_interactive import (
    plot_feature_importance_interactive,
    plot_predictions_vs_actual_interactive,
    plot_residuals_interactive,
)

# ── Feature metadata ─────────────────────────────────────────────────────────
FEATURE_INFO = {
    "MedInc": {
        "label": "Median Income",
        "help": "Median income in block group (tens of thousands USD).",
        "min": 0.0,
        "max": 15.0,
        "default": 3.87,
    },
    "HouseAge": {
        "label": "House Age",
        "help": "Median house age in block group (years).",
        "min": 1.0,
        "max": 52.0,
        "default": 29.0,
    },
    "AveRooms": {
        "label": "Average Rooms",
        "help": "Average number of rooms per household.",
        "min": 1.0,
        "max": 50.0,
        "default": 5.43,
    },
    "AveBedrms": {
        "label": "Average Bedrooms",
        "help": "Average number of bedrooms per household.",
        "min": 0.5,
        "max": 20.0,
        "default": 1.10,
    },
    "Population": {
        "label": "Population",
        "help": "Block group population.",
        "min": 1.0,
        "max": 40000.0,
        "default": 1425.0,
    },
    "AveOccup": {
        "label": "Average Occupancy",
        "help": "Average number of household members.",
        "min": 1.0,
        "max": 30.0,
        "default": 3.07,
    },
    "Latitude": {
        "label": "Latitude",
        "help": "Block group latitude.",
        "min": 32.0,
        "max": 42.0,
        "default": 35.63,
    },
    "Longitude": {
        "label": "Longitude",
        "help": "Block group longitude.",
        "min": -125.0,
        "max": -114.0,
        "default": -119.57,
    },
}

FEATURE_NAMES = list(FEATURE_INFO.keys())


# ── Cached helpers ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading California Housing dataset…")
def get_data():
    """Load the dataset (cached)."""
    return load_data()


@st.cache_resource(show_spinner="Training model – this may take a moment…")
def get_trained_artifacts():
    """Train model and return all artefacts needed by the app (cached).

    Returns a dict with keys:
        model, preprocessor, metrics, y_test, y_pred, feature_names,
        X_train, X_test
    """
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_p, X_test_p, preprocessor = preprocess_data(X_train, X_test)
    model = train_model(X_train_p, y_train)
    metrics = evaluate_model(model, X_test_p, y_test)
    y_pred = model.predict(X_test_p)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "metrics": metrics,
        "y_test": y_test,
        "y_pred": y_pred,
        "feature_names": list(X_train.columns),
        "X_train": X_train,
        "X_test": X_test,
    }


# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
)


# ── Sidebar navigation ───────────────────────────────────────────────────────
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Predict", "📊 Dashboard", "🔍 Data Explorer"],
)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Predict
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Predict":
    st.title("🏠 House Price Predictor")
    st.markdown(
        "Enter the details of a California block group below to get a "
        "predicted **median house value** (in hundreds of thousands of dollars)."
    )

    artifacts = get_trained_artifacts()
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]

    with st.form("predict_form"):
        cols = st.columns(2)
        user_inputs = {}
        for idx, (feat, info) in enumerate(FEATURE_INFO.items()):
            with cols[idx % 2]:
                user_inputs[feat] = st.number_input(
                    info["label"],
                    min_value=info["min"],
                    max_value=info["max"],
                    value=info["default"],
                    help=info["help"],
                    step=0.01 if info["max"] <= 60 else 1.0,
                )

        submitted = st.form_submit_button("Predict Price", type="primary")

    if submitted:
        input_df = pd.DataFrame([user_inputs])
        input_scaled = preprocessor.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        st.success(
            f"### Predicted Median House Value: **${prediction * 100_000:,.0f}**"
        )
        st.caption(
            "Value is in 1990 dollars. The California Housing dataset "
            "target represents median house value in units of $100 000."
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Dashboard
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Dashboard":
    st.title("📊 Model Performance Dashboard")

    artifacts = get_trained_artifacts()
    metrics = artifacts["metrics"]
    y_test = artifacts["y_test"]
    y_pred = artifacts["y_pred"]
    feature_names = artifacts["feature_names"]
    model = artifacts["model"]

    # Metric cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{metrics['MAE']:.4f}")
    m2.metric("MSE", f"{metrics['MSE']:.4f}")
    m3.metric("RMSE", f"{metrics['RMSE']:.4f}")
    m4.metric("R²", f"{metrics['R2']:.4f}")

    st.divider()

    # Charts
    tab1, tab2, tab3 = st.tabs(
        ["Feature Importance", "Predictions vs Actual", "Residuals"]
    )

    with tab1:
        st.plotly_chart(
            plot_feature_importance_interactive(model, feature_names),
            use_container_width=True,
        )

    with tab2:
        st.plotly_chart(
            plot_predictions_vs_actual_interactive(y_test, y_pred),
            use_container_width=True,
        )

    with tab3:
        st.plotly_chart(
            plot_residuals_interactive(y_test, y_pred),
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Data Explorer
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Data Explorer":
    st.title("🔍 Data Explorer")

    df = get_data()

    st.subheader("Dataset Overview")
    st.markdown(
        f"**{df.shape[0]:,}** samples × **{df.shape[1]}** columns "
        f"(8 features + target)"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**First rows**")
        st.dataframe(df.head(10), use_container_width=True)
    with col2:
        st.markdown("**Descriptive statistics**")
        st.dataframe(df.describe().T, use_container_width=True)

    st.divider()

    st.subheader("Feature Distributions")
    selected_feature = st.selectbox("Select a feature", FEATURE_NAMES + ["target"])

    import plotly.express as px

    fig = px.histogram(
        df,
        x=selected_feature,
        nbins=60,
        marginal="box",
        title=f"Distribution of {selected_feature}",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Correlation Heatmap")
    import plotly.figure_factory as ff

    corr = df.corr(numeric_only=True)
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Matrix",
        aspect="auto",
    )
    fig_corr.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_corr, use_container_width=True)
