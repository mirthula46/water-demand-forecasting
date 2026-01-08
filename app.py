import streamlit as st
import pandas as pd
from water_demand_forecasting import predict_water_consumption

# Page config
st.set_page_config(page_title="Water Consumption Forecast", layout="centered")

# Title & description
st.title("Campus Water Consumption Forecasting")

st.write(
    """
    This application predicts **daily water consumption** for a campus
    based on historical usage patterns, day of the week, and region.

    This model is built using **Machine Learning (Linear Regression)**.
    """
)

st.divider()

# ---------------- USER INPUTS ----------------
st.subheader("Enter Input Details")

prev_day_consumption = st.number_input(
    "Previous Day Water Consumption (liters)",
    min_value=0.0,
    step=100.0
)

is_weekend = st.checkbox("Is it a weekend?")

region = st.selectbox(
    "Select Region / Campus",
    ["East", "North", "South", "West"]
)

day_of_week = st.selectbox(
    "Select Day of the Week",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

st.divider()

# ---------------- PREDICTION ----------------
if st.button("Predict Water Consumption"):
    prediction = predict_water_consumption(
        prev_day_consumption=prev_day_consumption,
        is_weekend=int(is_weekend),
        region=region,
        day_of_week=day_of_week
    )

    st.success(f"Predicted Water Consumption: {prediction:.2f} liters")

    # Chart
    chart_df = pd.DataFrame(
        {
            "Consumption (liters)": [
                prev_day_consumption,
                prediction
            ]
        },
        index=["Previous Day", "Predicted Day"]
    )

    st.subheader("Consumption Comparison")
    st.bar_chart(chart_df)

# Footer
st.caption("Machine Learning Project | Campus Water Demand Forecasting")
