import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("water_consumption_forecasting.csv")

# Convert date column (Indian format safe)
df["date"] = pd.to_datetime(df["date"], dayfirst=True)

# Sort by date
df = df.sort_values("date").reset_index(drop=True)

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

# Day of week & weekend flag
df["day_of_week"] = df["date"].dt.day_name()
df["is_weekend"] = (df["date"].dt.weekday >= 5).astype(int)

# One-hot encode region
df = pd.get_dummies(df, columns=["region"], drop_first=True)

# One-hot encode day of week
df = pd.get_dummies(df, columns=["day_of_week"], drop_first=True)

# Convert boolean columns to int
for col in df.columns:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)

# -----------------------------
# LAG FEATURE (Previous Day)
# -----------------------------
region_cols = [c for c in df.columns if c.startswith("region_")]

df["prev_day_consumption"] = (
    df.groupby(region_cols)["consumption_liters"].shift(1)
)

# Remove rows with NaN
df = df.dropna()

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X = df.drop(columns=["date", "consumption_liters"])
y = df["consumption_liters"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Model Evaluation")
print("MAE:", mae)
print("MSE:", mse)

# -----------------------------
# EXPORT RESULTS (FOR TABLEAU)
# -----------------------------
results = pd.DataFrame({
    "Actual_Consumption": y_test.values,
    "Predicted_Consumption": y_pred
})

results.to_csv("water_forecast_results.csv", index=False)

# -----------------------------
# PREDICTION FUNCTION (USED BY STREAMLIT)
# -----------------------------
def predict_water_consumption(
    prev_day_consumption: float,
    is_weekend: int,
    region: str,
    day_of_week: str
) -> float:
    """
    Predict water consumption for a given day
    """

    # Create input row with zeros
    input_data = pd.DataFrame(columns=X.columns)
    input_data.loc[0] = 0

    # Base features
    input_data["prev_day_consumption"] = prev_day_consumption
    input_data["is_weekend"] = is_weekend

    # Region encoding
    region_col = f"region_{region}"
    if region_col in input_data.columns:
        input_data[region_col] = 1

    # Day encoding
    day_col = f"day_of_week_{day_of_week}"
    if day_col in input_data.columns:
        input_data[day_col] = 1

    # Prediction
    prediction = model.predict(input_data)[0]
    return float(prediction)
