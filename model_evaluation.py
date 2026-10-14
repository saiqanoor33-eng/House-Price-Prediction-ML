import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the Task 06 dataset
df = pd.read_csv("house_prices.csv")

# Standardize column names so small capitalization differences do not break the script
df.columns = [column.strip().lower() for column in df.columns]

required_columns = ["area", "bedrooms", "bathrooms", "price"]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}. "
        "The CSV must contain Area, Bedrooms, Bathrooms, and Price."
    )

df = df[required_columns].copy()
for column in required_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=["price"])
X = df[["area", "bedrooms", "bathrooms"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Evaluate the original algorithm and one alternative algorithm
models = {
    "Random Forest (baseline)": RandomForestRegressor(
        n_estimators=100, random_state=42
    ),
    "Linear Regression (alternative)": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LinearRegression())
    ])
}

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2 Score": round(r2, 4)
    })
    trained_models[name] = model

results_df = pd.DataFrame(results).sort_values(
    by=["MAE", "R2 Score"], ascending=[True, False]
)
results_df.to_csv("model_comparison.csv", index=False)

print("\nMODEL EVALUATION RESULTS")
print(results_df.to_string(index=False))

# Select the model with the lowest MAE; R2 is used as a tie-breaker
best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]
joblib.dump(best_model, "best_house_price_model.joblib")

print(f"\nSelected final model: {best_name}")
print("Selection rule: lowest MAE, with higher R2 used to break ties.")
print("Saved comparison to model_comparison.csv")
print("Saved selected model to best_house_price_model.joblib")

# Show the selected model's prediction for the example used in Task 06
sample_house = pd.DataFrame(
    {"area": [2000], "bedrooms": [3], "bathrooms": [2]}
)
sample_prediction = best_model.predict(sample_house)[0]
print(f"Example prediction (2000 sq ft, 3 bedrooms, 2 bathrooms): {sample_prediction:,.2f}")
