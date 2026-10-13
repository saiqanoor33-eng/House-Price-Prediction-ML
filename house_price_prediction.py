
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(42)

n = 300

area = np.random.randint(500, 4000, n)
bedrooms = np.random.randint(1, 7, n)
bathrooms = np.random.randint(1, 5, n)

price = (
    area * 8000
    + bedrooms * 500000
    + bathrooms * 700000
    + np.random.normal(0, 500000, n)
)

df = pd.DataFrame({
    "Area": area,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Price": price.astype(int)
})

df.to_csv("house_prices.csv", index=False)

X = df[["Area", "Bedrooms", "Bathrooms"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100, random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("House Price Prediction Model")
print("-----------------------------")
print("Total dataset rows:", len(df))
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 3))

new_house = pd.DataFrame({
    "Area": [2000],
    "Bedrooms": [3],
    "Bathrooms": [2]
})

predicted_price = model.predict(new_house)[0]

print("\nNew House Details")
print("Area: 2000 square feet")
print("Bedrooms: 3")
print("Bathrooms: 2")
print("Predicted Price:", round(predicted_price, 2))