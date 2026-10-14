# Task 07 — Evaluate & Improve the ML Model

## Project
House Price Prediction using machine learning.

## Objective
Evaluate the Task 06 model, identify weaknesses, test an alternative algorithm, compare results, and select a final model.

## Dataset
The project uses `house_prices.csv`, with these columns:
- `Area`
- `Bedrooms`
- `Bathrooms`
- `Price` (target)

## Models compared
1. **Random Forest Regressor** — baseline model from Task 06.
2. **Linear Regression** — alternative model tested as an improvement.

Both models use the same 80/20 train-test split (`random_state=42`) so the comparison is fair. The Linear Regression pipeline also fills missing feature values using the median.

## Evaluation metrics
- **MAE (Mean Absolute Error):** average absolute difference between actual and predicted prices. Lower is better.
- **RMSE (Root Mean Squared Error):** gives larger errors more weight. Lower is better.
- **R² Score:** indicates how much variation in prices is explained by the model. Closer to 1 is generally better.

## How to run
Install the required libraries:

```bash
pip install pandas numpy scikit-learn joblib
```

Keep `model_evaluation.py` and `house_prices.csv` in the same folder, then run:

```bash
python model_evaluation.py
```

## Outputs
The script creates:
- `model_comparison.csv` — evaluation metrics for both models.
- `best_house_price_model.joblib` — the selected model.

## Model weaknesses and improvement
Potential weaknesses include limited features (only area, bedrooms, and bathrooms), possible synthetic or small dataset bias, and performance that may not generalize to real housing markets. The alternative Linear Regression model is tested on the same held-out test data. The script selects the model with the lowest MAE, using R² as a tie-breaker.

## Results
After running the script, copy the actual MAE, RMSE, and R² values printed in the terminal into this section or into a screenshot of the output. Do not invent results; the winning model is determined by the evaluation run.
