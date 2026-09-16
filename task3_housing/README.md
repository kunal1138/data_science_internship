# Task 3: Simple Linear Regression on Housing

## Objective
Predict house sale prices using above-ground living area.

## Dataset
Kaggle House Prices: Advanced Regression Techniques:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

Uses train.csv, containing 1,460 houses.

## Method
- Selected GrLivArea (living area in square feet) as the input.
- Used SalePrice (USD) as the prediction target.
- Confirmed neither selected column contains missing values.
- Split data into 1,168 training and 292 testing houses.
- Applied MinMaxScaler and trained LinearRegression in a pipeline.
- Fitted scaling and regression only on training data.
- Used random_state=42 for a reproducible split.

## Test Results
| Metric | Result |
|---|---:|
| MAE | $38,341.20 |
| RMSE | $58,471.76 |
| R² | 0.5543 |

The model explains approximately 55.4% of the variation in
test-set prices. R² is not classification accuracy.

## Limitations
Living area alone does not capture location, condition,
quality or other factors affecting prices.
Results are based on one held-out test split.

## Run
Place train.csv beside housing_prediction.py.
From the data_science_internship folder, run:

```powershell
.\.venv\Scripts\python.exe task3_housing\housing_prediction.py
```

Required libraries: pandas, scikit-learn and matplotlib.

## Outputs
- test_predictions.csv: Actual and predicted test prices
- housing_regression.png: Test prices and fitted regression line