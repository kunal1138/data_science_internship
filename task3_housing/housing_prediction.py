# Task 3: Simple Linear Regression on Housing
# Step 1: Load and inspect the dataset

from pathlib import Path
import pandas as pd

dataset_path = Path(__file__).parent / "train.csv"
df = pd.read_csv(dataset_path)

print("\nDATASET SIZE (rows, columns):")
print(df.shape)

# Inspect the proposed input and prediction target.
columns = ["GrLivArea", "SalePrice"]

print("\nFIRST 5 ROWS:")
print(df[columns].head())

print("\nMISSING VALUES:")
print(df[columns].isna().sum())

print("\nSUMMARY:")
print(df[columns].describe())

# Step 2: Select the input and target, then split the data
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = df[["GrLivArea"]]
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTRAINING HOUSES:", len(X_train))
print("TESTING HOUSES:", len(X_test))

# Step 3: Normalise the input and train the model.
# Scaling is fitted only on training data to avoid data leakage.
model = Pipeline([
    ("scaler", MinMaxScaler()),
    ("regression", LinearRegression())
])

model.fit(X_train, y_train)

# Step 4: Evaluate predictions on unseen test houses
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nTEST RESULTS:")
print(f"MAE: ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R² score: {r2:.4f}")

# Step 5: Save test predictions
import matplotlib.pyplot as plt

output_folder = Path(__file__).parent

results = X_test.copy()
results["Actual_Price"] = y_test
results["Predicted_Price"] = predictions
results.to_csv(output_folder / "test_predictions.csv", index=False)

# Step 6: Plot actual test prices and the fitted regression line
sorted_inputs = X_test.sort_values("GrLivArea")
line_predictions = model.predict(sorted_inputs)

plt.figure(figsize=(10, 6))

plt.scatter(
    X_test["GrLivArea"], y_test,
    alpha=0.6, label="Actual test prices"
)

plt.plot(
    sorted_inputs["GrLivArea"], line_predictions,
    color="red", label="Regression line fitted on training data"
)

plt.title("House Price Prediction Using Living Area")
plt.xlabel("Above-ground living area (square feet)")
plt.ylabel("Sale price (USD)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(output_folder / "housing_regression.png", dpi=150)
plt.close()

print("\nSaved test_predictions.csv and housing_regression.png")