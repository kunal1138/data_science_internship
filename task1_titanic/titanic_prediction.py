# Task 1: Titanic Survival Prediction
# Step 1: Load and inspect the dataset

from pathlib import Path
import pandas as pd

# Find the CSV in the same folder as this Python file.
dataset_path = Path(__file__).parent / "Titanic-Dataset.csv"
df = pd.read_csv(dataset_path)

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nDATASET SIZE (rows, columns):")
print(df.shape)

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nSURVIVAL COUNTS (0 = Did not survive, 1 = Survived):")
print(df["Survived"].value_counts())

# Step 2: Select input features and the prediction target
from sklearn.model_selection import train_test_split

features = [
    "Pclass", "Sex", "Age", "SibSp",
    "Parch", "Fare", "Embarked"
]

# Leave out identifiers, names, tickets and the mostly missing Cabin column.
X = df[features]
y = df["Survived"]

# Step 3: Use 80% for training and 20% for testing.
# Stratification keeps similar survival proportions in both groups.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTRAINING PASSENGERS:", len(X_train))
print("TESTING PASSENGERS:", len(X_test))

# Step 4: Prepare numerical and categorical inputs
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

numeric_features = ["Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Pclass", "Sex", "Embarked"]

# Fill missing numbers with the training median, then scale.
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Fill missing categories with the most common training value.
# Convert categories into numerical columns.
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Step 5: Train a logistic regression classification model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Learn preprocessing values and model parameters only from training data.
model.fit(X_train, y_train)

# Step 6: Evaluate on unseen test passengers
predictions = model.predict(X_test)

print("\nTEST ACCURACY:")
print(f"{accuracy_score(y_test, predictions):.2%}")

print("\nCLASSIFICATION REPORT:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Did not survive", "Survived"],
    zero_division=0
))

print("\nCONFUSION MATRIX:")
print("Rows = actual, columns = predicted; order: 0, 1")
print(confusion_matrix(y_test, predictions))

# Step 7: Save actual outcomes and model predictions
results = X_test.copy()
results["Actual_Survived"] = y_test
results["Predicted_Survived"] = predictions

output_path = Path(__file__).parent / "test_predictions.csv"
results.to_csv(output_path, index=False)

print("\nSAMPLE TEST PREDICTIONS:")
print(results[["Actual_Survived", "Predicted_Survived"]].head(10))
print("\nPredictions saved to:", output_path)