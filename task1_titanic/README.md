# Task 1: Titanic Survival Prediction

Predicts passenger survival using a logistic regression classifier.

## Dataset
Source: https://www.kaggle.com/datasets/yasserh/titanic-dataset

891 passengers with 12 columns.
Target: Survived (0 = did not survive, 1 = survived).

## Method
- Selected passenger class, sex, age, family counts, fare and embarkation port.
- Split into 712 training and 179 testing passengers using stratification.
- Filled missing numerical values with training medians.
- Filled missing categories with the most common training values.
- Scaled numerical inputs and one-hot encoded categories.
- Trained and evaluated logistic regression.

Preprocessing was fitted only on training data.

## Test Results
- Accuracy: 80.45%
- Survivor precision: 0.79
- Survivor recall: 0.67
- Survivor F1-score: 0.72

These results describe one held-out test split.

## Run
From the data_science_internship folder:

```powershell
.\.venv\Scripts\python.exe task1_titanic\titanic_prediction.py
```

Place Titanic-Dataset.csv beside titanic_prediction.py before running.
Required libraries: pandas and scikit-learn.

## Output
test_predictions.csv contains the test inputs, actual outcomes and predictions.