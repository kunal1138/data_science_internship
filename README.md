# Data Science Internship — SAM AI Technologies

Three projects covering classification, data analysis and regression.

## Projects

| Task | Project | Main result |
|---|---|---|
| 1 | [Titanic Survival Prediction](task1_titanic/) | 80.45% test accuracy |
| 2 | [Unemployment Analysis](task2_unemployment/) | Monthly trends and matched-period comparisons |
| 3 | [Housing Price Prediction](task3_housing/) | Test R² of 0.5543 |

### Task 1: Titanic Survival Prediction
- Built a logistic regression classifier.
- Handled missing values, encoded categories and scaled numerical inputs.
- Fitted preprocessing only on training data.
- Evaluated on 179 test passengers.
- Saved actual and predicted survival outcomes.

### Task 2: Unemployment Analysis
- Cleaned 740 records covering May 2019–June 2020.
- Visualised rural and urban unemployment trends.
- Compared 47 matched region–area groups.
- Average unemployment rose from 9.70% in January–February
  to 24.87% in April–May 2020: a 15.17 percentage-point increase.

These are unweighted regional averages, not official national rates.
The comparison does not establish causation, and 14 months is
insufficient to reliably identify recurring seasonal patterns.

### Task 3: Housing Price Prediction
- Used living area to predict sale price with simple linear regression.
- Applied MinMaxScaler fitted only on training data.
- Evaluated on 292 test houses.
- MAE: $38,341.20
- RMSE: $58,471.76
- R²: 0.5543

Living area alone does not capture every factor affecting house prices.

## Tools
Python, pandas, scikit-learn, matplotlib, Git and GitHub.

## Setup on Windows

Clone the repository and enter its folder:

```powershell
git clone https://github.com/kunal1138/data_science_internship.git
cd data_science_internship
```

Create an environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Datasets
- [Titanic dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- [Unemployment in India](https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india)
- [Housing dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

For Task 3, download and extract the housing dataset, then place
train.csv inside task3_housing. This file is not included in the repository.

## Run the Projects

Run each command separately from the repository's main folder:

```powershell
.\.venv\Scripts\python.exe task1_titanic\titanic_prediction.py
```

```powershell
.\.venv\Scripts\python.exe task2_unemployment\unemployment_analysis.py
```

```powershell
.\.venv\Scripts\python.exe task3_housing\housing_prediction.py
```

Each task folder contains its own README with methods,
results and output descriptions.