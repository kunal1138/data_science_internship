# Task 2: Unemployment Analysis with Python

## Objective
Clean and visualise unemployment data, examine changes around
the COVID-19 period, and identify patterns and policy considerations.

## Dataset
Source: https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india

- Period: May 2019–June 2020
- Regions: 28
- Cleaned records: 740
- Area categories: Rural and Urban

## Data Cleaning
- Removed 28 completely empty rows.
- Removed extra spaces from column names and text values.
- Converted dates into datetime format.
- Found no duplicate records or missing values after cleaning.

## Key Findings
- The monthly unweighted average reached 24.88% in May 2020,
  then declined to 11.90% in June.
- For 47 region–area groups with data in all four selected months,
  the average rose from 9.70% in January–February to 24.87%
  in April–May 2020.
- This represents an increase of 15.17 percentage points.
- Puducherry Urban had the largest increase among matched groups,
  followed by Tamil Nadu Rural and Jharkhand Urban.

## Interpretation and Limitations
- The increase coincided with the COVID-19 period, but this
  descriptive comparison does not establish causation.
- Averages are unweighted regional averages, not official
  national unemployment rates.
- Monthly coverage varies. The period comparison uses the
  same 47 groups to reduce the effect of changing coverage.
- Fourteen months of data is insufficient to reliably establish
  recurring seasonal patterns.
- Extreme changes should be checked against the original
  source before informing decisions.

## Policy Considerations
These are suggestions for further investigation:
- Assess temporary income support and employment programmes
  in areas showing the largest increases.
- Examine local labour-market needs before designing training
  and job-placement programmes.
- Monitor rural and urban trends separately and improve
  consistency of regional reporting.

The analysis does not test the effectiveness of these measures.

## Run
From the data_science_internship folder:

```powershell
.\.venv\Scripts\python.exe task2_unemployment\unemployment_analysis.py
```

Required libraries: pandas and matplotlib.
Place Unemployment in India.csv beside the Python script.

## Outputs
- monthly_unemployment_trends.png: Rural and urban trend chart
- monthly_summary.csv: Monthly averages and record counts
- period_comparison.csv: Matched-group period comparisons