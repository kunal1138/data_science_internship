# Task 2: Unemployment Analysis
# Step 1: Load and inspect the dataset

from pathlib import Path
import pandas as pd

dataset_path = Path(__file__).parent / "Unemployment in India.csv"
df = pd.read_csv(dataset_path)

# Remove extra spaces around column names.
df.columns = df.columns.str.strip()

print("\nFIRST 5 ROWS:")
print(df.head().to_string(index=False))

print("\nDATASET SIZE (rows, columns):")
print(df.shape)

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nMISSING VALUES:")
print(df.isna().sum())

print("\nDUPLICATE ROWS:")
print(df.duplicated().sum())

# Step 2: Clean the dataset
print("\nCOMPLETELY EMPTY ROWS:", df.isna().all(axis=1).sum())

# Remove empty rows before checking for duplicate records.
df = df.dropna(how="all").copy()

for column in ["Region", "Date", "Frequency", "Area"]:
    df[column] = df[column].str.strip()

# Convert dates into a format Python can analyse.
df["Date"] = pd.to_datetime(
    df["Date"], format="%d-%m-%Y", errors="coerce"
)

duplicates = df.duplicated().sum()
print("DUPLICATE RECORDS AFTER REMOVING EMPTY ROWS:", duplicates)
df = df.drop_duplicates()

# Sort records into chronological order within each region and area.
df = df.sort_values(["Region", "Area", "Date"]).reset_index(drop=True)

print("\nCLEANED DATASET SIZE:", df.shape)
print("\nREMAINING MISSING VALUES:")
print(df.isna().sum())

print("\nDATE RANGE:")
print(df["Date"].min(), "to", df["Date"].max())

print("\nNUMBER OF REGIONS:", df["Region"].nunique())
print("\nRECORDS PER AREA:")
print(df["Area"].value_counts())

print("\nUNEMPLOYMENT RATE SUMMARY:")
print(df["Estimated Unemployment Rate (%)"].describe())

# Step 3: Summarise and plot monthly unemployment trends
import matplotlib.pyplot as plt

rate_column = "Estimated Unemployment Rate (%)"

monthly_summary = df.groupby("Date").agg(
    Average_Rate=(rate_column, "mean"),
    Record_Count=(rate_column, "count")
)

print("\nMONTHLY UNEMPLOYMENT SUMMARY:")
print(monthly_summary.round(2))

# Compare rural and urban averages for each month.
area_trends = df.groupby(["Date", "Area"])[rate_column].mean().unstack()

ax = area_trends.plot(
    figsize=(10, 5),
    marker="o"
)

ax.set_title("Monthly Unemployment Trends: Rural and Urban")
ax.set_xlabel("Month")
ax.set_ylabel("Mean unemployment rate (%)")
ax.grid(alpha=0.3)

plt.figtext(
    0.5, 0.01,
    "Unweighted averages of available regional records; coverage varies by month.",
    ha="center",
    fontsize=9
)
plt.tight_layout(rect=[0, 0.05, 1, 1])

output_folder = Path(__file__).parent
plt.savefig(output_folder / "monthly_unemployment_trends.png", dpi=150)
plt.close()

monthly_summary.to_csv(output_folder / "monthly_summary.csv")

print("\nSaved monthly_unemployment_trends.png and monthly_summary.csv")

# Step 4: Compare periods using consistent regional coverage
# Baseline: January–February 2020
# Comparison: April–May 2020
# March is excluded as a transition month.

selected_months = pd.to_datetime([
    "2020-01-31", "2020-02-29",
    "2020-04-30", "2020-05-31"
])

comparison = df.pivot_table(
    index=["Region", "Area"],
    columns="Date",
    values=rate_column,
    aggfunc="mean"
)

# Keep only groups with data in all four selected months.
comparison = comparison.reindex(columns=selected_months).dropna()

if comparison.empty:
    print("\nNot enough matching records for the period comparison.")
else:
    comparison["Jan-Feb mean (%)"] = comparison[
        selected_months[:2]
    ].mean(axis=1)

    comparison["Apr-May mean (%)"] = comparison[
        selected_months[2:]
    ].mean(axis=1)

    comparison["Change (percentage points)"] = (
        comparison["Apr-May mean (%)"]
        - comparison["Jan-Feb mean (%)"]
    )

    results = comparison[[
        "Jan-Feb mean (%)",
        "Apr-May mean (%)",
        "Change (percentage points)"
    ]]

    print("\nMATCHED REGION–AREA GROUPS:", len(results))
    print("\nAVERAGE PERIOD COMPARISON:")
    print(results.mean().round(2))

    print("\nTOP 5 INCREASES:")
    print(
        results.sort_values(
            "Change (percentage points)", ascending=False
        ).head(5).round(2)
    )

    results.to_csv(output_folder / "period_comparison.csv")
    print("\nSaved period_comparison.csv")