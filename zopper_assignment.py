# =====================================================
# Zopper | Data Science Internship Assignment
# =====================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns


# -----------------------------------------------------
# 1. Load Dataset
# -----------------------------------------------------
FILE_PATH = r"D:\ML project\zopper_ass\Jumbo & Company_ Attach  .xlsx"  # update path if required

df = pd.read_excel(FILE_PATH)
df.columns = df.columns.map(str).str.strip()

print("Dataset Loaded Successfully")
print(df.head())

# -----------------------------------------------------
# 2. Identify Columns
# -----------------------------------------------------
id_cols = df.columns[:3]        # Store / Branch identifiers
month_cols = df.columns[3:]     # Monthly attach %

# -----------------------------------------------------
# 3. Reshape Data for ML
# -----------------------------------------------------
# Convert wide format (months) to long format

long_df = df.melt(
    id_vars=id_cols,
    value_vars=month_cols,
    var_name="Month",
    value_name="Attach_Percentage"
)

# Create time index
long_df["Month_Index"] = long_df.groupby(id_cols[0]).cumcount()

print("\nLong Format Data:")
print(long_df.head())

# -----------------------------------------------------
# 4. Train Linear Regression Model (per store)
# -----------------------------------------------------
predictions = []

for store in long_df[id_cols[0]].unique():
    store_data = long_df[long_df[id_cols[0]] == store]

    X = store_data[["Month_Index"]]
    y = store_data["Attach_Percentage"]

    model = LinearRegression()
    model.fit(X, y)

    # Predict for next month (January)
    next_month_index = [[store_data["Month_Index"].max() + 1]]
    jan_pred = model.predict(next_month_index)[0]

    predictions.append({
        id_cols[0]: store,
        "January_Prediction": round(jan_pred, 2)
    })

prediction_df = pd.DataFrame(predictions)

# -----------------------------------------------------
# 5. Merge Predictions with Original Data
# -----------------------------------------------------
df["Historical_Average"] = df[month_cols].mean(axis=1)

final_df = df.merge(prediction_df, on=id_cols[0])

# -----------------------------------------------------
# 6. Visualization – Prediction vs History
# -----------------------------------------------------
# Correct scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='Historical_Average',
    y='January_Prediction',
    data=final_df,   # use final_df
    s=100,
    color='blue'
)

# Add diagonal line for reference
max_val = max(final_df['Historical_Average'].max(), final_df['January_Prediction'].max())
plt.plot([0, max_val], [0, max_val], 'r--', label='Ideal Match')

plt.title('January Attach % Prediction vs Historical Performance')
plt.xlabel('Historical Average Attach %')
plt.ylabel('Predicted January Attach %')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------------------------------
# 7. Store Categorization
# -----------------------------------------------------
final_df["Performance_Category"] = pd.qcut(
    final_df["Historical_Average"],
    q=3,
    labels=["Low Performer", "Medium Performer", "High Performer"]
)

print("\nPerformance Category Distribution:")
print(final_df["Performance_Category"].value_counts())

# -----------------------------------------------------
# 8. Final Output Table
# -----------------------------------------------------
output_cols = list(id_cols) + [
    "Historical_Average",
    "January_Prediction",
    "Performance_Category"
]

result = final_df[output_cols]

print("\nFinal Output Preview:")
print(result.head())

# -----------------------------------------------------
# 9. Save Output
# -----------------------------------------------------
result.to_excel(
    "Zopper_Attach_Analysis_Output_V4_ML.xlsx",
    index=False
)

