import pandas as pd

df = pd.read_csv("loan_data.csv")

print("=== Shape ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n=== Column Names ===")
for col in df.columns:
    print(f"  {col}")

print("\n=== First 5 Rows ===")
print(df.head())

print("\n=== Data Types ===")
print(df.dtypes)

print("\n=== Missing Values Before Cleaning ===")
print(df.isnull().sum())

# --- Data Cleaning ---

numerical_cols = ["LoanAmount", "Loan_Amount_Term", "Credit_History"]
categorical_cols = ["Gender", "Married", "Self_Employed", "Dependents"]

fill_values = {}

for col in numerical_cols:
    median = df[col].median()
    fill_values[col] = ("median", median)
    df[col] = df[col].fillna(median)

for col in categorical_cols:
    mode = df[col].mode()[0]
    fill_values[col] = ("mode", mode)
    df[col] = df[col].fillna(mode)

print("\n=== Fill Value Summary ===")
print(f"{'Column':<20} {'Method':<8} {'Value'}")
print("-" * 40)
for col, (method, value) in fill_values.items():
    print(f"{col:<20} {method:<8} {value}")

print("\n=== Missing Values After Cleaning ===")
missing_after = df.isnull().sum()
print(missing_after)
print(f"\nTotal missing values remaining: {missing_after.sum()}")

df.to_csv("loan_data_cleaned.csv", index=False)
print("\nCleaned data saved to loan_data_cleaned.csv")
