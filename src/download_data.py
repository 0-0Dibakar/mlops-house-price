from sklearn.datasets import fetch_california_housing
import pandas as pd
import os


print("Downloading California Housing dataset...")

housing = fetch_california_housing(as_frame=True)

df = housing.frame

os.makedirs("data", exist_ok=True)

df.to_csv("data/housing.csv", index=False)

print("Dataset downloaded successfully!")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())