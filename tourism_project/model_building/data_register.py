import numpy as np
import pandas as pd

MISSING = ["NA", "N/A", "null", "NULL", "None", "?", "unknown", "Unknown", "-"]

def clean_data(df):
    df = df.copy()
    df = df.replace(r"^\s*$", np.nan, regex=True).replace(MISSING, np.nan)
    df = df.drop_duplicates()
    df["Gender"] = df["Gender"].astype("string").str.strip().replace({"Fe Male": "Female", "FeMale": "Female"})
    df["MaritalStatus"] = df["MaritalStatus"].astype("string").str.strip().replace({"Unmarried": "Single"})
    visitors = pd.to_numeric(df["NumberOfPersonVisiting"], errors="coerce").replace(0, np.nan)
    df["IncomePerVisitor"] = pd.to_numeric(df["MonthlyIncome"], errors="coerce") / visitors
    return df

from pathlib import Path
RAW = Path("tourism_project/data/tourism.csv")
df = pd.read_csv(RAW)
required = {"ProdTaken", "Gender", "MaritalStatus", "MonthlyIncome", "NumberOfPersonVisiting"}
missing = required - set(df.columns)
if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
print("Dataset registered:", df.shape)
print("Nulls:", df.isna().sum().to_dict())
print("Exact duplicates:", df.duplicated().sum())
print("Target distribution:", df["ProdTaken"].value_counts().to_dict())
