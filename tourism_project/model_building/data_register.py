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
EXPECTED = ['CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier', 'Occupation', 'Gender', 'NumberOfPersonVisiting', 'PreferredPropertyStar', 'MaritalStatus', 'NumberOfTrips', 'Passport', 'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome', 'PitchSatisfactionScore', 'ProductPitched', 'NumberOfFollowups', 'DurationOfPitch']
FEATURES = ['Age', 'CityTier', 'Occupation', 'Gender', 'NumberOfPersonVisiting', 'PreferredPropertyStar', 'MaritalStatus', 'NumberOfTrips', 'Passport', 'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome', 'IncomePerVisitor']
RAW = Path("tourism_project/data/tourism.csv")
def validate(df):
    missing = set(EXPECTED) - set(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.CustomerID.isna().any() or df.CustomerID.duplicated().any():
        raise ValueError("Missing or duplicate CustomerID")
    if not df.ProdTaken.isin([0, 1]).all() or df.ProdTaken.nunique() != 2:
        raise ValueError("Target must contain both binary classes and no missing values")
    print("Dataset registered:", df.shape)
    print("Nulls:", df.isna().sum().to_dict())
    print("Target distribution:", df.ProdTaken.value_counts().to_dict())
if __name__ == "__main__":
    validate(pd.read_csv(RAW))
