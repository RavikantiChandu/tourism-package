import pandas as pd
from sklearn.model_selection import train_test_split
from data_register import clean_data, validate, FEATURES

raw = pd.read_csv("tourism_project/data/tourism.csv")
validate(raw)
df = clean_data(raw)
if df["ProdTaken"].isna().any():
    raise ValueError("Target ProdTaken contains missing values; inspect before training")

X = df[FEATURES].copy()
for col in set(FEATURES) - {"Occupation", "Gender", "MaritalStatus", "Designation"}:
    X[col] = pd.to_numeric(X[col], errors="coerce")
y = df["ProdTaken"].astype(int)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
for name, frame in [("Xtrain", Xtrain), ("Xtest", Xtest), ("ytrain", ytrain), ("ytest", ytest)]:
    frame.to_csv(name + ".csv", index=False)
print("Training:", Xtrain.shape, "Testing:", Xtest.shape)
