from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.title("Tourism Package Purchase Prediction")
model_path = Path(__file__).with_name("tourism_purchase_model.joblib")
if not model_path.exists():
    st.error("Trained model not found. Run the GitHub Actions pipeline first.")
    st.stop()
model = joblib.load(model_path)

# Input schema comes from the fitted training pipeline; collect all raw model features.
features = model.feature_names_in_.tolist()
num_columns = set(model.named_steps["preprocessor"].transformers_[0][2])
choices = {
    "Gender": ["Male", "Female"], "MaritalStatus": ["Married", "Single", "Divorced"],
    "TypeofContact": ["Self Enquiry", "Company Invited"],
    "Occupation": ["Salaried", "Small Business", "Large Business", "Free Lancer"],
    "Designation": ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
    "ProductPitched": ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"],
}
st.caption("Enter customer details to estimate purchase likelihood (ProdTaken).")
values = {}
for name in features:
    if name == "IncomePerVisitor":
        continue
    if name in choices:
        values[name] = st.selectbox(name, choices[name])
    elif name in num_columns:
        values[name] = st.number_input(name, value=0.0, step=1.0)
    else:
        values[name] = st.text_input(name, value="")
if "IncomePerVisitor" in features:
    visitors = values.get("NumberOfPersonVisiting", 0)
    values["IncomePerVisitor"] = values.get("MonthlyIncome", 0) / visitors if visitors else None
if st.button("Predict Purchase"):
    sample = pd.DataFrame([{name: values.get(name) for name in features}])
    probability = model.predict_proba(sample)[0, 1]
    st.metric("Estimated purchase probability", f"{probability:.1%}")
    st.write("Likely to purchase" if probability >= 0.5 else "Unlikely to purchase")
