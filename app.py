import pickle
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import streamlit as st

st.set_page_config(page_title="Used Car Valuation AI", page_icon="🚗")

st.title("🚗 Second-Hand Car Value Evaluator")
st.caption(
    "CBSE Class 12 AI Capstone Project — UN SDG Goal 8: Decent Work & Economic"
    " Growth"
)


# Load the internal trained tree model from your .pkcls file
@st.cache_resource
def load_trained_tree():
  with open("car model.pkcls", "rb") as f:
    saved_obj = pickle.load(f)

  # Check if saved object is the scikit-learn model or wrapped in Orange's learner container
  if hasattr(saved_obj, "skl_model"):
    return saved_obj.skl_model
  elif hasattr(saved_obj, "model"):
    return saved_obj.model
  return saved_obj


try:
  model = load_trained_tree()
except Exception as e:
  st.error(
      f"Unable to load 'car_model.pkcls'. Ensure the file is placed in the"
      f" repository root. Details: {e}"
  )
  st.stop()

# Inputs matching the features from your Orange Select Columns widget
col1, col2 = st.columns(2)

with col1:
  year = st.number_input("Year of Purchase", 2005, 2026, 2017)
  present_price = st.number_input(
      "Original Showroom Price (₹ Lakhs)", 0.3, 50.0, 8.5, 0.25
  )
  kms_driven = st.number_input("Kilometers Driven", 500, 250000, 35000, 1000)

with col2:
  fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
  seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
  transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
  owner = st.selectbox("Previous Owners", [0, 1, 2, 3])

if st.button("Evaluate Resale Value", use_container_width=True):
  # Map user inputs into numerical feature values
  fuel_diesel = 1 if fuel_type == "Diesel" else 0
  fuel_petrol = 1 if fuel_type == "Petrol" else 0
  seller_ind = 1 if seller_type == "Individual" else 0
  trans_manual = 1 if transmission == "Manual" else 0

  # Build the input row
  features = [
      year,
      present_price,
      kms_driven,
      owner,
      fuel_diesel,
      fuel_petrol,
      seller_ind,
      trans_manual,
  ]

  try:
    pred = float(model.predict([features])[0])
  except Exception:
    # If the model expects a DataFrame with names:
    df_in = pd.DataFrame([{
        "Year": year,
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Owner": owner,
        "Fuel_Type_Diesel": fuel_diesel,
        "Fuel_Type_Petrol": fuel_petrol,
        "Seller_Type_Individual": seller_ind,
        "Transmission_Manual": trans_manual,
    }])
    pred = float(model.predict(df_in)[0])

  predicted_price = max(0.15, pred)

  st.success(f"### Estimated Fair Resale Value: ₹ {predicted_price:.2f} Lakhs")
  st.info(f"Market Equivalent: approximately ₹{predicted_price * 100000:,.0f}")
