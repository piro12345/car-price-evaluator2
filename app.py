import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # Forces Qt to run headless without a display

import streamlit as st
import Orange
import pandas as pd

st.set_page_config(page_title="Used Car Valuation AI", page_icon="🚗", layout="centered")

st.title("🚗 Second-Hand Car Value Evaluator")
st.caption("CBSE Class 12 AI Capstone Project — Trained with Orange Data Mining")
st.divider()

# Load the exact Orange model file
@st.cache_resource
def load_orange_tree():
    return Orange.classification.load_learner("car_model.pkcls")

try:
    model = load_orange_tree()
except Exception as e:
    st.error(f"Error loading Orange model: {e}")
    st.stop()

# Inputs matching Orange Select Columns exactly
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Year", 2005, 2026, 2017)
    present_price = st.number_input("Present_Price (₹ Lakhs)", 0.3, 50.0, 8.5, 0.25)
    kms_driven = st.number_input("Kms_Driven", 500, 250000, 35000, 1000)

with col2:
    fuel_type = st.selectbox("Fuel_Type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Seller_Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Owner", [0, 1, 2, 3])

st.divider()

if st.button("Evaluate Fair Resale Value", use_container_width=True):
    # Construct single-row DataFrame matching the Orange feature domain
    sample_df = pd.DataFrame([{
        "Year": year,
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Fuel_Type": fuel_type,
        "Seller_Type": seller_type,
        "Transmission": transmission,
        "Owner": owner,
        "Selling_Price": 0.0  # Placeholder target required by Orange Table schema
    }])

    # Convert to Orange Table
    sample_df.to_csv("temp_row.csv", index=False)
    orange_table = Orange.data.Table("temp_row.csv")

    # Run direct inference using your saved Orange model
    prediction = float(model(orange_table[0]))
    final_price = max(0.15, prediction)

    st.success(f"### Estimated Fair Resale Value: ₹ {final_price:.2f} Lakhs")
    st.info(f"Market Equivalent: approximately ₹{final_price * 100000:,.0f}")
