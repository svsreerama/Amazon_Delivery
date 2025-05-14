# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Amazon Delivery Time Predictor", layout="centered")
st.title("🚚 Amazon Delivery Time Predictor")

# Load model
model_path = "models/xgboost_model.pkl"

if not os.path.exists(model_path):
    st.error("❌ Trained model not found. Please train the model first.")
    st.stop()

model = joblib.load(model_path)

# Input fields
st.subheader("Enter Delivery Details:")

agent_age = st.number_input("Agent Age", min_value=18, max_value=70, value=30)
agent_rating = st.slider("Agent Rating", min_value=0.0, max_value=5.0, value=4.5, step=0.1)
distance = st.number_input("Distance between Store and Drop (in km)", min_value=0.1, value=5.0, step=0.1)

weather = st.selectbox("Weather", ["Sunny", "Stormy", "Windy", "Cloudy", "Fog", "Sandstorms"])
traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High", "Jam"])
vehicle = st.selectbox("Vehicle Type", ["bike", "scooter", "car", "bicycle", "electric_scooter"])
area = st.selectbox("Area", ["Urban", "Metropolitian"])
category = st.selectbox("Product Category", ["Food", "Grocery", "Flowers", "Drinks", "Electronics", "Clothing", "Pharmacy", "Toys", "Documents"])

# Dummy values for hour and day
hour = 10
dayofweek = 2

# Prepare input DataFrame
input_df = pd.DataFrame({
    "Agent_Age": [agent_age],
    "Agent_Rating": [agent_rating],
    "Distance_km": [distance],
    "Hour": [hour],
    "DayOfWeek": [dayofweek],
    "Weather": [weather],
    "Traffic": [traffic],
    "Vehicle": [vehicle],
    "Area": [area],
    "Category": [category]
})

# Predict button
if st.button("Predict Delivery Time"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🕒 Estimated Delivery Time: **{prediction:.2f} minutes**")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
