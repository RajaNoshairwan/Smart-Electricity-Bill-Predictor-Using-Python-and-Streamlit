import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Electricity Bill Predictor", page_icon="⚡", layout="centered")

st.markdown("<h1 style='text-align: center;'>⚡ Smart Electricity Bill Predictor</h1>", unsafe_allow_html=True)
st.caption("📊 Predict daily electricity consumption & billing using a trained Machine Learning model (Linear Regression).")

@st.cache_data
def load_data():
    return pd.read_csv("electricity_usage_dataset.csv")

data = load_data()

X = data[['Fans', 'Lights', 'ACs', 'Refrigerator', 'Water_Pump', 'Temperature', 'Time']]
y = data['Consumption_kWh']

model = LinearRegression()
model.fit(X, y)
mse = mean_squared_error(y, model.predict(X))

st.sidebar.header("🔌 Enter Your Usage Details")

fans = st.sidebar.slider("🌀 Fans Running", 0, 10, 2, help="Number of ceiling or pedestal fans currently in use.")
lights = st.sidebar.slider("💡 Lights On", 0, 20, 5, help="Total lights turned on in your home.")
acs = st.sidebar.slider("❄️ ACs On", 0, 3, 1, help="Air Conditioners currently running.")
fridge = st.sidebar.radio("🧊 Refrigerator", ["On", "Off"])
pump = st.sidebar.radio("🚿 Water Pump", ["On", "Off"])
temp = st.sidebar.slider("🌡️ Temperature (°C)", 15, 45, 30, help="Current outside temperature.")
time = st.sidebar.slider("⏰ Time of Day (Hour)", 0, 23, 14, help="Current hour in 24-hour format.")

fridge_val = 1 if fridge == "On" else 0
pump_val = 1 if pump == "On" else 0

if st.sidebar.button("🔍 Predict My Bill"):
    input_data = np.array([[fans, lights, acs, fridge_val, pump_val, temp, time]])
    predicted_kwh = model.predict(input_data)[0]
    rate_per_unit = 55  
    estimated_bill = predicted_kwh * rate_per_unit

    st.markdown("## 🔎 Prediction Result")
    st.success(f"**Estimated Daily Electricity Consumption:** `{predicted_kwh:.2f} kWh`")
    st.info(f"**Estimated Daily Bill:** `Rs. {estimated_bill:.2f}`")
    st.warning(f"📉 Model Accuracy (MSE): `{mse:.3f}`")

    # Optional: Monthly Estimate
    monthly_bill = estimated_bill * 30
    st.markdown(f"💸 **Estimated Monthly Bill:** `Rs. {monthly_bill:.0f}`")

st.markdown("---")
st.caption("🧠 Built using Streamlit & Scikit-Learn | Developed by Noshairwan — A practical ML project")
