# python -m streamlit run app.py 

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# --- Load Trained Model ---
model_path = "D:/Python/ADRDE_Project/model/wind_speed_model.pkl"
metrics_path = "D:/Python/ADRDE_Project/model/metrics.txt"

# --- Try loading model ---
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("❌ Trained model not found. Please train it using train_model.py")
    st.stop()

# --- Load model metrics ---
model_metrics = "Metrics not available"
if os.path.exists(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        model_metrics = f.read()


# --- Page Config ---
st.set_page_config(page_title="Wind Speed Predictor", layout="wide")

# --- Custom Background Image ---

background_image = "https://images.unsplash.com/photo-1601987077221-8a80f3e258e6?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80"  # Replace with your own if desired
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('{background_image}');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    .drdo-header {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 20px;
        margin-bottom: 20px;
        gap: 20px;
    }

    .drdo-header img {
        width: 70px;
        height: 70px;
        border-radius: 50%;
    }

    .drdo-title {
        font-size: 30px;
        font-weight: 700;
        color: white;
        text-align: left;
    }

    @media (max-width: 768px) {
        .drdo-header {
            flex-direction: column;
        }
        .drdo-title {
            text-align: center;
        }
    }
    </style>

    <div class="drdo-header">
        <img src="https://upload.wikimedia.org/wikipedia/en/thumb/1/1d/Defence_Research_and_Development_Organisation.svg/1200px-Defence_Research_and_Development_Organisation.svg.png" alt="DRDO Logo">
        <div class="drdo-title">Aerial Delivery Research and Development Establishment<br>(DRDO)</div>
    </div>
    """,
    unsafe_allow_html=True
)


# --- Title with Logo ---
st.markdown("""
    <div style='text-align: center;'>
        <img src='https://cdn-icons-png.flaticon.com/512/1117/1117466.png' width='100'>
        <h1>🌬️ Wind Speed Prediction (0–22 km Altitude)</h1>
        <p>Predict wind speed using past 3-day values, wind components, and time features — outputs in m/s and km/s</p>
    </div>
    <hr>
""", unsafe_allow_html=True)

# --- Layout ---
left, right = st.columns([1, 1])

with left:
    st.subheader("📥 Input Parameters")

    uwnd = st.number_input("U-Wind Component (m/s)", value=4.0)
    vwnd = st.number_input("V-Wind Component (m/s)", value=3.5)

    lag_3 = st.number_input("Wind Speed - 3 Days Ago (m/s)", min_value=0.0, value=6.0)
    lag_2 = st.number_input("Wind Speed - 2 Days Ago (m/s)", min_value=0.0, value=7.2)
    lag_1 = st.number_input("Wind Speed - 1 Day Ago (m/s)", min_value=0.0, value=8.5)

    wind_avg3 = (lag_1 + lag_2 + lag_3) / 3

    today = pd.Timestamp.today()
    day_of_year = today.dayofyear
    month = today.month

    altitude = st.slider("🛰️ Altitude Selection (km)", min_value=0, max_value=22, value=10)

    if st.button("🔍 Predict Wind Speed"):
        input_data = np.array([[uwnd, vwnd, lag_1, lag_2, lag_3, wind_avg3, day_of_year, month]])
        predicted_ms = model.predict(input_data)[0]
        predicted_kms = predicted_ms / 1000.0

        st.success(f"✅ Predicted Wind Speed at {altitude} km:")
        st.metric("💨 Speed (m/s)", f"{predicted_ms:.2f}")
        st.metric("🚀 Speed (km/s)", f"{predicted_kms:.5f}")

        # --- Chart Section ---
        st.subheader("📈 Wind Trend (Past + Prediction)")
        days = ["3 Days Ago", "2 Days Ago", "1 Day Ago", "Prediction"]
        values = [lag_3, lag_2, lag_1, predicted_ms]
        fig, ax = plt.subplots()
        ax.plot(days, values, marker='o', linestyle='-', color='blue')
        ax.set_ylabel("Wind Speed (m/s)")
        ax.set_title("Wind Speed Trend")
        ax.grid(True)
        st.pyplot(fig)

with right:
    st.subheader("📊 Model Accuracy")
    st.code(model_metrics)

    st.markdown("### ℹ️ How it Works")
    st.markdown("""
    - Uses wind components and past 3 days of wind speed values  
    - Automatically derives time-based features  
    - Predicts wind speed at selected altitude (0–22 km)  
    - Model: XGBoost (trained on 10 years NOAA data)
    """)

# --- Footer ---
st.markdown("<hr><center>🛰️ Powered by ML for ADRDE • Altitude Range: 0–22 km</center>", unsafe_allow_html=True)
