# 🌬️ Wind Speed Prediction using Machine Learning (ADRDE – DRDO Project)

This AI-powered project predicts wind speed in the **altitude range of 0–22 km** using **NOAA atmospheric data (2016–2025)**. Developed as part of a final-year engineering project at **RBS Engineering Technical Campus, Agra**, the model serves critical applications in **aerospace, missile technology, and atmospheric research** under the supervision of **ADRDE – DRDO**.

---

## 📌 Project Objective

To build a machine learning model that can:
- Accurately **predict wind speed** at different altitudes
- Use historical **uwnd (zonal)** and **vwnd (meridional)** wind component data
- Forecast wind speed in **m/s and km/s**
- Help in decision-making for **high-altitude aerospace missions**

---

## 📂 Data Source

- **NOAA (National Oceanic and Atmospheric Administration)**
- Data Range: **2016 to 2025**
- Features used:
  - `time` (date/time of measurement)
  - `uwnd` (zonal wind speed component)
  - `vwnd` (meridional wind speed component)

---

## 🧠 Machine Learning Approach

- Model: **Random Forest Regressor**
- Input Features: Past 3-day wind speeds (lag values)
- Target: Wind speed calculated using  
  `wind_speed = √(uwnd² + vwnd²)`
- Performance:
  - **MSE (Mean Squared Error)**: ~14.71
  - **R² Score**: ~0.015 (baseline)

---

## 🖥️ Web App Features

Developed with **Streamlit**, the web app allows:

- 📥 **Input**: Last 3 days' wind speed
- 📊 **Output**: Predicted wind speed (m/s and km/s)
- 📈 **Visualization**: Wind speed trend chart
- 📡 **Altitude selection**: 0 to 22 km

---

## 🚀 Live Demo

Try the live model here:  
🔗 [Streamlit App](https://wind-speed-predictor-vktvgzhb6f2wwiwmgcdnu3.streamlit.app/)

Website version:  
🌐 [GitHub Pages](https://ram-narayan-gupta-02.github.io/windspeedpredictor/)

---

## 📁 Folder Structure

wind-speed-predictor/ 
├── app/ 
│   └── app.py         # Streamlit frontend 
├── model/ 
    └── train_model.py # ML model training script 
    └── wind_speed_model.pkl # Saved model 
    └── metrics.txt    # Model evaluation metrics 
├── wind_data_combined_2016_2025.csv 
├── requirements.txt 
└── README.md

---

## 🧪 How to Run Locally

```bash
git clone https://github.com/ram-narayan-gupta-02/wind-speed-predictor.git
cd wind-speed-predictor
pip install -r requirements.txt
python model/train_model.py    # Optional if model already trained
streamlit run app/app.py