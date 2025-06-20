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

- Model: **XGBoost**
- Input Features: Past 3-day wind speeds (lag values)
- Target: Wind speed calculated using  
  `wind_speed = √(uwnd² + vwnd²)`
- Performance:
  - **MSE (Mean Squared Error)**: ~0.0804
  - **R² Score**: ~0.9947 (baseline)
  - **MAE (Mean Absolute Error)**: ~0.0890

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
```
wind-speed-predictor/
├── app/                           # Streamlit application
│   └── app.py                     # Main web app interface
│
├── model/                         # ML model training and output
│   ├── train_model.py             # Python script to train model
│   ├── wind_speed_model.pkl       # Trained ML model (joblib)
│   └── metrics.txt                # Evaluation metrics (MSE, R²)
│
├── wind_data_combined_2016_2025.csv   # NOAA historical wind data (2016–2025)
├── requirements.txt              # List of Python dependencies
└── README.md                     # Project documentation
```
---

## 🧪 How to Run Locally

```bash
git clone https://github.com/ram-narayan-gupta-02/wind-speed-predictor.git
cd wind-speed-predictor
pip install -r requirements.txt
python model/train_model.py    # Optional if model already trained
streamlit run app/app.py
```
---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing
Feel free to contribute by submitting a pull request or reporting issues!

---

## 📩 Contact
📧 Email: [ramnrngupta@gmail.com](mailto:ramnrngupta@gmail.com)
📌 GitHub: [ram-narayan-gupta-02](https://github.com/ram-narayan-gupta-02)
