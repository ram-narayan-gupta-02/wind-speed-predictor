import pandas as pd
import joblib
from utils.preprocess import calculate_wind_speed, create_features

def predict_from_new_data(filepath):
    model = joblib.load("model/wind_speed_model.pkl")
    
    df = pd.read_csv(filepath)
    df = calculate_wind_speed(df)
    df = create_features(df)

    features = [col for col in df.columns if "lag" in col]
    predictions = model.predict(df[features])
    
    df["predicted_wind_speed"] = predictions
    return df[["time", "predicted_wind_speed"]]
