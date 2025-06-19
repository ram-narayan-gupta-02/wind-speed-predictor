import pandas as pd
import numpy as np

def calculate_wind_speed(df):
    df["wind_speed"] = np.sqrt(df["uwnd"]**2 + df["vwnd"]**2)
    return df

def create_features(df):
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")
    for lag in [1, 2, 3]:
        df[f"wind_speed_lag_{lag}"] = df["wind_speed"].shift(lag)
    df = df.dropna()
    return df
