import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Weather Rainfall Prediction API",
    version="1.0"
)

model = joblib.load("model.pkl")


class WeatherInput(BaseModel):
    temperature: float
    humidity: float
    wind_speed: float
    wind_direction: float
    atmospheric_pressure: float
    cloud_cover: float
    timestamp: str
    humidity_lag_3: float
    humidity_lag_6: float
    pressure_lag_3: float
    pressure_lag_6: float
    cloud_lag_3: float
    cloud_lag_6: float


@app.get("/")
def home():
    return {
        "message": "Weather Rainfall Prediction API is running"
    }


@app.post("/predict")
def predict(data: WeatherInput):

    timestamp = pd.to_datetime(data.timestamp)

    month = timestamp.month
    day = timestamp.day
    day_of_week = timestamp.dayofweek
    hour = timestamp.hour
    day_of_year = timestamp.dayofyear

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    day_of_year_sin = np.sin(
        2 * np.pi * day_of_year / 365.25
    )

    day_of_year_cos = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    wind_direction_sin = np.sin(
        np.deg2rad(data.wind_direction)
    )

    wind_direction_cos = np.cos(
        np.deg2rad(data.wind_direction)
    )

    humidity_change_3 = (
        data.humidity - data.humidity_lag_3
    )

    pressure_change_3 = (
        data.atmospheric_pressure - data.pressure_lag_3
    )

    cloud_change_3 = (
        data.cloud_cover - data.cloud_lag_3
    )

    features = pd.DataFrame([{
        "Temperature": data.temperature,
        "Humidity": data.humidity,
        "Wind_Speed": data.wind_speed,
        "Wind_Direction_sin": wind_direction_sin,
        "Wind_Direction_cos": wind_direction_cos,
        "Atmospheric_Pressure": data.atmospheric_pressure,
        "Cloud_Cover": data.cloud_cover,
        "Month": month,
        "Day": day,
        "DayOfWeek": day_of_week,
        "Hour_sin": hour_sin,
        "Hour_cos": hour_cos,
        "DayOfYear_sin": day_of_year_sin,
        "DayOfYear_cos": day_of_year_cos,
        "Humidity_Lag_3": data.humidity_lag_3,
        "Humidity_Lag_6": data.humidity_lag_6,
        "Pressure_Lag_3": data.pressure_lag_3,
        "Pressure_Lag_6": data.pressure_lag_6,
        "Cloud_Lag_3": data.cloud_lag_3,
        "Cloud_Lag_6": data.cloud_lag_6,
        "Humidity_Change_3": humidity_change_3,
        "Pressure_Change_3": pressure_change_3,
        "Cloud_Change_3": cloud_change_3
    }])

    prediction = model.predict(features)[0]
    prediction = max(float(prediction), 0.0)

    return {
        "predicted_rainfall_mm": round(prediction, 3),
        "timestamp": data.timestamp
    }
