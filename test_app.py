from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Weather Rainfall Prediction API is running"


def test_predict():
    data = {
        "temperature": 28.5,
        "humidity": 80.0,
        "wind_speed": 10.0,
        "wind_direction": 180.0,
        "atmospheric_pressure": 1005.0,
        "cloud_cover": 75.0,
        "timestamp": "2026-09-20 14:00:00",
        "humidity_lag_3": 78.0,
        "humidity_lag_6": 76.0,
        "pressure_lag_3": 1006.0,
        "pressure_lag_6": 1007.0,
        "cloud_lag_3": 70.0,
        "cloud_lag_6": 65.0
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200
    assert "predicted_rainfall_mm" in response.json()

