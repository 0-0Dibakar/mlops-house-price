from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction():
    payload = {
        "MedInc": 5.0,
        "HouseAge": 20,
        "AveRooms": 5.0,
        "AveBedrms": 1.0,
        "Population": 1000,
        "AveOccup": 3.0,
        "Latitude": 34.0,
        "Longitude": -118.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200