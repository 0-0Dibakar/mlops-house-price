import joblib

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(
    title="House Price Prediction API",
    version="1.0.0"
)


model = joblib.load(
    "models/house_price_model.pkl"
)


class HouseInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


@app.get("/")
def home():
    return {
        "message": "House Price Prediction API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: HouseInput):

    features = [[
        data.MedInc,
        data.HouseAge,
        data.AveRooms,
        data.AveBedrms,
        data.Population,
        data.AveOccup,
        data.Latitude,
        data.Longitude
    ]]

    prediction = model.predict(features)

    return {
        "predicted_house_value": float(prediction[0])
    }