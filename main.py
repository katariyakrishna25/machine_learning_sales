import os
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("saleslinear.pkl")


class SalesInput(BaseModel):
    AdvertisingBudget: float


@app.get("/")
def home():
    return {"msg": "uvicorn is running"}


@app.post("/predsales")
def datainsert(ds: SalesInput):
    prediction = model.predict([[ds.AdvertisingBudget]])
    return {"predicted_sales": round(float(prediction[0]), 2)}
