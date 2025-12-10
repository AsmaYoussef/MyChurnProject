from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

from data_preprocessing import preprocess_data

app = FastAPI(title="Churn Prediction API")

model = joblib.load("churn_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(data: CustomerData):
    df = pd.DataFrame([data.dict()])
    processed, _ = preprocess_data(df, preprocessor=preprocessor)
    prediction = model.predict(processed)[0]

    return {"churn": "Yes" if prediction == 1 else "No"}
