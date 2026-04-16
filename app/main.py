import os
import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -----------------------------
# App Initialization
# -----------------------------
app = FastAPI(
    title="Loan Prediction API",
    description="Predicts loan approval using MLflow Model Registry",
    version="2.0.0"
)

MLFLOW_MODEL_URI = "models:/LoanPredictionModel/Production"

model = None


# -----------------------------
# Load Model on Startup
# -----------------------------
@app.on_event("startup")
def load_model():
    global model

    try:
        model = mlflow.pyfunc.load_model(MLFLOW_MODEL_URI)
        print("✅ Model loaded from MLflow registry")

    except Exception as e:
        model = None
        print(f"❌ Failed to load model: {e}")


# -----------------------------
# Input Schema
# -----------------------------
class LoanApplication(BaseModel):
    Gender: str
    Married: str
    Dependents: int
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str


# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def root():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "model_source": "MLflow Registry"
    }


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(application: LoanApplication):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Train and register a model first."
        )

    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([application.model_dump()])

        # Predict
        prediction = model.predict(input_data)[0]

        # Some models may not support predict_proba
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_data)[0][1])

        result = "Approved" if prediction == 1 else "Rejected"

        return {
            "prediction": result,
            "probability_of_approval": round(probability, 4) if probability else None,
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )