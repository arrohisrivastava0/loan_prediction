import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# initialise app
app = FastAPI(
    title="Loan Prediction API",
    description="Predicts loan approval based on applicant details",
    version="1.0.0"
)

# load model once at startup — not on every request
with open("artifacts/best_model.pkl", "rb") as f:
    model = pickle.load(f)


# input schema — defines exactly what the API expects
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


# health check endpoint — tells you the API is running
@app.get("/")
def root():
    return {"status": "online", "model": "loan-prediction-v1"}


# prediction endpoint
@app.post("/predict")
def predict(application: LoanApplication):
    # convert input to dataframe — same format model was trained on
    input_data = pd.DataFrame([application.model_dump()])

    # get prediction and probability
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # model outputs 0/1 — convert back to human readable
    result = "Approved" if prediction == 1 else "Rejected"

    return {
        "prediction": result,
        "probability_of_approval": round(float(probability), 4),
        "status": "success"
    }