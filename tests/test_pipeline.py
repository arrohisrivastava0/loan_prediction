import sys
import os
import pytest
import pandas as pd
import numpy as np
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_data
from src.preprocessing import build_preprocessor


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_data():
    df = pd.read_csv("data/sample_data.csv")
    return df


@pytest.fixture
def prepared_data(sample_data):
    X = sample_data.drop(columns=["Loan_Status", "Loan_ID"], errors="ignore")
    y = sample_data["Loan_Status"].map({"N": 0, "Y": 1})
    return X, y


@pytest.fixture
def trained_model():
    """Load the saved model from artifacts."""
    with open("artifacts/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


# ── data tests ────────────────────────────────────────────────────────────────

def test_data_loads_correctly(sample_data):
    """Dataset should have 614 rows and 13 columns."""
    assert sample_data.shape == (614, 13), \
        f"Expected (614, 13), got {sample_data.shape}"


def test_data_has_required_columns(sample_data):
    """All expected columns must be present."""
    required_columns = [
        "Gender", "Married", "Dependents", "Education",
        "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
        "LoanAmount", "Loan_Amount_Term", "Credit_History",
        "Property_Area", "Loan_Status"
    ]
    for col in required_columns:
        assert col in sample_data.columns, f"Missing column: {col}"


def test_target_has_only_valid_values(sample_data):
    """Loan_Status should only contain Y or N."""
    valid_values = {"Y", "N"}
    actual_values = set(sample_data["Loan_Status"].unique())
    assert actual_values == valid_values, \
        f"Unexpected values in target: {actual_values}"


# ── preprocessing tests ───────────────────────────────────────────────────────

def test_preprocessor_transforms_data(prepared_data):
    """Preprocessor should transform X without errors."""
    X, y = prepared_data
    preprocessor = build_preprocessor(X)
    X_transformed = preprocessor.fit_transform(X)
    assert X_transformed is not None
    assert X_transformed.shape[0] == len(X), \
        "Row count changed after preprocessing"


def test_preprocessor_handles_missing_values(prepared_data):
    """After preprocessing, there should be no NaN values."""
    X, y = prepared_data
    preprocessor = build_preprocessor(X)
    X_transformed = preprocessor.fit_transform(X)
    assert not np.isnan(X_transformed).any(), \
        "NaN values found after preprocessing"


# ── model tests ───────────────────────────────────────────────────────────────

def test_model_loads_successfully(trained_model):
    """Model file should load without errors."""
    assert trained_model is not None


def test_model_predicts_valid_values(trained_model, prepared_data):
    """Model predictions should only be 0 or 1."""
    X, y = prepared_data
    predictions = trained_model.predict(X)
    unique_preds = set(predictions)
    assert unique_preds.issubset({0, 1}), \
        f"Unexpected prediction values: {unique_preds}"


def test_model_predict_proba_returns_valid_probabilities(trained_model, prepared_data):
    """Probabilities should be between 0 and 1."""
    X, y = prepared_data
    probabilities = trained_model.predict_proba(X)[:, 1]
    assert probabilities.min() >= 0.0, "Probability below 0"
    assert probabilities.max() <= 1.0, "Probability above 1"


def test_model_prediction_count_matches_input(trained_model, prepared_data):
    """Number of predictions should match number of input rows."""
    X, y = prepared_data
    predictions = trained_model.predict(X)
    assert len(predictions) == len(X), \
        f"Expected {len(X)} predictions, got {len(predictions)}"