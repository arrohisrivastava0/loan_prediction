from datetime import datetime
import yaml
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.logger import setup_logger
from src.data_loader import load_data
from src.preprocessing import build_preprocessor
from src.train import train_model


def run_pipeline():

    # -----------------------------
    # Load config
    # -----------------------------
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    logger = setup_logger()
    logger.info("Pipeline started")

    # -----------------------------
    # Load data
    # -----------------------------
    df = load_data(config["data"]["path"])

    if df.empty:
        raise ValueError("Dataset is empty")

    target_col = config["data"]["target"]

    X = df.drop(columns=[target_col, 'Loan_ID'], errors='ignore')
    y = df[target_col].map({'N': 0, 'Y': 1})

    # -----------------------------
    # Build preprocessor
    # -----------------------------
    preprocessor = build_preprocessor(X)

    # -----------------------------
    # MLflow setup
    # -----------------------------
    mlflow.set_experiment("loan_prediction")

    best_auc = 0
    best_run_id = None
    best_model_name = None

    # -----------------------------
    # Train models
    # -----------------------------
    for model_name, model_params in config["models_to_compare"].items():

        with mlflow.start_run(run_name=model_name):

            logger.info(f"Training model: {model_name}")

            pipeline, metrics = train_model(
                X, y,
                preprocessor=preprocessor,
                model_name=model_name,
                model_params=model_params,
                test_size=config["training"]["test_size"],
                random_state=config["training"]["random_state"]
            )

            # Log params + metrics
            mlflow.log_params({
                "model": model_name,
                **model_params
            })

            mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            logger.info(f"{model_name} metrics: {metrics}")

            # Track best model
            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_run_id = mlflow.active_run().info.run_id
                best_model_name = model_name

    # -----------------------------
    # Safety check
    # -----------------------------
    if best_run_id is None:
        raise RuntimeError("No model was successfully trained")

    # -----------------------------
    # Register best model
    # -----------------------------
    model_uri = f"runs:/{best_run_id}/model"

    logger.info(f"Registering best model: {best_model_name}")

    result = mlflow.register_model(
        model_uri=model_uri,
        name="LoanPredictionModel"
    )

    # -----------------------------
    # Promote to Production
    # -----------------------------
    client = MlflowClient()

    client.transition_model_version_stage(
        name="LoanPredictionModel",
        version=result.version,
        stage="Production"
    )

    logger.info(
        f"Best model: {best_model_name} | "
        f"AUC: {best_auc:.4f} | "
        f"Version: {result.version} promoted to Production"
    )

    return best_run_id, best_auc