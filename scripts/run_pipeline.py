from datetime import datetime
import sys
import os
import yaml
import mlflow
import mlflow.sklearn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.logger import setup_logger
from src.data_loader import load_data
from src.preprocessing import build_preprocessor
from src.train import train_model, save_model
from src.evaluate import evaluate_model, print_evaluation

# load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)
    
# initialize logger
logger = setup_logger()
logger.info("Config loaded successfully")

#load data
df = load_data(config["data"]["path"])
logger.info(f"Data loaded: {df.shape}")

# split features and target — map target to 0/1 once here
target_col = config["data"]["target"]
X = df.drop(columns=[target_col, 'Loan_ID'], errors='ignore')
y = df[target_col].map({'N': 0, 'Y': 1})

# preprocess data
preprocessor = build_preprocessor(X)
logger.info(f"Preprocessor built. Features: {X.shape[1]}")

# mlflow setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("loan_prediction")

# train and log all models
models_to_compare = config["models_to_compare"]

best_model = None        # ← must be here
best_roc_auc = 0         # ← must be here

for model_name, model_params in models_to_compare.items():
    with mlflow.start_run(run_name=model_name):
        logger.info(f"Training model: {model_name}")
        
        # log params to mlflow
        mlflow.log_params({
            "model_name": model_name,
            "test_size": config["training"]["test_size"],
            "random_state": config["training"]["random_state"],
            **model_params
        })
        
        # train
        pipeline, X_train, X_test, y_train, y_test = train_model(
            X, y,
            preprocessor=preprocessor,
            model_name=model_name,
            model_params=model_params,
            test_size=config["training"]["test_size"],
            random_state=config["training"]["random_state"]
        )
        
        metrics, y_pred = evaluate_model(pipeline, X_test, y_test)
        print(f"\n>>> {model_name.upper()}")
        print_evaluation(metrics, y_test, y_pred)
        
        # log metrics to mlflow
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, name="model")
        
        logger.info(f"Done: {model_name} | metrics: {metrics}")
        
        
        
        # track best model
        if metrics["roc_auc"] > best_roc_auc:
            best_roc_auc = metrics["roc_auc"]
            best_model = pipeline
            best_model_name = model_name
        
        # save_model(model)
        # logger.info(f"Model {model_name} saved to artifacts/model.pkl")
        
# save best model
save_model(best_model, path="artifacts/best_model.pkl")
logger.info(f"Best model: {best_model_name} (ROC-AUC: {best_roc_auc}) saved to artifacts/best_model.pkl")
print(f"\n✓ Best model: {best_model_name} with ROC-AUC: {best_roc_auc}")





# train model
# model, X_train, X_test, y_train, y_test = train_model(
#     X, y,
#     preprocessor=preprocessor,
#     model_name=config["model"]["name"],
#     model_params=config["model"]["params"],
#     test_size=config["training"]["test_size"],
#     random_state=config["training"]["random_state"]
#     )
# logger.info(f"Model training completed - model: {config['model']['name']}")

# # evaluate model
# metrics, y_pred = evaluate_model(model, X_test, y_test)
# print_evaluation(metrics, y_test, y_pred)
# logger.info(f"Metrics: {metrics}")

# save model
