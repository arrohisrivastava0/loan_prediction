import sys
import os
import yaml

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
logger.info(f"Data loaded from {config['data']['path']} with shape {df.shape}")

# preprocess data
X = df.drop(columns=[config["data"]["target"]])
y = df[config["data"]["target"]]
preprocessor = build_preprocessor(X)
logger.info("Data preprocessing completed. Features: {}, Target: {}".format(X.shape[1], y.shape))

# train model
model, X_train, X_test, y_train, y_test = train_model(
    X, y,
    preprocessor=preprocessor,
    model_params=config["model"]["params"],
    test_size=config["training"]["test_size"],
    random_state=config["training"]["random_state"]
    )
logger.info("Model training completed")

# evaluate model
metrics, y_pred = evaluate_model(model, X_test, y_test)
print_evaluation(metrics, y_test, y_pred)
logger.info(f"Metrics: {metrics}")

# save model
save_model(model)
logger.info(f"Model saved to artifacts/model.pkl")