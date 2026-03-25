import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
from src.logger import setup_logger

# initialize logger
logger = setup_logger()

# load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

logger.info("Config loaded successfully")

# log config content
logger.info(f"Config content: {config}")