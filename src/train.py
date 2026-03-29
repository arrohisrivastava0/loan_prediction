import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def train_model(X,y,preprocessor,model_name: str, model_params:dict, test_size:float, random_state:int):
    # models = {
    #     'logistic_regression': LogisticRegression(class_weight="balanced", **model_params),
    #     'random_forest': RandomForestClassifier(**model_params),
    #     'xgboost': XGBClassifier(**model_params, eval_metric='logloss')
    # }
    
    if model_name == "logistic_regression":
        model = LogisticRegression(class_weight="balanced", **model_params)

    elif model_name == "random_forest":
        model = RandomForestClassifier(**model_params)

    elif model_name == "xgboost":
        model = XGBClassifier(**model_params, eval_metric="logloss")

    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # if model_name not in models:
    #     raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
        )
    
    # model = LogisticRegression(class_weight="balanced", **model_params)

    pipeline.fit(X_train, y_train)

    return pipeline, X_train, X_test, y_train, y_test


def save_model(model, path:str="artifacts/model.pkl"):
    """
    Save the trained model to disk
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)