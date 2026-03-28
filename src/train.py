import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def train_model(X,y,preprocessor,model_params:dict, test_size:float, random_state:int):
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
        )
    
    model = LogisticRegression(**model_params)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    return pipeline, X_train, X_test, y_train, y_test

# def train_model(X, y, model_params:dict, test_size:float, random_state:int):
#     """
#     Train a logistic regression model and save it to disk
#     """
#     
    
#     model = LogisticRegression(**model_params)
#     model.fit(X_train, y_train)

#     return model, X_train, X_test, y_train, y_test

def save_model(model, path:str="artifacts/model.pkl"):
    """
    Save the trained model to disk
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)