import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from given path
    """
    df = pd.read_csv(path)
    return df


def validate_data(df: pd.DataFrame, target_col: str):
    """
    Validate dataset structure
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    if df.empty:
        raise ValueError("Dataset is empty")

    return True


def split_features_target(df: pd.DataFrame, target_col: str):
    """
    Split dataset into features (X) and target (y)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y