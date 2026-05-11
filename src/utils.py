import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    df["high_risk"] = (df["age"] > 50) & (df["visits"] > 5)
    return df