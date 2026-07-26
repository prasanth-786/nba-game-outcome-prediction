import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna()
    X = df.drop(columns=["HOME_TEAM_WINS"])
    y = df["HOME_TEAM_WINS"]
    return X, y
