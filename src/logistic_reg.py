"""
Logistic regression model
"""

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.config import FEATURES, TARGET


def get_data(file: str = "train.parquet", path: str = "data") -> pd.DataFrame:
    return pd.read_parquet(os.path.join(path, file))


class Model:
    def __init__(self):
        self.pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('cls', LogisticRegression()),
        ])

    def fit(self, x, y):
        self.pipe.fit(x, y)
        return self

    def predict(self, x):
        return self.pipe.predict(x)

    def predict_proba(self, x):
        return self.pipe.predict_proba(x)

    def predict_log_proba(self, x):
        return self.pipe.predict_log_proba(x)

    def score(self, x, y):
        return self.pipe.score(x, y)

    def save(self, file: str = "model.pickle", path: str = "model"):
        with open(os.path.join(path, file)) as f:
            pickle.dump(self, f)


def main():
    df = get_data(path="../data")
    print(df)
    x, y = df[FEATURES], df[TARGET]

    model = Model()
    model.fit(x, y)
    print(model.score(x, y))


if __name__ == "__main__":
    main()
