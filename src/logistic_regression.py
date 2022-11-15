"""
Logistic regression model
"""

import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from src.config import FEATURES, TARGET, ONEHOT_FEATS, ORDINAL_FEATS, LOG_FEATS
from src.model import Model


def get_data(file: str = "train.parquet", path: str = "data") -> pd.DataFrame:
    return pd.read_parquet(os.path.join(path, file))


class LRModel(Model):
    def __init__(self, **parameters: dict):
        super().__init__()
        transformers = []
        other_feats = FEATURES
        if len(ORDINAL_FEATS) > 0:
            for f in ORDINAL_FEATS.keys():
                other_feats.remove(f)
            categories = list(ORDINAL_FEATS.values())
            cols = list(ORDINAL_FEATS.keys())
            transformers += [("oe", OrdinalEncoder(categories=categories), cols)]
        if len(ONEHOT_FEATS) > 0:
            for f in ONEHOT_FEATS.keys():
                other_feats.remove(f)
            categories = list(ONEHOT_FEATS.values())
            cols = list(ONEHOT_FEATS.keys())
            transformers += [("ohe", OneHotEncoder(categories=categories, drop="first"), cols)]
        if len(LOG_FEATS) > 0:
            for f in LOG_FEATS:
                other_feats.remove(f)
            cols = LOG_FEATS
            processor = make_pipeline(FunctionTransformer(func=np.log1p), StandardScaler())
            transformers += [("log", processor, cols)]
        if len(other_feats) > 0:
            cols = other_feats
            transformers += [("std", StandardScaler(), cols)]

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        classifier = LogisticRegression(**parameters)

        self.pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("cls", classifier),
        ])


def main():
    df = get_data(path="../data")
    x, y = df[FEATURES], df[TARGET].values.ravel()

    model = LRModel()
    model.save_estimator_html(path="../data")
    model.fit(x, y)
    # model.save(path="../data")
    print(model.score(x, y))


if __name__ == "__main__":
    main()
