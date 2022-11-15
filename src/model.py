import os
import pickle
from sklearn.pipeline import Pipeline
from sklearn.utils import estimator_html_repr


class Model:
    def __init__(self):
        self.pipe = Pipeline([])

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
        with open(os.path.join(path, file), "w") as f:
            pickle.dump(self, f)
        return self

    def save_estimator_html(self, file: str = "estimator.html", path: str = "data"):
        """
        Save model visualization
        """
        with open(os.path.join(path, file), "w") as f:
            f.write(estimator_html_repr(self.pipe))
        return self
