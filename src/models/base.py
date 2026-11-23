from abc import ABC, abstractmethod
from typing import Self

import joblib


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train on the supplied training split."""

    @abstractmethod
    def predict(self, X):
        """Return one class label per row."""

    def predict_proba(self, X):
        """Return probabilities in sorted class order, or None if unavailable."""
        return None

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path) -> Self:
        """Load a trusted joblib artifact saved by this model class."""
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Expected {cls.__name__}, loaded {type(model).__name__}")
        return model
