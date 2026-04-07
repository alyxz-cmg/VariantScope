"""Three classifiers: Logistic Regression, XGBoost, and PyTorch Neural Network."""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

class LRModel(BaseModel):
    def __init__(self, C: float = 1.0, max_iter: int = 1000, seed: int = 42):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            C=C, max_iter=max_iter, random_state=seed, class_weight="balanced"
        )

    @property
    def name(self) -> str:
        return "LogisticRegression"

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_scaled = self.scaler.fit_transform(X_train)
        self.clf.fit(X_scaled, y_train)
        logger.info("LogisticRegression trained")

    def predict(self, X):
        return self.clf.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        return self.clf.predict_proba(self.scaler.transform(X))[:, 1]