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
    
class XGBModel(BaseModel):
    def __init__(self, n_estimators=200, max_depth=6, lr=0.1, seed=42):
        from xgboost import XGBClassifier
        self.clf = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=lr,
            random_state=seed,
            eval_metric="logloss",
            use_label_encoder=False,
        )

    @property
    def name(self) -> str:
        return "XGBoost"

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False
            
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        if n_pos > 0:
            self.clf.set_params(scale_pos_weight=n_neg / n_pos)
        self.clf.fit(X_train, y_train, **fit_params)
        logger.info("XGBoost trained")

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)[:, 1]