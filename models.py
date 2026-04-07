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
    
class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list, dropout: float):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class NNModel(BaseModel):
    def __init__(self, input_dim: int, hidden_layers=None, dropout=0.3,
                 lr=1e-3, epochs=100, batch_size=64, patience=10, seed=42):
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden_layers = hidden_layers or [128, 64, 32]
        self.model = _MLP(input_dim, hidden_layers, dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.scaler = StandardScaler()

    @property
    def name(self) -> str:
        return "NeuralNetwork"

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_scaled = self.scaler.fit_transform(X_train)
        n_pos = max(y_train.sum(), 1)
        n_neg = max(len(y_train) - n_pos, 1)
        pos_weight = torch.tensor([n_neg / n_pos], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        ds = TensorDataset(
            torch.tensor(X_scaled, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_s = torch.tensor(self.scaler.transform(X_val), dtype=torch.float32).to(self.device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * len(xb)

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_loss = criterion(self.model(X_val_s), y_val_t).item()
                self.model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"NN early stopping at epoch {epoch+1}")
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        logger.info("NeuralNetwork trained")

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            )
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs