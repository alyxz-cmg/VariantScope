"""Model evaluation, comparison, and SHAP explainability."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    roc_curve,
)

from models import BaseModel

logger = logging.getLogger(__name__)


def evaluate_model(model: BaseModel, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_prob),
        "avg_precision": average_precision_score(y, y_prob),
    }

def compare_models(
    models: List[BaseModel], X: np.ndarray, y: np.ndarray
) -> pd.DataFrame:
    rows = []
    for m in models:
        metrics = evaluate_model(m, X, y)
        metrics["model"] = m.name
        rows.append(metrics)
    df = pd.DataFrame(rows).set_index("model")
    logger.info(f"Model comparison:\n{df.round(4).to_string()}")
    return df