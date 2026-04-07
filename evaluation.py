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

def get_roc_curves(
    models: List[BaseModel], X: np.ndarray, y: np.ndarray
) -> Dict[str, Dict[str, np.ndarray]]:
    curves = {}
    for m in models:
        y_prob = m.predict_proba(X)
        fpr, tpr, _ = roc_curve(y, y_prob)
        curves[m.name] = {"fpr": fpr, "tpr": tpr}
    return curves


def compute_shap_values(
    model: BaseModel,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: List[str],
    max_background: int = 100,
) -> Optional[np.ndarray]:
    try:
        import shap

    except ImportError:
        logger.warning("SHAP not installed; skipping explainability.")
        return None

    bg = X_background[:max_background]

    from models import XGBModel, LRModel, NNModel

    if isinstance(model, XGBModel):
        explainer = shap.TreeExplainer(model.clf)
        sv = explainer.shap_values(X_explain)

    elif isinstance(model, LRModel):
        X_bg_scaled = model.scaler.transform(bg)
        X_exp_scaled = model.scaler.transform(X_explain)
        explainer = shap.LinearExplainer(model.clf, X_bg_scaled)
        sv = explainer.shap_values(X_exp_scaled)

    elif isinstance(model, NNModel):
        def predict_fn(x):
            return model.predict_proba(x)
        explainer = shap.KernelExplainer(predict_fn, bg)
        sv = explainer.shap_values(X_explain, nsamples=200)
        
    else:
        logger.warning(f"Unknown model type for SHAP: {type(model)}")
        return None

    logger.info(f"SHAP values computed for {model.name}")
    return np.array(sv)