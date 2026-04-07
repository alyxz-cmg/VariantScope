"""End-to-end training and evaluation."""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from config import Config
from data_loader import generate_synthetic_dataset, prepare_xy
from models import BaseModel, LRModel, XGBModel, NNModel
from evaluation import compare_models, compute_shap_values

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def build_models(cfg: Config, input_dim: int) -> List[BaseModel]:
    return [
        LRModel(C=cfg.lr_C, max_iter=cfg.lr_max_iter, seed=cfg.random_seed),
        XGBModel(
            n_estimators=cfg.xgb_n_estimators,
            max_depth=cfg.xgb_max_depth,
            lr=cfg.xgb_learning_rate,
            seed=cfg.random_seed,
        ),
        NNModel(
            input_dim=input_dim,
            hidden_layers=cfg.nn_hidden_layers,
            dropout=cfg.nn_dropout,
            lr=cfg.nn_lr,
            epochs=cfg.nn_epochs,
            batch_size=cfg.nn_batch_size,
            patience=cfg.nn_patience,
            seed=cfg.random_seed,
        ),
    ]


def run_pipeline(cfg: Config = None) -> Tuple[List[BaseModel], Dict, np.ndarray, np.ndarray]:
    if cfg is None:
        cfg = Config()

    logger.info("Generating synthetic dataset...")
    df = generate_synthetic_dataset(cfg)
    X, y = prepare_xy(df, cfg)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_seed, stratify=y
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=cfg.random_seed, stratify=y_train
    )

    models = build_models(cfg, input_dim=X.shape[1])
    for m in models:
        logger.info(f"Training {m.name}...")
        m.fit(X_tr, y_tr, X_val, y_val)

    comparison_df = compare_models(models, X_test, y_test)

    shap_results = {}
    n_explain = min(50, len(X_test))
    for m in models:
        sv = compute_shap_values(
            m, X_train, X_test[:n_explain], cfg.feature_names
        )

        if sv is not None:
            shap_results[m.name] = sv

    for m in models:
        path = cfg.model_dir / f"{m.name}.pkl"
        
        with open(path, "wb") as f:
            pickle.dump(m, f)
        logger.info(f"Saved {m.name} → {path}")

    return models, {"comparison": comparison_df, "shap": shap_results}, X_test, y_test


if __name__ == "__main__":
    models, results, X_test, y_test = run_pipeline()
    print("\n=== Model Comparison ===")
    print(results["comparison"].round(4).to_string())