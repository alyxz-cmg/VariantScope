"""Central configuration for the Variant Impact Predictor."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

@dataclass
class Config:
    data_dir: Path = Path("data")
    model_dir: Path = Path("models_saved")

    n_synthetic_samples: int = 5000
    test_size: float = 0.2
    random_seed: int = 42
    class_ratio_benign: float = 0.6

    feature_names: List[str] = field(default_factory=lambda: [
        "grantham_distance",
        "blosum62_score",
        "ref_hydrophobicity",
        "alt_hydrophobicity",
        "delta_hydrophobicity",
        "ref_volume",
        "alt_volume",
        "delta_volume",
        "ref_molecular_weight",
        "alt_molecular_weight",
        "delta_molecular_weight",
        "ref_charge",
        "alt_charge",
        "delta_charge",
        "ref_polarity",
        "alt_polarity",
        "delta_polarity",
        "conservation_score",
        "in_domain",
        "sequence_position_pct",
    ])

    lr_C: float = 1.0
    lr_max_iter: int = 1000

    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1

    nn_hidden_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    nn_dropout: float = 0.3
    nn_lr: float = 1e-3
    nn_epochs: int = 100
    nn_batch_size: int = 64
    nn_patience: int = 10

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)