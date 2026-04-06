"""Generate synthetic data, later real ClinVar data, for training."""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from config import Config
from amino_acid_data import VALID_AAS, get_grantham, get_blosum62, get_aa_properties
from feature_engineering import extract_features

logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def generate_synthetic_dataset(cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_seed)
    aa_list = sorted(VALID_AAS)
    records = []

    for _ in range(cfg.n_synthetic_samples):
        ref = rng.choice(aa_list)
        alt = rng.choice([a for a in aa_list if a != ref])
        position = int(rng.integers(1, 1001))
        protein_length = int(rng.integers(position, position + 2000))

        in_domain = bool(rng.random() < 0.35)
        if in_domain:
            conservation = float(np.clip(rng.beta(5, 2), 0, 1))
        else:
            conservation = float(np.clip(rng.beta(2, 5), 0, 1))

        feats = extract_features(
            ref, alt, position, protein_length,
            conservation_score=conservation, in_domain=in_domain,
        )
        feats["ref_aa"] = ref
        feats["alt_aa"] = alt
        feats["position"] = position
        feats["protein_length"] = protein_length
        records.append(feats)

    df = pd.DataFrame(records)

    logit = (
        0.012 * df["grantham_distance"]
        - 0.25 * df["blosum62_score"]
        + 1.8 * df["conservation_score"]
        + 0.8 * df["in_domain"]
        + 0.005 * df["delta_volume"].abs()
        + 0.15 * df["delta_hydrophobicity"].abs()
        + 0.6 * df["delta_charge"].abs()
        - 1.5
    )
    prob = _sigmoid(logit.values + rng.normal(0, 0.5, len(df)))
    df["label"] = (rng.random(len(df)) < prob).astype(int)

    label_counts = df["label"].value_counts()
    logger.info(
        f"Synthetic data: {len(df)} variants, "
        f"benign={label_counts.get(0, 0)}, pathogenic={label_counts.get(1, 0)}"
    )
    return df


def load_clinvar_tsv(filepath: str) -> pd.DataFrame:
    df_raw = pd.read_csv(filepath, sep="\t", low_memory=False)

    mask = (
        df_raw["Type"].str.contains("single nucleotide", case=False, na=False)
        & df_raw["ClinicalSignificance"].str.contains(
            "Pathogenic|Benign", case=False, na=False
        )
    )
    df_filt = df_raw.loc[mask].copy()


    df_filt["label"] = df_filt["ClinicalSignificance"].apply(
        lambda s: 1 if "pathogenic" in s.lower() else 0
    )
    logger.info(f"ClinVar loaded: {len(df_filt)} variants after filtering")
    return df_filt

def prepare_xy(df: pd.DataFrame, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and label vector y from dataframe."""
    missing = [f for f in cfg.feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in dataframe: {missing}")

    X = df[cfg.feature_names].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)
    return X, y