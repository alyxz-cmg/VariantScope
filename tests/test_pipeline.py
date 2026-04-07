"""Tests for core pipeline components."""

import numpy as np
import pytest

from config import Config
from amino_acid_data import get_grantham, get_blosum62, get_aa_properties, VALID_AAS
from feature_engineering import parse_variant, extract_features, extract_features_from_string
from data_loader import generate_synthetic_dataset, prepare_xy


class TestAminoAcidData:
    def test_grantham_symmetric(self):
        assert get_grantham("A", "R") == get_grantham("R", "A")

    def test_grantham_identity_zero(self):
        for aa in VALID_AAS:
            assert get_grantham(aa, aa) == 0

    def test_grantham_known_value(self):
        assert get_grantham("A", "R") == 112
    
    def test_blosum62_returns_int(self):
        score = get_blosum62("A", "A")
        assert isinstance(score, int)

    def test_properties_all_amino_acids(self):
        for aa in VALID_AAS:
            props = get_aa_properties(aa)
            assert props is not None
            assert "hydrophobicity" in props

class TestFeatureEngineering:
    def test_parse_single_letter(self):
        ref, pos, alt = parse_variant("A123T")
        assert ref == "A" and pos == 123 and alt == "T"

    def test_parse_three_letter(self):
        ref, pos, alt = parse_variant("p.Ala123Thr")
        assert ref == "A" and pos == 123 and alt == "T"

    def test_parse_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_variant("INVALID")

    def test_extract_features_shape(self):
        feats = extract_features("A", "T", position=50, protein_length=500,
                                 conservation_score=0.8, in_domain=True)
        assert len(feats) == 20

    def test_extract_features_from_string(self):
        feats = extract_features_from_string("G56D", conservation_score=0.5, in_domain=False)
        assert "grantham_distance" in feats
        assert feats["in_domain"] == 0.0

    def test_invalid_aa_raises(self):
        with pytest.raises(ValueError):
            extract_features("X", "Z")

class TestDataLoader:
    def test_synthetic_data_shape(self):
        cfg = Config(n_synthetic_samples=100)
        df = generate_synthetic_dataset(cfg)
        assert len(df) == 100
        assert "label" in df.columns

    def test_prepare_xy(self):
        cfg = Config(n_synthetic_samples=50)
        df = generate_synthetic_dataset(cfg)
        X, y = prepare_xy(df, cfg)
        assert X.shape == (50, 20)
        assert y.shape == (50,)
        assert set(np.unique(y)).issubset({0.0, 1.0})