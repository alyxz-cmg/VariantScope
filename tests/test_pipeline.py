"""Tests for core pipeline components."""

import numpy as np
import pytest

from config import Config
from amino_acid_data import get_grantham, get_blosum62, get_aa_properties, VALID_AAS

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