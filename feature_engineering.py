"""Extract a 20-dimensional feature vector for a missense variant."""

import re
import logging
from typing import Dict, Optional, Tuple
import numpy as np

from amino_acid_data import (
    VALID_AAS,
    get_aa_properties,
    get_grantham,
    get_blosum62,
)

logger = logging.getLogger(__name__)

THREE_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}

def parse_variant(variant_str: str) -> Tuple[str, int, str]:
    variant_str = variant_str.strip().lstrip("p.")

    m = re.match(r"([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})", variant_str)

    if m:
        ref = THREE_TO_ONE.get(m.group(1))
        alt = THREE_TO_ONE.get(m.group(3))
        if ref and alt:
            return ref, int(m.group(2)), alt
        
    m = re.match(r"([A-Z])(\d+)([A-Z])", variant_str)

    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    
    raise ValueError(f"Cannot parse variant: '{variant_str}'. Use format A123T or p.Ala123Thr")

def extract_features(
    ref_aa: str,
    alt_aa: str,
    position: int = 0,
    protein_length: int = 500,
    conservation_score: Optional[float] = None,
    in_domain: Optional[bool] = None,
) -> Dict[str, float]:
    ref_aa, alt_aa = ref_aa.upper(), alt_aa.upper()
    if ref_aa not in VALID_AAS or alt_aa not in VALID_AAS:
        raise ValueError(f"Invalid amino acid(s): {ref_aa}, {alt_aa}")

    ref_props = get_aa_properties(ref_aa)
    alt_props = get_aa_properties(alt_aa)

    grantham = get_grantham(ref_aa, alt_aa) or 0
    blosum = get_blosum62(ref_aa, alt_aa) or 0

    if conservation_score is None:
        conservation_score = float(np.random.uniform(0, 1))

    if in_domain is None:
        in_domain = bool(np.random.random() < 0.3)

    pos_pct = position / max(protein_length, 1)

    return {
        "grantham_distance": float(grantham),
        "blosum62_score": float(blosum),
        "ref_hydrophobicity": ref_props["hydrophobicity"],
        "alt_hydrophobicity": alt_props["hydrophobicity"],
        "delta_hydrophobicity": alt_props["hydrophobicity"] - ref_props["hydrophobicity"],
        "ref_volume": ref_props["volume"],
        "alt_volume": alt_props["volume"],
        "delta_volume": alt_props["volume"] - ref_props["volume"],
        "ref_molecular_weight": ref_props["mw"],
        "alt_molecular_weight": alt_props["mw"],
        "delta_molecular_weight": alt_props["mw"] - ref_props["mw"],
        "ref_charge": ref_props["charge"],
        "alt_charge": alt_props["charge"],
        "delta_charge": alt_props["charge"] - ref_props["charge"],
        "ref_polarity": ref_props["polarity"],
        "alt_polarity": alt_props["polarity"],
        "delta_polarity": alt_props["polarity"] - ref_props["polarity"],
        "conservation_score": conservation_score,
        "in_domain": float(in_domain),
        "sequence_position_pct": pos_pct,
    }


def extract_features_from_string(
    variant_str: str,
    protein_length: int = 500,
    conservation_score: Optional[float] = None,
    in_domain: Optional[bool] = None,
    ) -> Dict[str, float]:
    ref, pos, alt = parse_variant(variant_str)
    return extract_features(ref, alt, pos, protein_length, conservation_score, in_domain)