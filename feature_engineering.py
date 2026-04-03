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

