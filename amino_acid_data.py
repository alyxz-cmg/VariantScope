"""Amino acid physicochemical properties and substitution matrices."""

from typing import Dict, Optional, Tuple

AA_PROPERTIES: Dict[str, Dict[str, float]] = {
    "A": {"hydrophobicity": 1.8,  "volume": 88.6,  "mw": 89.1,  "charge": 0.0, "polarity": 8.1},
    "R": {"hydrophobicity": -4.5, "volume": 173.4, "mw": 174.2, "charge": 1.0, "polarity": 10.5},
    "N": {"hydrophobicity": -3.5, "volume": 114.1, "mw": 132.1, "charge": 0.0, "polarity": 11.6},
    "D": {"hydrophobicity": -3.5, "volume": 111.1, "mw": 133.1, "charge": -1.0,"polarity": 13.0},
    "C": {"hydrophobicity": 2.5,  "volume": 108.5, "mw": 121.2, "charge": 0.0, "polarity": 5.5},
    "E": {"hydrophobicity": -3.5, "volume": 138.4, "mw": 147.1, "charge": -1.0,"polarity": 12.3},
    "Q": {"hydrophobicity": -3.5, "volume": 143.8, "mw": 146.2, "charge": 0.0, "polarity": 10.5},
    "G": {"hydrophobicity": -0.4, "volume": 60.1,  "mw": 75.0,  "charge": 0.0, "polarity": 9.0},
    "H": {"hydrophobicity": -3.2, "volume": 153.2, "mw": 155.2, "charge": 0.5, "polarity": 10.4},
    "I": {"hydrophobicity": 4.5,  "volume": 166.7, "mw": 131.2, "charge": 0.0, "polarity": 5.2},
    "L": {"hydrophobicity": 3.8,  "volume": 166.7, "mw": 131.2, "charge": 0.0, "polarity": 4.9},
    "K": {"hydrophobicity": -3.9, "volume": 168.6, "mw": 146.2, "charge": 1.0, "polarity": 11.3},
    "M": {"hydrophobicity": 1.9,  "volume": 162.9, "mw": 149.2, "charge": 0.0, "polarity": 5.7},
    "F": {"hydrophobicity": 2.8,  "volume": 189.9, "mw": 165.2, "charge": 0.0, "polarity": 5.2},
    "P": {"hydrophobicity": -1.6, "volume": 112.7, "mw": 115.1, "charge": 0.0, "polarity": 8.0},
    "S": {"hydrophobicity": -0.8, "volume": 89.0,  "mw": 105.1, "charge": 0.0, "polarity": 9.2},
    "T": {"hydrophobicity": -0.7, "volume": 116.1, "mw": 119.1, "charge": 0.0, "polarity": 8.6},
    "W": {"hydrophobicity": -0.9, "volume": 227.8, "mw": 204.2, "charge": 0.0, "polarity": 5.4},
    "Y": {"hydrophobicity": -1.3, "volume": 193.6, "mw": 181.2, "charge": 0.0, "polarity": 6.2},
    "V": {"hydrophobicity": 4.2,  "volume": 140.0, "mw": 117.1, "charge": 0.0, "polarity": 5.9},
}


VALID_AAS = set(AA_PROPERTIES.keys())


_GRANTHAM_ORDER = "ARNDCQEGHILKMFPSTWYV"
_GRANTHAM_LOWER = [
    [],                                                                     # A
    [112],                                                                  # R
    [111, 86],                                                              # N
    [126, 96, 23],                                                          # D
    [195, 180, 139, 154],                                                   # C
    [91, 43, 46, 61, 154],                                                  # Q
    [107, 54, 42, 45, 170, 29],                                             # E
    [60, 125, 80, 94, 159, 87, 98],                                         # G
    [86, 29, 68, 81, 174, 24, 40, 98],                                      # H
    [94, 97, 149, 168, 198, 109, 134, 135, 94],                             # I
    [96, 102, 153, 172, 198, 113, 138, 138, 99, 5],                         # L
    [106, 26, 94, 101, 202, 53, 56, 127, 32, 102, 107],                     # K
    [84, 91, 142, 160, 196, 101, 126, 127, 87, 10, 15, 95],                 # M
    [113, 97, 158, 177, 205, 116, 140, 153, 100, 21, 22, 102, 28],          # F
    [27, 103, 91, 108, 169, 76, 93, 42, 77, 95, 98, 103, 87, 114],          # P
    [99, 110, 46, 65, 112, 68, 80, 56, 89, 142, 145, 121, 135, 155, 74],    # S
    [58, 71, 65, 85, 149, 42, 65, 59, 47, 89, 92, 78, 81, 103, 38, 58],     # T
    [148, 101, 174, 181, 215, 130, 152, 184, 115, 61, 61, 110, 67, 50, 147, 177, 128],  # W
    [112, 77, 143, 160, 194, 99, 121, 147, 83, 33, 36, 85, 36, 22, 110, 144, 92, 37],   # Y
    [64, 96, 133, 152, 192, 96, 121, 109, 84, 29, 32, 97, 21, 50, 68, 124, 69, 88, 55], # V
]

GRANTHAM: Dict[Tuple[str, str], int] = {}
for i, aa_i in enumerate(_GRANTHAM_ORDER):
    GRANTHAM[(aa_i, aa_i)] = 0
    for j, dist in enumerate(_GRANTHAM_LOWER[i]):
        aa_j = _GRANTHAM_ORDER[j]
        GRANTHAM[(aa_i, aa_j)] = dist
        GRANTHAM[(aa_j, aa_i)] = dist


def get_grantham(aa1: str, aa2: str) -> Optional[int]:
    return GRANTHAM.get((aa1.upper(), aa2.upper()))


def get_blosum62(aa1: str, aa2: str) -> Optional[int]:
    try:
        from Bio.Align.substitution_matrices import load
        mat = load("BLOSUM62")
        return int(mat[aa1.upper()][aa2.upper()])
    except Exception:
        if aa1.upper() == aa2.upper():
            return 4
        gd = get_grantham(aa1, aa2)
        if gd is not None:
            if gd < 60:
                return 1
            elif gd < 100:
                return 0
            else:
                return -2
        return None


def get_aa_properties(aa: str) -> Optional[Dict[str, float]]:
    return AA_PROPERTIES.get(aa.upper())