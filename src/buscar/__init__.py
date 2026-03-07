"""Buscar: Bioactive Unbiased Single-cell Compound Assessment and Ranking.

A Python framework for prioritizing compounds in high-content imaging drug screening
using single-cell profiles.
"""

from buscar.data_utils import add_cell_id_hash
from buscar.metrics import (
    compute_earth_movers_distance,
    measure_phenotypic_activity,
)
from buscar.preprocess import apply_pca, apply_umap
from buscar.signatures import get_signatures

__version__ = "0.1.0"

__all__ = [
    "add_cell_id_hash",
    "apply_pca",
    "apply_umap",
    "compute_earth_movers_distance",
    "get_signatures",
    "measure_phenotypic_activity",
]
