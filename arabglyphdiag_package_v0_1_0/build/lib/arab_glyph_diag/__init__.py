"""ArabGlyphDiag public API."""

from .diagnostic import GlyphDiagnostic, failure_mode_for_pair
from .metadata import ARABIC_LETTERS, DEFAULT_METADATA, DOTTED_CLASS_IDS
from .priors import (
    binarize_ink,
    build_empirical_priors,
    detect_secondary_components,
    describe_secondary_components,
    synthetic_spatial_prior,
    top_area_mask,
)

__all__ = [
    "GlyphDiagnostic",
    "failure_mode_for_pair",
    "ARABIC_LETTERS",
    "DEFAULT_METADATA",
    "DOTTED_CLASS_IDS",
    "binarize_ink",
    "detect_secondary_components",
    "describe_secondary_components",
    "build_empirical_priors",
    "synthetic_spatial_prior",
    "top_area_mask",
]

__version__ = "0.1.0"
