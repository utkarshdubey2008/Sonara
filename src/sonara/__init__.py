"""
Sonara: Musical Representation and Intelligence System.

A research-oriented framework for decoding musical audio into structured,
machine-readable representations (SMR), analyzing musical collections and
style profiles, and guiding music generation.
"""

__version__ = "0.1.0"
__author__ = "Utkarsh Dubey"
__license__ = "Apache-2.0"

from sonara.pipeline import SonaraPipeline
from sonara.representation.schema import SMRDocument
from sonara.analysis.style_profile import StyleProfile

__all__ = [
    "SonaraPipeline",
    "SMRDocument",
    "StyleProfile",
    "__version__",
]
