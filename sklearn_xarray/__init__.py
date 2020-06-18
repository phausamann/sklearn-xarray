""" ``sklearn_xarray`` """

from sklearn_xarray.common.wrappers import (
    wrap,
    EstimatorWrapper,
    ClassifierWrapper,
    RegressorWrapper,
    TransformerWrapper,
)
from sklearn_xarray.target import Target

import os


__all__ = [
    "wrap",
    "EstimatorWrapper",
    "ClassifierWrapper",
    "RegressorWrapper",
    "TransformerWrapper",
    "Target",
]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

__version__ = "0.3.0"
