""" ``sklearn_xarray.dataset`` """

from sklearn_xarray.common.wrappers import (
    wrap, TransformerWrapper, RegressorWrapper, ClassifierWrapper)

import warnings


warnings.simplefilter('always', DeprecationWarning)

warnings.warn(
    'Importing wrappers from sklearn_xarray.dataset is deprecated and will '
    'be removed in a future version. Please import the wrappers directly from '
    'the top-level module, e.g. `from sklearn_xarray import wrap`.',
    DeprecationWarning)

warnings.simplefilter('ignore', DeprecationWarning)

__all__ = [
    'wrap',
    'ClassifierWrapper',
    'RegressorWrapper',
    'TransformerWrapper',
]
