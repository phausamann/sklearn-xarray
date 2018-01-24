""" ``sklearn_xarray.data`` """

from sklearn_xarray.datasets import *

import warnings


warnings.simplefilter('always', DeprecationWarning)

warnings.warn(
    'Importing data from sklearn_xarray.data is deprecated and will '
    'be removed in a future version. Please use sklearn_xarray.datasets '
    'instead.', DeprecationWarning)

warnings.simplefilter('ignore', DeprecationWarning)

__all__ = [
    'load_dummy_dataarray',
    'load_digits_dataarray',
    'load_wisdm_dataarray',
    'load_dummy_dataset'
]
