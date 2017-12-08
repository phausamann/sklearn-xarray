"""" ``sklearn_xarray.externals`` """

try:
    import numpy_groupies
except ImportError:
    from . import _numpy_groupies_np as numpy_groupies