"""
`sklearn_xarray.utils`
"""


import numpy as np


def get_group_indices(X, groupby):
    """

    Parameters
    ----------
    X : xarray DataArray or Dataset
        The data structure for which to determine the indices.
    groupby : str or list
        Name of coordinate or list of coordinates by which the groups are
        determined.

    Returns
    -------
    idx: list of boolean numpy vectors
        List of logical indices for each group.
    """

    import itertools

    if isinstance(groupby, str):
        groupby = [groupby]

    idx_props = [[X[p].values == v for v in np.unique(X[p].values)]
                 for p in groupby]

    # TODO axis!
    idx_all = [np.all(e, axis=0) for e in itertools.product(*idx_props)]

    return [i for i in idx_all if np.any(i)]