"""
`sklearn_xarray.utils`
"""


import numpy as np


def convert_to_ndarray(X, new_dim_last=True, new_dim_name='variable'):
    """ Convert xarray DataArray or Dataset to numpy ndarray.

    Parameters
    ----------
    X : xarray DataArray or Dataset
        The input data.

    new_dim_last : bool
        TODO

    new_dim_name : str
        TODO

    Returns
    -------
    X_arr : numpy ndarray
        The data as an ndarray.
    """

    if hasattr(X, 'data_vars'):

        if len(X.data_vars) == 1:
            X_arr = X[tuple(X.data_vars)[0]]
        else:
            X_arr = X.to_array(dim=new_dim_name)
            if new_dim_last:
                new_order = list(X_arr.dims)
                new_order.append(new_dim_name)
                new_order.remove(new_dim_name)
                X_arr = X_arr.transpose(*new_order)

    return np.array(X_arr)


def get_group_indices(X, groupby, group_dim=None):
    """

    Parameters
    ----------
    X : xarray DataArray or Dataset
        The data structure for which to determine the indices.

    groupby : str or list
        Name of coordinate or list of coordinates by which the groups are
        determined.

    group_dim : str or None, optional
        Name of dimension along which the groups are indexed.

    Returns
    -------
    idx: list of boolean numpy vectors
        List of logical indices for each group.
    """

    import itertools

    if isinstance(groupby, str):
        groupby = [groupby]

    idx_groups = []
    for g in groupby:
        if group_dim is None or group_dim not in X[g].dims:
            values = X[g].values
        else:
            other_dims = set(X[g].dims) - {group_dim}
            values = X[g].isel(**{d: 0 for d in other_dims}).values
        idx_groups.append([values == v for v in np.unique(values)])

    idx_all = [np.all(e, axis=0) for e in itertools.product(*idx_groups)]

    return [i for i in idx_all if np.any(i)]