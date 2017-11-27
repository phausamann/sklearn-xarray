"""
`sklearn_xarray.utils`
"""


import numpy as np

from .target import Target


def is_dataarray(X, require_attrs=None):
    """ Check whether an object is a DataArray.

    Parameters
    ----------
    X : anything
        The object to be checked.

    require_attrs : list of str, optional
        The attributes the object has to have in order to pass as a DataArray.

    Returns
    -------
    Whether the object is a DataArray or not.
    """

    if require_attrs is None:
        require_attrs = [
            'values',
            'coords',
            'dims',
            'to_dataset'
        ]

    return np.all([hasattr(X, name) for name in require_attrs])


def is_dataset(X, require_attrs=None):
    """ Check whether an object is a Dataset.

    Parameters
    ----------
    X : anything
        The object to be checked.

    require_attrs : list of str, optional
        The attributes the object has to have in order to pass as a Dataset.

    Returns
    -------
    Whether the object is a Dataset or not.
    """

    if require_attrs is None:
        require_attrs = [
            'data_vars',
            'coords',
            'dims',
            'to_array'
        ]

    return np.all([hasattr(X, name) for name in require_attrs])


def is_target(X, require_attrs=None):
    """ Check whether an object is a Target.

    Parameters
    ----------
    X : anything
        The object to be checked.

    require_attrs : list of str, optional
        The attributes the object has to have in order to pass as a Target.

    Returns
    -------
    Whether the object is a Target or not.
    """

    if require_attrs is None:
        require_attrs = (
            name for name in vars(Target) if not name.startswith('_'))

    return np.all([hasattr(X, name) for name in require_attrs])


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

    if is_dataset(X):

        if len(X.data_vars) == 1:
            X = X[tuple(X.data_vars)[0]]
        else:
            X = X.to_array(dim=new_dim_name)
            if new_dim_last:
                new_order = list(X.dims)
                new_order.append(new_dim_name)
                new_order.remove(new_dim_name)
                X = X.transpose(*new_order)

    return np.array(X)


def get_group_indices(X, groupby, group_dim=None):
    """ Get logical index vectors for each group.

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
