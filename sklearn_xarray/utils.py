"""
``sklearn_xarray.utils``
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
    bool
        Whether the object is a DataArray or not.
    """

    if require_attrs is None:
        require_attrs = [
            'values',
            'coords',
            'dims',
            'to_dataset'
        ]

    return all([hasattr(X, name) for name in require_attrs])


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
    bool
        Whether the object is a Dataset or not.
    """

    if require_attrs is None:
        require_attrs = [
            'data_vars',
            'coords',
            'dims',
            'to_array'
        ]

    return all([hasattr(X, name) for name in require_attrs])


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
    bool
        Whether the object is a Target or not.
    """

    if require_attrs is None:
        require_attrs = (
            name for name in vars(Target) if not name.startswith('_'))

    return all([hasattr(X, name) for name in require_attrs])


def convert_to_ndarray(X, new_dim_last=True, new_dim_name='variable'):
    """ Convert xarray DataArray or Dataset to numpy ndarray.

    Parameters
    ----------
    X : xarray DataArray or Dataset
        The input data.

    new_dim_last : bool, default true
        If true, put the new dimension last when converting a Dataset with
        multiple variables.

    new_dim_name : str, default 'variable'
        The name of the new dimension when converting a Dataset with multiple
        variables.

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


def segment_array(arr, axis, new_len, step=1, new_axis=None, return_view=False):
    """ Segment an array along some axis.

    Parameters
    ----------
    arr : array-like
        The input array.

    axis : int
        The axis along which to segment.

    new_len : int
        The length of each segment.

    step : int, default 1
        The offset between the start of each segment.

    new_axis : int, optional
        The position where the newly created axis is to be inserted. By
        default, the axis will be added at the end of the array.

    return_view : bool, default False
        If True, return a view of the segmented array instead of a copy.

    Returns
    -------
    arr_seg : array-like
        The segmented array.
    """

    from numpy.lib.stride_tricks import as_strided

    # handle the case that the segmented axis is singleton after segmentation
    if (arr.shape[axis] - new_len) // step == 0:
        idx = [slice(None)] * arr.ndim
        idx[axis] = slice(new_len)
        arr_seg = arr[tuple(idx)][..., np.newaxis]
        if new_axis is None:
            return np.moveaxis(arr_seg, (axis, -1), (-1, axis))
        else:
            return np.moveaxis(arr_seg, (axis, -1), (new_axis, axis))

    old_shape = np.array(arr.shape)

    assert new_len <= old_shape[axis], \
        "new_len is bigger than input array in axis"
    seg_shape = old_shape.copy()
    seg_shape[axis] = new_len

    steps = np.ones_like(old_shape)
    if step:
        step = np.array(step, ndmin=1)
        assert step > 0, "Only positive steps allowed"
        steps[axis] = step

    arr_strides = np.array(arr.strides)

    shape = tuple((old_shape - seg_shape) // steps + 1) + tuple(seg_shape)
    strides = tuple(arr_strides * steps) + tuple(arr_strides)

    arr_seg = np.squeeze(
        as_strided(arr, shape=shape, strides=strides))

    # squeeze will move the segmented axis to the first position
    arr_seg = np.moveaxis(arr_seg, 0, axis)

    # the new axis comes right after
    if new_axis is not None:
        arr_seg = np.moveaxis(arr_seg, axis + 1, new_axis)
    else:
        arr_seg = np.moveaxis(arr_seg, axis + 1, -1)

    if return_view:
        return arr_seg
    else:
        return arr_seg.copy()
