"""
The ``sklearn_xarray.preprocessing`` module contains various preprocessing
methods that work on xarray DataArrays and Datasets.
"""

from __future__ import division

import numpy as np
import pandas as pd
import xarray as xr

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .utils import get_group_indices, is_dataarray, is_dataset
from .externals import numpy_groupies as npg


def preprocess(X, function, groupby=None, group_dim='sample', **fit_params):
    """ Wraps preprocessing functions from sklearn for use with xarray types.

    Parameters
    ----------
    X : xarray DataArray or Dataset
        The input data.

    function : callable
        The function to apply to the data. Note that this function cannot
        change the shape of the data.

    groupby : str or list, optional
        Name of coordinate or list of coordinates by which the groups are
        determined.

    group_dim : str, optional
        Name of dimension along which the groups are indexed.

    Returns
    -------
    Xt : xarray DataArray or Dataset
        The transformed data.
    """

    if hasattr(X, 'to_dataset'):
        was_array = True
        Xt = X.to_dataset(name='tmp_var')
    else:
        was_array = False
        Xt = X

    if groupby is None:
        Xt = Xt.apply(function, **fit_params)
    else:
        group_idx = get_group_indices(X, groupby, group_dim)
        Xt_list = []
        for i in group_idx:
            x = Xt.isel(**{group_dim: i})
            Xt_list.append(x.apply(function, **fit_params))
        Xt = xr.concat(Xt_list, dim=group_dim)

    if was_array:
        Xt = Xt['tmp_var'].rename(X.name)

    return Xt


class BaseTransformer(BaseEstimator, TransformerMixin):
    """ Base class for transformers. """

    def _call_groupwise(self, function, X, y=None):
        """ Call a function function on groups of data. """

        group_idx = get_group_indices(X, self.groupby, self.group_dim)
        Xt_list = []
        for i in group_idx:
            x = X.isel(**{self.group_dim: i})
            Xt_list.append(function(x))

        return xr.concat(Xt_list, dim=self.group_dim)

    def fit(self, X, y=None, **fit_params):
        """ Fit estimator to data.

        Parameters
        ----------
        X : xarray DataArray or Dataset
            Training set.

        y : xarray DataArray or Dataset
            Target values.

        Returns
        -------
        self:
            The estimator itself.
        """

        if is_dataset(X):
            self.type_ = 'Dataset'
        elif is_dataarray(X):
            self.type_ = 'DataArray'
        else:
            raise ValueError(
                'The input appears to be neither a DataArray nor a Dataset.')

        return self

    def transform(self, X):
        """ Transform input data.

        Parameters
        ----------
        X : xarray DataArray or Dataset
            The input data.

        Returns
        -------
        Xt : xarray DataArray or Dataset
            The transformed data.
        """

        if self.type_ == 'Dataset' and not is_dataset(X):
            raise ValueError(
                'This estimator was fitted for Dataset inputs, but the '
                'provided X does not seem to be a Dataset.')
        elif self.type_ == 'DataArray' and not is_dataarray(X):
            raise ValueError(
                'This estimator was fitted for DataArray inputs, but the '
                'provided X does not seem to be a DataArray.')

        if self.groupby is not None:
            return self._call_groupwise(self._transform, X)
        else:
            return self._transform(X)

    def inverse_transform(self, X):
        """ Reverse the transformation.

        Parameters
        ----------
        X : xarray DataArray or Dataset
            The input data.

        Returns
        -------
        Xt : xarray DataArray or Dataset
            The transformed data.
        """

        if self.type_ == 'Dataset' and not is_dataset(X):
            raise ValueError(
                'This estimator was fitted for Dataset inputs, but the '
                'provided X does not seem to be a Dataset.')
        elif self.type_ == 'DataArray' and not is_dataarray(X):
            raise ValueError(
                'This estimator was fitted for DataArray inputs, but the '
                'provided X does not seem to be a DataArray.')

        if self.groupby is not None:
            return self._call_groupwise(self._inverse_transform, X)
        else:
            return self._inverse_transform(X)


class Transposer(BaseTransformer):
    """ Reorder data dimensions.

    Parameters
    ----------
    order : list or tuple
        The new order of the dimensions.

    groupby : str or list, optional
        Name of coordinate or list of coordinates by which the groups are
        determined.

    group_dim : str, optional
        Name of dimension along which the groups are indexed.
    """

    def __init__(self, order=None, groupby=None, group_dim='sample'):

        self.order = order

        self.groupby = groupby
        self.group_dim = group_dim

    def fit(self, X, y=None, **fit_params):
        """ Fit the estimator.

        Parameters
        ----------
        X : xarray DataArray or Dataset
            The input data.

        y : None
            For compatibility.

        Returns
        -------
        self :
            The estimator itself.
        """

        super(Transposer, self).fit(X, y, **fit_params)

        # we need to determine the initial order for each variable seperately
        # because they might have a different order than the dataset
        if self.type_ == 'Dataset':
            self.initial_order_ = {
                v: [d for d in X[v].dims if d in self.order]
                for v in X.data_vars}
        else:
            self.initial_order_ = [d for d in X.dims if d in self.order]

        return self

    @staticmethod
    def _transpose_subset(X, target_order):
        """ Transpose X with a subset of X.dims. """

        # remove dims not in X.dims
        new_order = [d for d in reversed(target_order) if d in X.dims]

        # add dims not in target_order
        order = []
        for d in X.dims:
            if d not in target_order:
                order.append(d)
            else:
                order.append(new_order.pop())

        return X.transpose(*order)

    def _transform_var(self, X):
        """ Transform a single variable. """

        if self.order is None:
            return X.transpose()
        elif set(self.order) == set(X.dims):
            return X.transpose(*self.order)
        else:
            return self._transpose_subset(X, self.order)

    def _inverse_transform_var(self, X, initial_order):
        """ Inverse transform a single variable. """

        if self.order is None:
            return X.transpose()
        elif set(initial_order) == set(X.dims):
            return X.transpose(*initial_order)
        else:
            return self._transpose_subset(X, initial_order)

    def _transform(self, X):
        """ Transform. """

        check_is_fitted(self, ['initial_order_'])

        if is_dataset(X):
            return xr.Dataset({
                v: self._transform_var(X[v]) for v in X.data_vars})
        else:
            return self._transform_var(X)

    def _inverse_transform(self, X):
        """ Reverse transform. """

        check_is_fitted(self, ['initial_order_'])

        if is_dataset(X):
            return xr.Dataset({
                v: self._inverse_transform_var(X[v], self.initial_order_[v])
                for v in X.data_vars})
        else:
            return self._inverse_transform_var(X, self.initial_order_)


def transpose(X, return_estimator=False, **fit_params):
    """ Reorders data dimensions.

    Parameters
    ----------
    X : xarray DataArray or Dataset
        The input data.

    return_estimator : bool
        Whether to return the fitted estimator along with the transformed data.

    Returns
    -------
    Xt : xarray DataArray or Dataset
        The transformed data.
    """

    estimator = Transposer(**fit_params)

    Xt = estimator.fit_transform(X)

    if return_estimator:
        return Xt, estimator
    else:
        return Xt


class Splitter(BaseTransformer):
    """ Split along some dimension.

    Parameters
    ----------
    dim : str
        Name of the dimension along which to split.

    new_dim : str
        Name of the newly added dimension.

    new_len : int
        Length of the newly added dimension.

    axis : int
        Axis position where new dimension is to be inserted. If None,
        the dimension will be inserted at the end.

    reduce_index : str
        How to reduce the index of the split dimension.

        - ``'head'`` : Take the first `n` values where `n` is the length of the
          dimension after splitting.
        - ``'subsample'`` : Take every ``new_len`` th value.

    new_index_func : callable
        A function that takes ``new_len`` as a parameter and returns a vector of
        length ``new_len`` to be used as the indices for the new dimension.

    keep_coords_as : str or None
        If set, the coordinate of the split dimension will be kept as a
        separate coordinate with this name. This allows ``inverse_transform``
        to reconstruct the original coordinate.

    groupby : str or list, optional
        Name of coordinate or list of coordinates by which the groups are
        determined.

    group_dim : str, optional
        Name of dimension along which the groups are indexed.
    """

    def __init__(self, dim='sample', new_dim=None, new_len=None, axis=None,
                 reduce_index='subsample', new_index_func=np.arange,
                 keep_coords_as=None, groupby=None, group_dim='sample'):

        self.dim = dim
        self.new_dim = new_dim
        self.new_len = new_len
        self.axis = axis
        self.reduce_index = reduce_index
        self.new_index_func = new_index_func
        self.keep_coords_as = keep_coords_as

        self.groupby = groupby
        self.group_dim = group_dim

    def _transpose_var(self, xt, order=None, dims=None):
        """ Transpose a single variable. """

        xt = xt.to_dataset(name='tmptmp')

        if dims is not None:
            if self.axis is None:
                order = list(dims) + [self.new_dim]
            else:
                order = list(dims)[:self.axis] + [self.new_dim] + \
                        list(dims)[self.axis:]

        return xt.transpose(*order)['tmptmp']

    def _transform(self, X):
        """ Transform. """

        if self.type_ == 'DataArray':
            Xt = X.to_dataset(name='tmp_var')
        else:
            Xt = X

        if None in (self.new_dim, self.new_len):
            raise ValueError('Name and length of new dimension must be '
                             'specified')

        # temporary dimension name
        tmp_dim = 'tmp'

        # reduce indices of original dimension
        trimmed_len = (len(Xt[self.dim])//self.new_len)*self.new_len
        if self.reduce_index == 'subsample':
            dim_idx = np.arange(0, trimmed_len, self.new_len)
        elif self.reduce_index == 'head':
            dim_idx = np.arange(trimmed_len // self.new_len)
        else:
            raise KeyError('Unrecognized mode for index reduction')

        dim_coord = Xt[self.dim][dim_idx]

        # keep the original coord if desired
        if self.keep_coords_as is not None:
            Xt.coords[self.keep_coords_as] = Xt[self.dim]

        # get indices of new dimension
        if self.new_index_func is None:
            new_dim_coord = Xt[self.dim][:self.new_len]
        else:
            new_dim_coord = self.new_index_func(self.new_len)

        # create MultiIndex
        index = pd.MultiIndex.from_product((dim_coord, new_dim_coord),
                                           names=(tmp_dim, self.new_dim))

        # trim length and reshape
        Xt = Xt.isel(**{self.dim: slice(len(index))})
        Xt = Xt.assign(**{self.dim: index}).unstack(self.dim)
        Xt = Xt.rename({tmp_dim: self.dim})

        # move new dimension
        if self.type_ == 'Dataset':
            # we have to transpose each variable individually
            for v in X.data_vars:
                if self.new_dim in Xt[v].dims:
                    Xt[v] = self._transpose_var(Xt[v], dims=X[v].dims)
        else:
            Xt = self._transpose_var(Xt['tmp_var'], dims=X.dims)
            Xt = Xt.rename(X.name)

        return Xt

    def _inverse_transform(self, X):
        """ Reverse transform. """

        # temporary dimension name
        tmp_dim = 'tmp'

        Xt = X.stack(**{tmp_dim: (self.dim, self.new_dim)})

        if self.keep_coords_as is not None:
            Xt[tmp_dim] = Xt[self.keep_coords_as]
            Xt = Xt.drop(self.keep_coords_as)

        # transpose to original dimensions
        Xt = Xt.rename({tmp_dim: self.dim})
        if self.type_ == 'Dataset':
            # we have to transpose each variable individually
            for v in X.data_vars:
                old_dims = list(X[v].dims)
                old_dims.remove(self.new_dim)
                Xt[v] = self._transpose_var(Xt[v], old_dims)
        else:
            old_dims = list(X.dims)
            old_dims.remove(self.new_dim)
            Xt = self._transpose_var(Xt, old_dims)

        return Xt


def split(X, return_estimator=False, **fit_params):
    """ Splits X along some dimension.

    Parameters
    ----------
    X : xarray DataArray or Dataset
        The input data.

    return_estimator : bool
        Whether to return the fitted estimator along with the transformed data.

    Returns
    -------
    Xt : xarray DataArray or Dataset
        The transformed data.
    """

    estimator = Splitter(**fit_params)

    Xt = estimator.fit_transform(X)

    if return_estimator:
        return Xt, estimator
    else:
        return Xt


class Segmenter(BaseTransformer):
    """ Split into segments along some dimension.

    Parameters
    ----------
    dim : str
        Name of the dimension along which to split.

    new_dim : str
        Name of the newly added dimension.

    new_len : int
        Length of the newly added dimension.

    step: int
        Number of values between the start of a segment and the next one.

    axis : int
        Axis position where new dimension is to be inserted. If None,
        the dimension will be inserted at the end.

    reduce_index : str
        How to reduce the index of the split dimension.

        - ``'head'`` : Take the first `n` values where `n` is the length of the
          dimension after segmenting.
        - ``'subsample'`` : Take the values corresponding to the first element
          of every segment.

    new_index_func : callable
        A function that takes ``new_len`` as a parameter and returns a vector of
        length ``new_len`` to be used as the indices for the new dimension.

    keep_coords_as : str or None
        If set, the coordinate of the split dimension will be kept as a
        separate coordinate with this name. This allows ``inverse_transform``
        to reconstruct the original coordinate.

    groupby : str or list, optional
        Name of coordinate or list of coordinates by which the groups are
        determined.

    group_dim : str, optional
        Name of dimension along which the groups are indexed.
    """

    # TODO: put step calculation in fit()?

    def __init__(self, dim='sample', new_dim=None, new_len=None, step=None,
                 axis=None, reduce_index='subsample', new_index_func=np.arange,
                 keep_coords_as=None, groupby=None, group_dim='sample'):

        self.dim = dim
        self.new_dim = new_dim
        self.new_len = new_len
        self.step = step
        self.axis = axis
        self.reduce_index = reduce_index
        self.new_index_func = new_index_func
        self.keep_coords_as = keep_coords_as

        self.groupby = groupby
        self.group_dim = group_dim

    def _transpose_var(self, xt, order=None, dims=None):
        """ Transpose a single variable. """

        xt = xt.to_dataset(name='tmptmp')

        if dims is not None:
            if self.axis is None:
                order = list(dims) + [self.new_dim]
            else:
                order = list(dims)[:self.axis] + [self.new_dim] + \
                        list(dims)[self.axis:]

        return xt.transpose(*order)['tmptmp']

    def _segment_array(self, arr, axis):
        """ Segment an array along some axis. """

        from sklearn_xarray.utils import segment_array

        if self.step is None:
            step = self.new_len
        else:
            step = self.step

        return segment_array(arr, axis, self.new_len, step, self.axis)

    def _rebuild_array(self, arr, axis):
        """ Rebuild an array along some axis. """

        if self.step is None:
            step = self.new_len
        else:
            step = self.step
        # calculate shape before transformation and create empty array
        old_shape = list(arr.shape)
        old_shape[axis] = old_shape[axis] * step + self.new_len - step
        if self.axis is None:
            del old_shape[-1]
        else:
            del old_shape[self.axis]

        # check if the new dimension was inserted before the axis
        if self.axis is not None and self.axis < axis:
            axis_old = axis - 1
        else:
            axis_old = axis

        if np.issubdtype(arr.dtype, np.number):

            # fast aggregate implementation for vars and numeric coords
            old_ranges = [range(s) for s in old_shape]
            if len(old_ranges) > 1:
                mg_ord = [1, 0] + list(range(2, len(old_ranges)))
            else:
                mg_ord = [0]
            idx = np.vstack(
                self._segment_array(np.transpose(g, mg_ord), axis_old).flatten()
                for g in np.meshgrid(*old_ranges))
            return npg.aggregate(
                idx, arr.flatten().T, size=old_shape, func='mean')

        else:

            # slow implementation for non-numeric coords
            arr_old = np.zeros(old_shape, dtype=arr.dtype)

            # get order of transposition for assigning slices to the new array
            order = list(range(arr.ndim-1))
            if self.axis is None:
                order[-1], order[axis] = order[axis], order[-1]
            elif self.axis > axis:
                order[self.axis-1], order[axis] = order[axis], \
                                                  order[self.axis-1]

            # setup up indices
            idx_old = [slice(None)] * len(old_shape)
            idx_new = [slice(None)] * arr.ndim

            # loop over axis
            for n in range(arr.shape[axis]):
                idx_old[axis_old] = n * step + np.arange(self.new_len)
                idx_new[axis] = n
                arr_old[tuple(idx_old)] = np.transpose(arr[idx_new], order)

            return arr_old

    def _transform_var(self, X):
        """ Transform a single variable. """

        if self.dim in X.dims:
            new_dims = list(X.dims)
            if self.axis is None:
                new_dims.append(self.new_dim)
            else:
                new_dims.insert(self.axis, self.new_dim)
            var_t = self._segment_array(
                X.values, tuple(X.dims).index(self.dim))
        else:
            new_dims = X.dims
            var_t = X

        return new_dims, var_t

    def _inverse_transform_var(self, X):
        """ Inverse transform a single variable. """

        if self.dim in X.dims:
            new_dims = list(X.dims)
            new_dims.remove(self.new_dim)
            var_t = self._rebuild_array(
                X.values, tuple(X.dims).index(self.dim))
        else:
            new_dims = X.dims
            var_t = X

        return new_dims, var_t

    def _update_coords(self, X):
        """ Update coordinates. """

        if self.step is None:
            step = self.new_len
        else:
            step = self.step

        # get indices of new dimension
        if self.new_index_func is None:
            new_dim_coords = X[self.dim][:self.new_len]
        else:
            new_dim_coords = self.new_index_func(self.new_len)

        # reduce indices of original dimension
        if self.reduce_index == 'subsample':
            dim_idx = np.arange(
                0, (len(X[self.dim]) - self.new_len + 1), step)
        elif self.reduce_index == 'head':
            dim_idx = np.arange(
                (len(X[self.dim]) - self.new_len + step) // step)
        else:
            raise KeyError('Unrecognized mode for index reduction')

        # assign coordinates
        coords_new = {
            self.dim: X[self.dim].values[dim_idx],
            self.new_dim: new_dim_coords
        }

        for c in X.coords:
            if c != self.dim and self.dim in X[c].dims:
                new_dims = list(X[c].dims)
                if self.axis is None:
                    new_dims.append(self.new_dim)
                else:
                    new_dims.insert(self.axis, self.new_dim)
                coords_new[c] = (new_dims,
                    self._segment_array(X[c].values,
                                        tuple(X[c].dims).index(self.dim)))
            elif c != self.dim:
                coords_new[c] = (X[c].dims, X[c])

        return coords_new

    def _restore_coords(self, X):

        # restore original coord
        coords_old = {
            self.dim: self._rebuild_array(
                X[self.keep_coords_as].values,
                tuple(X[self.keep_coords_as].dims).index(self.dim))
        }

        X = X.drop(self.keep_coords_as)

        for c in X.coords:
            if c not in (self.dim, self.new_dim) and self.dim in X[c].dims:
                new_dims = list(X[c].dims)
                axis = new_dims.index(self.dim)
                new_dims.remove(self.new_dim)
                coords_old[c] = (new_dims,
                    self._rebuild_array(X[c].values, axis))
            elif c not in (self.dim, self.new_dim):
                coords_old[c] = (X[c].dims, X[c])

        return coords_old

    def _transform(self, X):
        """ Transform. """

        if None in (self.new_dim, self.new_len):
            raise ValueError('Name and length of new dimension must be '
                             'specified')

        Xt = X.copy()

        # keep the original coord if desired
        if self.keep_coords_as is not None:
            Xt.coords[self.keep_coords_as] = Xt[self.dim]

        if self.type_ == 'Dataset':
            vars_t = dict()
            for v in Xt.data_vars:
                vars_t[v] = self._transform_var(Xt[v])
            coords_t = self._update_coords(Xt)
            Xt = xr.Dataset(vars_t, coords=coords_t)

        else:
            new_dims, var_t = self._transform_var(Xt)
            coords_t = self._update_coords(Xt)
            Xt = xr.DataArray(var_t, coords=coords_t, dims=new_dims)

        if self.type_ == 'Dataset':
            # we have to transpose each variable individually
            for v in X.data_vars:
                if self.new_dim in Xt[v].dims:
                    Xt[v] = self._transpose_var(Xt[v], dims=X[v].dims)
        else:
            Xt = self._transpose_var(Xt, dims=X.dims)

        return Xt

    def _inverse_transform(self, X):
        """ Reverse transform. """

        if None in (self.new_dim, self.new_len):
            raise ValueError('Name and length of new dimension must be '
                             'specified')

        if self.keep_coords_as is None:
            raise ValueError('keep_coords_as must be specified in order for '
                             'inverse_transform to work.')

        Xt = X.copy()

        if self.type_ == 'Dataset':
            vars_it = {
                v: self._inverse_transform_var(Xt[v]) for v in Xt.data_vars}
            coords_it = self._restore_coords(Xt)
            Xt = xr.Dataset(vars_it, coords=coords_it)

        else:
            new_dims, var_it = self._inverse_transform_var(Xt)
            coords_it = self._restore_coords(Xt)
            Xt = xr.DataArray(var_it, coords=coords_it, dims=new_dims)

        # transpose to original dimensions
        if self.type_ == 'Dataset':
            # we have to transpose each variable individually
            for v in X.data_vars:
                old_dims = list(X[v].dims)
                if self.new_dim in old_dims:
                    old_dims.remove(self.new_dim)
                Xt[v] = self._transpose_var(Xt[v], old_dims)
        else:
            old_dims = list(X.dims)
            old_dims.remove(self.new_dim)
            Xt = self._transpose_var(Xt, old_dims)

        return Xt


def segment(X, return_estimator=False, **fit_params):
    """ Segments X along some dimension.

    Parameters
    ----------
    X : xarray DataArray or Dataset
        The input data.

    return_estimator : bool
        Whether to return the fitted estimator along with the transformed data.

    Returns
    -------
    Xt : xarray DataArray or Dataset
        The transformed data.
    """

    estimator = Segmenter(**fit_params)

    Xt = estimator.fit_transform(X)

    if return_estimator:
        return Xt, estimator
    else:
        return Xt


class Resampler(BaseTransformer):
    """ Resample along some dimension.

    Parameters
    ----------
    freq : str
        Frequency after resampling.

    dim : str
        Name of the dimension along which to resample.

    groupby : str or list, optional
        Name of coordinate or list of coordinates by which the groups are
        determined.

    group_dim : str, optional
        Name of dimension along which the groups are indexed.
    """

    def __init__(self, freq=None, dim='sample', groupby=None,
                 group_dim='sample'):

        self.freq = freq
        self.dim = dim

        self.groupby = groupby
        self.group_dim = group_dim

    def fit(self, X, y=None, **fit_params):
        """ Fit the estimator.

        Parameters
        ----------
        X : xarray DataArray or Dataset
            The input data.

        y : None
            For compatibility.

        Returns
        -------
        self :
            The estimator itself.
        """

        super(Resampler, self).fit(X, y, **fit_params)

        if hasattr(X[self.dim], 'freq'):
            self.initial_freq_ = X[self.dim].freq
        else:
            self.initial_freq_ = None

        return self

    def _transform(self, X):
        """ Transform. """

        import scipy.signal as sig
        from fractions import Fraction

        check_is_fitted(self, ['initial_freq_'])

        if self.freq is None:
            return X

        # resample coordinates along resampling dimension
        # TODO: warn if timestamps are not monotonous
        Xt_dim = X[self.dim].to_dataframe().resample(rule=self.freq).first()

        coords_t = dict()
        for c in X.coords:
            if self.dim in X[c].dims:
                coords_t[c] = (X[c].dims, Xt_dim[c])
            else:
                coords_t[c] = X[c]

        # get the numerator and the denominator for the polyphase resampler
        factor = coords_t[self.dim][1].size / X[self.dim].values.size
        frac = Fraction(factor).limit_denominator(100)
        num, den = frac.numerator, frac.denominator
        # the effective fraction can be a little bigger but not smaller
        if num / den < factor:
            num += 1

        # resample data along resampling dimension
        if self.type_ == 'Dataset':

            vars_t = dict()
            for v in X.data_vars:
                if self.dim in X[v].dims:
                    axis = X[v].dims.index(self.dim)
                    v_t = sig.resample_poly(X[v], num, den, axis=axis)
                    # trim the results because the length might be greater
                    I = [slice(None)] * v_t.ndim
                    I[axis] = np.arange(len(Xt_dim[self.dim]))
                    vars_t[v] = (X[v].dims, v_t[tuple(I)])

            # combine to new dataset
            return xr.Dataset(vars_t, coords=coords_t)

        else:

            axis = X.dims.index(self.dim)
            x_t = sig.resample_poly(X, num, den, axis=axis)
            # trim the results because the length might be greater
            I = [slice(None)] * x_t.ndim
            I[axis] = np.arange(len(Xt_dim[self.dim]))

            # combine to new array
            return xr.DataArray(x_t, coords=coords_t, dims=X.dims)

    def _inverse_transform(self, X):
        """ Reverse transform. """

        raise NotImplementedError(
            'inverse_transform has not yet been implemented for this estimator')


def resample(X, return_estimator=False, **fit_params):
    """ Resamples along some dimension.

    Parameters
    ----------
    X : xarray DataArray or Dataset
        The input data.

    return_estimator : bool
        Whether to return the fitted estimator along with the transformed data.

    Returns
    -------
    Xt : xarray DataArray or Dataset
        The transformed data.
    """

    estimator = Resampler(**fit_params)

    Xt = estimator.fit_transform(X)

    if return_estimator:
        return Xt, estimator
    else:
        return Xt


class Concatenator(BaseTransformer):
    """ Concatenate variables along a dimension.

    Parameters
    ----------
    dim : str
        Name of the dimension along which to concatenate the Dataset.

    new_dim : str
        New name of the dimension, if desired.

    variables : list or tuple
        Names of the variables to concatenate, default all.

    new_var :
        Name of the new variable created by the concatenation.

    new_index_func : function
        A function that takes the length of the concatenated dimension as a
        parameter and returns a vector of this length to be used as the
        index for that dimension.

    return_array: bool
        Whether to return a DataArray when a Dataset was passed.

    groupby : str or list, optional
        Name of coordinate or list of coordinates by which the groups are
        determined.

    group_dim : str, optional
        Name of dimension along which the groups are indexed.
    """

    def __init__(self, dim='feature', new_dim=None, variables=None,
                 new_var='Feature', new_index_func=None,
                 return_array=False, groupby=None, group_dim='sample'):

        self.dim = dim
        self.new_dim = new_dim
        self.variables = variables
        self.new_var = new_var
        self.new_index_func = new_index_func
        self.return_array = return_array

        self.groupby = groupby
        self.group_dim = group_dim

    def _transform(self, X):
        """ Transform. """

        if self.type_ == 'DataArray':
            raise ValueError('The Concatenator can only be applied to Datasets')

        if self.variables is None:

            Xt = xr.concat([X[v] for v in X.data_vars], dim=self.dim)
            if self.new_dim is not None:
                Xt = Xt.rename({self.dim: self.new_dim})

            # return a DataArray if requested
            if self.return_array:
                return Xt
            else:
                return Xt.to_dataset(name=self.new_var)
        else:

            Xt = xr.concat([X[v] for v in self.variables], dim=self.dim)

            if self.new_index_func is not None:
                Xt[self.dim] = self.new_index_func(Xt.sizes[self.dim])

            if self.new_dim is not None:
                Xt = Xt.rename({self.dim: self.new_dim})

            X_list = [X[v] for v in X.data_vars if v not in self.variables]
            X_list.append(Xt.to_dataset(name=self.new_var))

            if self.return_array:
                raise ValueError(
                    'Cannot return a DataArray when a subset of variables is '
                    'concatenated.')
            else:
                return xr.merge(X_list)

    def _inverse_transform(self, X):
        """ Reverse transform. """

        raise NotImplementedError(
            'inverse_transform has not yet been implemented for this estimator')


def concatenate(X, return_estimator=False, **fit_params):
    """ Concatenates variables along a dimension.

    Parameters
    ----------
    X : xarray DataArray or Dataset
        The input data.

    return_estimator : bool
        Whether to return the fitted estimator along with the transformed data.

    Returns
    -------
    Xt : xarray DataArray or Dataset
        The transformed data.
    """

    estimator = Concatenator(**fit_params)

    Xt = estimator.fit_transform(X)

    if return_estimator:
        return Xt, estimator
    else:
        return Xt


class Featurizer(BaseTransformer):
    """ Stack all dimensions and variables except for sample dimension.

    Parameters
    ----------
    sample_dim : str
        Name of the sample dimension.

    feature_dim : str
        Name of the feature dimension.

    var_name : str
        Name of the new variable (for Datasets).

    order : list or tuple
        Order of dimension stacking.

    return_array: bool
        Whether to return a DataArray when a Dataset was passed.

    groupby : str or list, optional
        Name of coordinate or list of coordinates by which the groups are
        determined.

    group_dim : str, optional
        Name of dimension along which the groups are indexed.
    """

    def __init__(self, sample_dim='sample', feature_dim='feature',
                 var_name='Features', order=None, return_array=False,
                 groupby=None, group_dim='sample'):

        self.sample_dim = sample_dim
        self.feature_dim = feature_dim
        self.var_name = var_name
        self.order = order
        self.return_array = return_array

        self.groupby = groupby
        self.group_dim = group_dim

    def _transform_var(self, X):
        """ Transform a single variable. """

        if self.order is not None:
            stack_dims = self.order
        else:
            stack_dims = tuple(set(X.dims) - {self.sample_dim})

        return X.stack(**{self.feature_dim: stack_dims})

    def _transform(self, X):
        """ Transform. """

        # stack all dimensions except for sample dimension
        if self.type_ == 'Dataset':
            X = xr.concat([self._transform_var(X[v]) for v in X.data_vars],
                           dim=self.feature_dim)
            if self.return_array:
                return X
            else:
                return X.to_dataset(name=self.var_name)
        else:
            return self._transform_var(X)

    def _inverse_transform(self, X):
        """ Reverse transform. """

        raise NotImplementedError(
            'inverse_transform has not yet been implemented for this estimator')


def featurize(X, return_estimator=False, **fit_params):
    """ Stacks all dimensions and variables except for sample dimension.

    Parameters
    ----------
    X : xarray DataArray or Dataset
        The input data.

    return_estimator : bool
        Whether to return the fitted estimator along with the transformed data.

    Returns
    -------
    Xt : xarray DataArray or Dataset
        The transformed data.
    """

    estimator = Featurizer(**fit_params)

    Xt = estimator.fit_transform(X)

    if return_estimator:
        return Xt, estimator
    else:
        return Xt


class Sanitizer(BaseTransformer):
    """ Remove elements containing NaNs.

    Parameters
    ----------
    dim : str
        Name of the sample dimension.

    groupby : str or list, optional
        Name of coordinate or list of coordinates by which the groups are
        determined.

    group_dim : str, optional
        Name of dimension along which the groups are indexed.
    """

    def __init__(self, dim='sample', groupby=None, group_dim='sample'):

        self.dim = dim

        self.groupby = groupby
        self.group_dim = group_dim

    def _transform(self, X):
        """ Transform. """

        idx_nan = np.zeros(len(X[self.dim]), dtype=bool)

        if self.type_ == 'Dataset':
            for v in X.data_vars:
                axis = np.delete(np.arange(X[v].ndim),
                                 X[v].dims.index(self.dim))
                idx_nan = idx_nan | np.any(np.isnan(X[v]), axis=tuple(axis))
        else:
            axis = np.delete(np.arange(X.ndim), X.dims.index(self.dim))
            idx_nan = idx_nan | np.any(np.isnan(X), axis=tuple(axis))

        return X.isel(**{self.dim: np.logical_not(idx_nan)})

    def _inverse_transform(self, X):
        """ Reverse transform. """

        raise NotImplementedError(
            'inverse_transform cannot be implemented for this estimator')


def sanitize(X, return_estimator=False, **fit_params):
    """ Removes elements containing NaNs.

    Parameters
    ----------
    X : xarray DataArray or Dataset
        The input data.

    return_estimator : bool
        Whether to return the fitted estimator along with the transformed data.

    Returns
    -------
    Xt : xarray DataArray or Dataset
        The transformed data.
    """

    estimator = Sanitizer(**fit_params)

    Xt = estimator.fit_transform(X)

    if return_estimator:
        return Xt, estimator
    else:
        return Xt


class Reducer(BaseTransformer):
    """ Reduce data along some dimension.

    Parameters
    ----------
    dim : str
        Name of the dimension.

    func : function
        Reduction function.

    groupby : str or list, optional
        Name of coordinate or list of coordinates by which the groups are
        determined.

    group_dim : str, optional
        Name of dimension along which the groups are indexed.
    """

    def __init__(self, dim='feature', func=np.linalg.norm, groupby=None,
                 group_dim='sample'):

        self.dim = dim
        self.func = func

        self.groupby = groupby
        self.group_dim = group_dim

    def _transform(self, X):
        """ Transform. """

        return X.reduce(self.func, dim=self.dim)

    def _inverse_transform(self, X):
        """ Reverse transform. """

        raise NotImplementedError(
            'inverse_transform cannot be implemented for this estimator')


def reduce(X, return_estimator=False, **fit_params):
    """ Reduces data along some dimension.

    Parameters
    ----------
    X : xarray DataArray or Dataset
        The input data.

    return_estimator : bool
        Whether to return the fitted estimator along with the transformed data.

    Returns
    -------
    Xt : xarray DataArray or Dataset
        The transformed data.
    """

    estimator = Reducer(**fit_params)

    Xt = estimator.fit_transform(X)

    if return_estimator:
        return Xt, estimator
    else:
        return Xt
