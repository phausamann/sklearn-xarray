"""
The `sklearn_xarray.preprocessing` module contains various preprocessing
methods that work on xarray DataArrays and Datasets.
"""

import numpy as np
import pandas as pd
import xarray as xr

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .utils import get_group_indices


def _wrap_class(class_handle, X, groupby=None, group_dim='sample', **kwargs):
    """ Wraps a functional interface around the classes defined in this module.

    Parameters
    ----------
    class_handle : class
        The class to be wrapped.
    X : xarray DataArray or Dataset
        The input data.
    groupby : str or list, optional
        Name of coordinate or list of coordinates by which the groups are
        determined.
    group_dim : str, optional
        Name of dimension along which the groups are indexed.

    Returns
    -------
    Xt : xarray DataArray or Dataset
        The preprocessed data.
    """

    if groupby is None:
        return class_handle(**kwargs).fit_transform(X)
    else:
        group_idx = get_group_indices(X, groupby)
        xt = []
        for i in group_idx:
            x = X.isel(**{group_dim: i})
            xt.append(class_handle(**kwargs).fit_transform(x))
        return xr.concat(xt, dim=group_dim)


def preprocess(X, function, groupby=None, group_dim='sample', **kwargs):
    """ Wraps preprocessing functions from sklearn for use with xarray types.

    Parameters
    ----------
    X : xarray DataArray or Dataset
        The input data.
    function : function
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
        The preprocessed data.
    """

    if hasattr(X, 'to_dataset'):
        was_array = True
        Xt = X.to_dataset(name='tmp_var')
    else:
        was_array = False
        Xt = X

    if groupby is None:
        Xt = Xt.apply(function, **kwargs)
    else:
        group_idx = get_group_indices(X, groupby)
        Xt_list = []
        for i in group_idx:
            x = Xt.isel(**{group_dim: i})
            Xt_list.append(x.apply(function, **kwargs))
        Xt = xr.concat(Xt_list, dim=group_dim)

    if was_array:
        Xt = Xt['tmp_var'].rename(X.name)

    return Xt


class BaseTransformer(BaseEstimator, TransformerMixin):
    """ Base class for transformers. """
    
    def fit(self, X, y=None):
        """ Fit estimator to data.

        Parameters
        ----------
        X : xarray DataArray or Dataset
            The training data.
        y : None
            For compatibility.

        Returns
        -------
        self:
            The estimator itself.
        """

        if hasattr(X, 'data_vars'):
            self.type_ = 'Dataset'
        else:
            self.type_ = 'DataArray'

        return self


class Transposer(BaseTransformer):
    """ Reorders data dimensions.

    Parameters
    ----------
    order : list or tuple
        The new order of the dimensions.
    """

    def __init__(self, order=None):

        self.order = order

    def transform(self, X, y=None):
        """ Reorder dimensions.

        Parameters
        ----------
        X : xarray DataArray or Dataset
            The input data.
        y : None
            For compatibility.

        Returns
        -------
        Xt : xarray DataArray or Dataset
            The preprocessed data.
        """

        check_is_fitted(self, ['type_'])

        return X.transpose(*self.order)

    def inverse_transform(self, X, y=None):
        """

        Parameters
        ----------
        X : xarray DataArray or Dataset
            The input data.
        y : None
            For compatibility.

        Returns
        -------
        Xt: xarray DataArray or Dataset
            The transformed data.
        """

        raise NotImplementedError(
            'inverse_transform has not yet been implemented for this estimator')


def transpose(X, groupby=None, group_dim='sample', **kwargs):
    """ Reorders data dimensions.

    Parameters
    ----------
    X : xarray DataArray or Dataset
        The input data.
    groupby : str or list, optional
        Name of coordinate or list of coordinates by which the groups are
        determined.
    group_dim : str, optional
        Name of dimension along which the groups are indexed.

    Returns
    -------
    Xt : xarray DataArray or Dataset
        The preprocessed data.
    """

    return _wrap_class(Transposer, X, groupby, group_dim, **kwargs)


class Splitter(BaseTransformer):
    """ Splits X along some dimension.

    Parameters
    ----------
    dim : str
        Name of the dimension along which to split.
    new_dim : str
        Name of the newly added dimension
    new_len : int
        Length of the newly added dimension.
    reduce_index : str
        How to reduce the index of the split dimension.
        'head' : Take the first `n` values where `n` is the length of the
            dimension after splitting.
        'subsample' : Take every `new_len`th value.
    new_index_func : function
        A function that takes `new_len` as a parameter and returns a vector of
        length `new_len` to be used as the indices for the new dimension.
    keep_coords_as : str or None
        If set, the coordinate of the split dimension will be kept as a
        separate coordinate with this name. This allows `inverse_transform`
        to reconstruct the original coordinate.
    """

    def __init__(self, dim='sample', new_dim=None, new_len=None,
                 reduce_index='subsample', new_index_func=np.arange,
                 keep_coords_as=None):

        self.dim = dim
        self.new_dim = new_dim
        self.new_len = new_len
        self.reduce_index = reduce_index
        self.new_index_func = new_index_func
        self.keep_coords_as = keep_coords_as

    def transform(self, X, y=None):
        """ Split X along some dimension.

        Parameters
        ----------
        X : xarray DataArray or Dataset
            The input data.
        y : None
            For compatibility.

        Returns
        -------
        Xt : xarray DataArray or Dataset
            The preprocessed data.
        """

        check_is_fitted(self, ['type_'])

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

        dim_coords = Xt[self.dim][dim_idx]

        # keep the original coord if desired
        if self.keep_coords_as is not None:
            Xt.coords[self.keep_coords_as] = Xt[self.dim]

        # get indices of new dimension
        if self.new_index_func is None:
            new_dim_coords = Xt[self.dim][:self.new_len]
        else:
            new_dim_coords = self.new_index_func(self.new_len)

        # create MultiIndex
        index = pd.MultiIndex.from_product((dim_coords, new_dim_coords),
                                           names=(tmp_dim, self.new_dim))

        # trim length, reshape and move new dimension to the end
        Xt = Xt.isel(**{self.dim: slice(len(index))})
        Xt = Xt.assign(**{self.dim: index}).unstack(self.dim)
        Xt = Xt.rename({tmp_dim: self.dim})
        Xt = Xt.transpose(*(tuple(X.dims) + (self.new_dim,)))

        if self.type_ == 'DataArray':
            Xt = Xt['tmp_var'].rename(X.name)

        return Xt

    def inverse_transform(self, X, y=None):
        """ Undo the split.

        Parameters
        ----------
        X : xarray DataArray or Dataset
            The input data.
        y : None
            For compatibility.

        Returns
        -------
        Xt: xarray DataArray or Dataset
            The transformed data.
        """

        check_is_fitted(self, ['type_'])

        # temporary dimension name
        tmp_dim = 'tmp'

        Xt = X.stack(**{tmp_dim: (self.dim, self.new_dim)})

        if self.keep_coords_as is not None:
            Xt[tmp_dim] = Xt[self.keep_coords_as]
            Xt = Xt.drop(self.keep_coords_as)

        # tranpose to original dimensions
        Xt = Xt.rename({tmp_dim: self.dim})
        Xt = Xt.transpose(*(tuple(d for d in X.dims if d != self.new_dim)))

        return Xt


def split(X, groupby=None, group_dim='sample', **kwargs):
    """ Splits X along some dimension.

    Parameters
    ----------
    X : xarray DataArray or Dataset
        The input data.
    groupby : str or list, optional
        Name of coordinate or list of coordinates by which the groups are
        determined.
    group_dim : str, optional
        Name of dimension along which the groups are indexed.

    Returns
    -------
    Xt : xarray DataArray or Dataset
        The preprocessed data.
    """

    return _wrap_class(Splitter, X, groupby, group_dim, **kwargs)


class Segmenter(BaseTransformer):
    """ Segments X along some dimension.

    Parameters
    ----------
    dim : str
        Name of the dimension along which to split.
    new_dim : str
        Name of the newly added dimension
    new_len : int
        Length of the newly added dimension.
    step: int
        Number of values between the start of a segment and the next one.
    reduce_index : str
        How to reduce the index of the split dimension.
        'head' : Take the first `n` values where `n` is the length of the
            dimension after segmenting.
        'subsample' : Take the values corresponding to the first element of
            every segment.
    new_index_func : function
        A function that takes `new_len` as a parameter and returns a vector of
        length `new_len` to be used as the indices for the new dimension.
    """

    def __init__(self, dim='sample', new_dim=None, new_len=None, step=None,
                 reduce_index='subsample', new_index_func=np.arange):

        self.dim = dim
        self.new_dim = new_dim
        self.new_len = new_len
        self.step = step
        self.reduce_index = reduce_index
        self.new_index_func = new_index_func

    def _segment_array(self, arr, step, new_len, axis):
        """ Segment an array along some axis.

        Parameters
        ----------
        arr : array-like
            The input array.
        step : int
            The step length.
        new_len : int
            The segment length.
        axis : int
            The axis along which to segment

        Returns
        -------
        arr_new: array-like
            The segmented array.
        """

        new_shape = list(arr.shape)
        new_shape[axis] = (new_shape[axis] - new_len + step) // step
        new_shape.append(new_len)
        arr_new = np.zeros(new_shape)

        idx_old = [slice(None)] * arr.ndim
        idx_new = [slice(None)] * len(new_shape)

        for n in range(new_shape[axis]):
            idx_old[axis] = n * step + np.arange(new_len)
            idx_new[axis] = n
            arr_new[tuple(idx_new)] = arr[idx_old].T

        return arr_new

    def transform(self, X, y=None):
        """ Segments X along some dimension.

        Parameters
        ----------
        X : xarray DataArray or Dataset
            The input data.
        y : None
            For compatibility.

        Returns
        -------
        Xt : xarray DataArray or Dataset
            The preprocessed data.
        """

        check_is_fitted(self, ['type_'])

        if None in (self.new_dim, self.new_len):
            raise ValueError('Name and length of new dimension must be '
                             'specified')

        if self.step is None:
            step = self.new_len
        else:
            step = self.step

        # reduce indices of original dimension
        if self.reduce_index == 'subsample':
            dim_idx = np.arange(0, (len(X[self.dim]) - self.new_len + 1), step)
        elif self.reduce_index == 'head':
            dim_idx = np.arange(
                (len(X[self.dim]) - self.new_len + step) // step)
        else:
            raise KeyError('Unrecognized mode for index reduction')

        # get indices of new dimension
        if self.new_index_func is None:
            new_dim_coords = X[self.dim][:self.new_len]
        else:
            new_dim_coords = self.new_index_func(self.new_len)

        if self.type_ == 'Dataset':

            vars_t = dict()
            for v in X.data_vars:
                if self.dim in X[v].dims:
                    new_dims = list(X[v].dims) + [self.new_dim]
                    xv_t = self._segment_array(X[v].values, step, self.new_len,
                                               X[v].dims.index(self.dim))
                    vars_t[v] = (new_dims, xv_t)

            coords_t = {self.new_dim: new_dim_coords}
            for c in X.coords:
                if self.dim in X[c].dims:
                    coords_t[c] = (X[c].dims, X[c].isel(**{self.dim: dim_idx}))

            return xr.Dataset(vars_t, coords=coords_t)

        else:

            if self.dim in X.dims:
                new_dims = list(X.dims) + [self.new_dim]
                x_t = self._segment_array(X.values, step, self.new_len,
                                           X.dims.index(self.dim))

            coords_t = {self.new_dim: new_dim_coords}
            for c in X.coords:
                if self.dim in X[c].dims:
                    coords_t[c] = (X[c].dims, X[c].isel(**{self.dim: dim_idx}))

            return xr.DataArray(x_t, coords=coords_t, dims=new_dims)

    def inverse_transform(self, X, y=None):
        """

        Parameters
        ----------
        X : xarray DataArray or Dataset
            The input data.
        y : None
            For compatibility.

        Returns
        -------
        Xt: xarray DataArray or Dataset
            The transformed data.
        """

        raise NotImplementedError(
            'inverse_transform has not yet been implemented for this estimator')


def segment(X, groupby=None, group_dim='sample', **kwargs):
    """ Segments X along some dimension.

    Parameters
    ----------
    X : xarray DataArray or Dataset
        The input data.
    groupby : str or list, optional
        Name of coordinate or list of coordinates by which the groups are
        determined.
    group_dim : str, optional
        Name of dimension along which the groups are indexed.

    Returns
    -------
    Xt : xarray DataArray or Dataset
        The preprocessed data.
    """

    return _wrap_class(Segmenter, X, groupby, group_dim, **kwargs)


class Resampler(BaseTransformer):
    """ Resamples along some dimension.

    Parameters
    ----------
    freq : str
        Frequency after resampling.
    dim : str
        Name of the dimension along which to resample.
    """

    def __init__(self, freq=None, dim='sample'):

        self.freq = freq
        self.dim = dim

    def transform(self, X, y=None):
        """ Resamples along some dimension.

        Parameters
        ----------
        X : xarray DataArray or Dataset
            The input data.
        y : None
            For compatibility.

        Returns
        -------
        Xt : xarray DataArray or Dataset
            The preprocessed data.
        """

        import scipy.signal as sig
        from fractions import Fraction

        check_is_fitted(self, ['type_'])

        if self.freq is None:
            return X

        # resample coordinates along resampling dimension
        # TODO: warn if timestamps are not monotonous
        Xt = X[self.dim].to_dataframe().resample(rule=self.freq).first()

        coords_t = dict()
        for c in X.coords:
            if self.dim in X[c].dims:
                coords_t[c] = (X[c].dims, Xt[c])
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
                    I[axis] = np.arange(len(Xt[self.dim]))
                    vars_t[v] = (X[v].dims, v_t[tuple(I)])

            # combine to new dataset
            return xr.Dataset(vars_t, coords=coords_t)

        else:

            if self.dim in X.dims:
                axis = X.dims.index(self.dim)
                x_t = sig.resample_poly(X, num, den, axis=axis)
                # trim the results because the length might be greater
                I = [slice(None)] * x_t.ndim
                I[axis] = np.arange(len(Xt[self.dim]))

            # combine to new array
            return xr.DataArray(x_t, coords=coords_t, dims=X.dims)

    def inverse_transform(self, X, y=None):
        """

        Parameters
        ----------
        X : xarray DataArray or Dataset
            The input data.
        y : None
            For compatibility.

        Returns
        -------
        Xt: xarray DataArray or Dataset
            The transformed data.
        """

        raise NotImplementedError(
            'inverse_transform has not yet been implemented for this estimator')


def resample(X, groupby=None, group_dim='sample', **kwargs):
    """ Resamples along some dimension.

    Parameters
    ----------
    X : xarray DataArray or Dataset
        The input data.
    groupby : str or list, optional
        Name of coordinate or list of coordinates by which the groups are
        determined.
    group_dim : str, optional
        Name of dimension along which the groups are indexed.

    Returns
    -------
    Xt : xarray DataArray or Dataset
        The preprocessed data.
    """

    return _wrap_class(Resampler, X, groupby, group_dim, **kwargs)


class Concatenator(BaseTransformer):
    """ Concatenates variables along a dimension.

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
    """

    def __init__(self, dim='feature', new_dim=None, variables=None,
                 new_var='Feature'):

        self.dim = dim
        self.new_dim = new_dim
        self.variables = variables
        self.new_var = new_var

    def transform(self, X, y=None):
        """ Concatenates variables along a dimension.

        Parameters
        ----------
        X : xarray Dataset
            The input data.
        y : None
            For compatibility.

        Returns
        -------
        Xt : xarray DataArray or Dataset
            The preprocessed data.
        """

        check_is_fitted(self, ['type_'])

        if self.type_ == 'DataArray':
            raise ValueError('The Concatenator can only be applied to Datasets')

        if self.variables is None:
            Xt = xr.concat([X[v] for v in X.data_vars], dim=self.dim)
            if self.new_dim is not None:
                Xt = Xt.rename({self.dim: self.new_dim})
            return Xt.to_dataset(name=self.new_var)
        else:
            Xt = xr.concat([X[v] for v in self.variables], dim=self.dim)
            if self.new_dim is not None:
                Xt = Xt.rename({self.dim: self.new_dim})
            X_list = [X[v] for v in X.data_vars if v not in self.variables]
            X_list.append(Xt.to_dataset(name=self.new_var))
            return xr.merge(X_list)

    def inverse_transform(self, X, y=None):
        """

        Parameters
        ----------
        X : xarray DataArray or Dataset
            The input data.
        y : None
            For compatibility.

        Returns
        -------
        Xt: xarray DataArray or Dataset
            The transformed data.
        """

        raise NotImplementedError(
            'inverse_transform has not yet been implemented for this estimator')


def concatenate(X, groupby=None, group_dim='sample', **kwargs):
    """ Concatenates variables along a dimension.

    Parameters
    ----------
    X : xarray Dataset
        The input data.
    groupby : str or list, optional
        Name of coordinate or list of coordinates by which the groups are
        determined.
    group_dim : str, optional
        Name of dimension along which the groups are indexed.

    Returns
    -------
    Xt : xarray Dataset
        The preprocessed data.
    """

    return _wrap_class(Concatenator, X, groupby, group_dim, **kwargs)


class Featurizer(BaseTransformer):
    """ Stacks all dimensions and variables except for sample dimension.

    Parameters
    ----------
    sample_dim : str
        Name of the sample dimension.
    feature_dim : str
        Name of the feature dimension.
    var_name : str
        Name of the new variable (for Datasets)
    order : list or tuple
        Order of dimension stacking.
    """

    def __init__(self, sample_dim='sample', feature_dim='feature',
                 var_name='Features', order=None):

        self.sample_dim = sample_dim
        self.feature_dim = feature_dim
        self.var_name = var_name
        self.order = order

    def transform(self, X, y=None):
        """ Stacks all dimensions and variables except for sample dimension.

        Parameters
        ----------
        X : xarray DataArray or Dataset
            The input data.
        y : None
            For compatibility.

        Returns
        -------
        Xt : xarray DataArray or Dataset
            The preprocessed data.
        """

        check_is_fitted(self, ['type_'])

        # convert to DataArray if necessary
        if self.type_ == 'Dataset':
            if len(X.data_vars) > 1:
                X = X.to_array(dim='variable')
            else:
                X = X[X.data_vars.pop()]

        # stack all dimensions except for sample dimension
        if self.order is not None:
            stack_dims = self.order
        else:
            stack_dims = tuple(set(X.dims) - {self.sample_dim})

        X = X.stack(**{self.feature_dim: stack_dims})

        if self.type_ == 'Dataset':
            return X.to_dataset(name=self.var_name)
        else:
            return X

    def inverse_transform(self, X, y=None):
        """

        Parameters
        ----------
        X : xarray DataArray or Dataset
            The input data.
        y : None
            For compatibility.

        Returns
        -------
        Xt: xarray DataArray or Dataset
            The transformed data.
        """

        raise NotImplementedError(
            'inverse_transform has not yet been implemented for this estimator')


def featurize(X, groupby=None, group_dim='sample', **kwargs):
    """ Stacks all dimensions and variables except for sample dimension.

    Parameters
    ----------
    X : xarray DataArray or Dataset
        The input data.
    groupby : str or list, optional
        Name of coordinate or list of coordinates by which the groups are
        determined.
    group_dim : str, optional
        Name of dimension along which the groups are indexed.

    Returns
    -------
    Xt : xarray DataArray or Dataset
        The preprocessed data.
    """

    return _wrap_class(Featurizer, X, groupby, group_dim, **kwargs)


class Sanitizer(BaseTransformer):
    """ Removes elements containing NaNs.

    Parameters
    ----------
    dim : str
        Name of the sample dimension
    """

    def __init__(self, dim='sample'):

        self.dim = dim

    def transform(self, X, y=None):
        """ Removes elements containing NaNs.

        Parameters
        ----------
        X : xarray DataArray or Dataset
            The input data.
        y : None
            For compatibility.

        Returns
        -------
        Xt : xarray DataArray or Dataset
            The preprocessed data.
        """

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

    def inverse_transform(self, X, y=None):
        """

        Parameters
        ----------
        X : xarray DataArray or Dataset
            The input data.
        y : None
            For compatibility.

        Returns
        -------
        Xt: xarray DataArray or Dataset
            The transformed data.
        """

        raise NotImplementedError(
            'inverse_transform cannot be implemented for this estimator')


def sanitize(X, groupby=None, group_dim='sample', **kwargs):
    """ Removes elements containing NaNs.

    Parameters
    ----------
    X : xarray DataArray or Dataset
        The input data.
    groupby : str or list, optional
        Name of coordinate or list of coordinates by which the groups are
        determined.
    group_dim : str, optional
        Name of dimension along which the groups are indexed.

    Returns
    -------
    Xt : xarray DataArray or Dataset
        The preprocessed data.
    """

    return _wrap_class(Sanitizer, X, groupby, group_dim, **kwargs)


class Reducer(BaseTransformer):
    """ Reduces data along some dimension.

    Parameters
    ----------
    dim : str
        Name of the dimension.
    func : function
        Reduction function.
    """

    def __init__(self, dim='feature', func=np.linalg.norm):

        self.dim = dim
        self.func = func

    def transform(self, X, y=None):
        """ Reduces data along some dimension.

        Parameters
        ----------
        X : xarray DataArray or Dataset
            The input data.
        y : None
            For compatibility.

        Returns
        -------
        Xt : xarray DataArray or Dataset
            The preprocessed data.
        """

        return X.reduce(self.func, dim=self.dim)

    def inverse_transform(self, X, y=None):
        """

        Parameters
        ----------
        X : xarray DataArray or Dataset
            The input data.
        y : None
            For compatibility.

        Returns
        -------
        Xt: xarray DataArray or Dataset
            The transformed data.
        """

        raise NotImplementedError(
            'inverse_transform cannot be implemented for this estimator')


def reduce(X, groupby=None, group_dim='sample', **kwargs):
    """ Reduces data along some dimension.

    Parameters
    ----------
    X : xarray DataArray or Dataset
        The input data.
    groupby : str or list, optional
        Name of coordinate or list of coordinates by which the groups are
        determined.
    group_dim : str, optional
        Name of dimension along which the groups are indexed.

    Returns
    -------
    Xt : xarray DataArray or Dataset
        The preprocessed data.
    """

    return _wrap_class(Reducer, X, groupby, group_dim, **kwargs)
