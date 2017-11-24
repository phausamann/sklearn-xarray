"""
`sklearn_xarray.dataarray.wrappers`
"""

import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class EstimatorWrapper(BaseEstimator):
    """ A wrapper around sklearn estimators compatible with xarray DataArrays.

    Parameters
    ----------
    estimator : sklearn estimator
        The estimator instance this instance wraps around.
    reshapes : str or dict, optional
        The dimension(s) reshaped by this estimator. Any coordinates in the
        DataArray along these dimensions will be dropped. If the estimator drops
        this dimension (e.g. a binary classifier returning a 1D vector), the
        dimension itself will also be dropped.

        You can specify multiple dimensions mapping to multiple new dimensions
        with a dict whose items are lists of reshaped dimensions, e.g.
        {'new_feature': ['feature_1', 'feature_2'], ...}
    """

    def __init__(self, estimator=None, reshapes=None):

        self.estimator = estimator
        self.reshapes = reshapes

    def _update_dims(self, X_in, X_out):
        """ Predict with `self.estimator` and update dims.

        Parameters
        ----------
        X_in : xarray DataArray
            The input array.
        X_out : xarray DataArray
            The output array.

        Returns
        -------
        dims_new : list
            The output array's dims.
        """

        dims_new = list(X_in.dims)

        # dict syntax
        if hasattr(self.reshapes, 'items'):

            # check if new dims are dropped by estimator
            all_old_dims = []
            for _, old_dims in self.reshapes.items():
                all_old_dims += old_dims

            if X_out.ndim == X_in.ndim-len(all_old_dims)+len(self.reshapes):
                drop_new_dims = False
            elif X_out.ndim == X_in.ndim-len(all_old_dims):
                drop_new_dims = True
            else:
                raise ValueError(
                    'Inconsistent dimensions returned by estimator')

            # get new dims
            for new_dim, old_dims in self.reshapes.items():
                for d in old_dims:
                    dims_new.remove(d)
                if not drop_new_dims:
                    dims_new.append(new_dim)

        # string syntax
        else:
            # check if dim is dropped by estimator
            if X_out.ndim < X_in.ndim:
                dims_new.remove(self.reshapes)

        return dims_new

    def _update_coords(self, X):
        """ Update the coords of a reshaped/resampled DataArray.

        Parameters
        ----------
        X : xarray DataArray
            The input array.

        Returns
        -------
        coords_new : dict
            The array's coords after transformation/prediction.
        """

        coords_new = dict()

        # dict syntax
        if hasattr(self.reshapes, 'items'):
            # drop all coords along the reshaped dimensions
            for _, old_dims in self.reshapes.items():
                for c in X.coords:
                    old_dims_in_c = [x for x in X[c].dims if x in old_dims]
                    if any(old_dims_in_c) and c not in old_dims:
                        c_t = X[c].isel(**{d: 0 for d in old_dims_in_c})
                        new_dims = [d for d in X[c].dims if d not in old_dims]
                        coords_new[c] = (new_dims, c_t.drop(old_dims_in_c))
                    elif c not in old_dims:
                        coords_new[c] = X[c]

        # string syntax
        else:
            # drop all coords along the reshaped dimensions
            for c in X.coords:
                if self.reshapes in X[c].dims and c != self.reshapes:
                    c_t = X[c].isel(**{self.reshapes: 0})
                    new_dims = [d for d in X[c].dims if d != self.reshapes]
                    coords_new[c] = (new_dims, c_t.drop(self.reshapes))
                elif c != self.reshapes:
                    coords_new[c] = X[c]

        return coords_new

    def _predict(self, X):
        """ Predict with `self.estimator` and update dims.

        Parameters
        ----------
        X : xarray DataArray
            The input array.

        Returns
        -------
        yp: array-like
            The predicted output.
        dims_new : dict
            The array's dims after prediction.
        """

        yp = self.estimator.predict(np.array(X))
        dims_new = self._update_dims(X, yp)

        return yp, dims_new

    def fit(self, X, y=None, **fit_params):
        """ A wrapper around the fitting function.

        Parameters
        ----------
        X : xarray DataArray
            The training input samples.
        y : xarray DataArray
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """

        if y is None:
            if X.ndim < 3:
                check_array(X)
        else:
            if hasattr(y, 'assign_to'):
                y = y.assign_to(X)
            if not hasattr(X, 'ndim') or X.ndim < 3:
                check_X_y(X, y, multi_output=True)

        if self.estimator is not None:
            self.estimator.fit(X, y, **fit_params)

        return self

    def predict(self, X):
        """ A wrapper around the predicting function.

        Parameters
        ----------
        X : xarray DataArray
            The input samples.

        Returns
        -------
        y : xarray DataArray
            The predicted output.
        """

        X_arr = np.array(X)

        if X_arr.ndim < 3:
            check_array(X_arr)

        if self.estimator is not None:
            if self.reshapes is not None:
                data, dims = self._predict(X)
                coords = self._update_coords(X)
                return xr.DataArray(data, coords=coords, dims=dims)
            else:
                return xr.DataArray(
                    self.estimator.predict(X), coords=X.coords, dims=X.dims)
        else:
            return xr.DataArray(X_arr)


class TransformerWrapper(EstimatorWrapper):
    """ A wrapper around sklearn transformers compatible with xarray DataArrays.

    Parameters
    ----------
    estimator : sklearn estimator
        The estimator this instance wraps around.
    reshapes : str or dict, optional
        The dimension reshaped by this estimator.
    """

    # TODO: check if it's possible to inherit from TransformerMixin

    def _transform(self, X):
        """ Transform with `self.estimator` and update dims.

        Parameters
        ----------
        X : xarray DataArray
            The input array.

        Returns
        -------
        Xt: array-like
            The transformed output.
        dims_new : dict
            The array's dims after transformation.
        """

        Xt = self.estimator.transform(np.array(X))
        dims_new = self._update_dims(X, Xt)

        return Xt, dims_new

    def transform(self, X):
        """ A wrapper around the transformation function.

        Parameters
        ----------
        X : xarray DataArray
            The input samples.

        Returns
        -------
        y : xarray DataArray
            The transformed output.
        """

        X_arr = np.array(X)

        if X_arr.ndim < 3:
            check_array(X_arr, estimator=self)

        if self.estimator is not None:
            if self.reshapes is not None:
                data, dims = self._transform(X)
                coords = self._update_coords(X)
                return xr.DataArray(data, coords=coords, dims=dims)
            else:
                return xr.DataArray(
                    self.estimator.transform(X), coords=X.coords, dims=X.dims)
        else:
            return xr.DataArray(X_arr)

    def fit_transform(self, X, y=None, **fit_params):
        """ Fit to data, then transform it.

        Fits transformer to X and y with optional parameters kwargs
        and returns a transformed version of X.

        Parameters
        ----------
        X : xarray DataArray or Dataset
            The training set.
        y : xarray DataArray or Dataset
            The target values.

        Returns
        -------
        X_new : xarray DataArray or Dataset
            The transformed data.
        """

        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)
