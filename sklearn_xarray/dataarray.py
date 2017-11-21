"""
This is a module to be used as a reference for building other modules
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
        The estimator this instance wraps around.
    reshapes : str or dict, optional
        The dimension reshaped by this estimator.
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
            for new_dim, old_dims in self.reshapes.items():
                for d in old_dims:
                    dims_new.remove(d)
                # only if new_dim is not singleton after prediction
                if X_out.ndim == X_in.ndim-len(old_dims)+1:
                    dims_new.append(new_dim)

        else:
            # handle the case that dimension is singleton after prediction
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

    def fit(self, X, y=None):
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
            if not hasattr(X, 'ndim') or X.ndim < 3:
                check_X_y(X, y, multi_output=True)

        if self.estimator is not None:
            self.estimator.fit(X, y)

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
