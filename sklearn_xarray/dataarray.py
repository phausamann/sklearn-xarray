"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class BaseEstimatorWrapper(BaseEstimator):
    """ A wrapper around sklearn's BaseEstimator for compatibility with xarray.

    Parameters
    ----------
    estimator : sklearn estimator
    reshapes : str or dict
        The dimension reshaped by this estimator
    """
    def __init__(self, estimator=None, reshapes=None):
        self.estimator = estimator
        self.reshapes = reshapes

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
            # drop the reshaped dimensions
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
            # drop the reshaped dimension
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

        yp = self.estimator.predict(X)
        dims_new = list(X.dims)

        # dict syntax
        if hasattr(self.reshapes, 'items'):
            for new_dim, old_dims in self.reshapes.items():
                for d in old_dims:
                    dims_new.remove(d)
                # only if new_dim is not singleton after prediction
                if yp.ndim == X.ndim-len(old_dims)+1:
                    dims_new.append(new_dim)

        else:
            # handle the case that dimension is singleton after prediction
            if yp.ndim < X.ndim:
                dims_new.remove(self.reshapes)

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
            if X.ndim < 3:
                check_X_y(X, y, multi_output=True)

        if self.estimator is not None:
            self.estimator.fit(X, y)

        # Return the estimator
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
            Returns prediction of estimator.
        """

        if X.ndim < 3:
            check_array(X)

        if self.estimator is not None:
            if self.reshapes is not None:
                data, dims = self._predict(X)
                coords = self._update_coords(X)
                return xr.DataArray(data, coords=coords, dims=dims)
            else:
                return xr.DataArray(
                    self.estimator.predict(X), coords=X.coords, dims=X.dims)
        else:
            return X