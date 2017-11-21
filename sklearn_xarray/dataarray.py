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
        X : xarray.DataArray
            The input array.

        Returns
        -------
        coords_t : dict
            The array's coords after transformation/prediction.
        """

        coords_t = dict()

        # dict syntax
        if hasattr(self.reshapes, 'items'):
            # drop the reshaped dimensions
            for _, old_dims in self.reshapes.items():
                for c in X.coords:
                    old_dims_in_c = [x for x in X[c].dims if x in old_dims]
                    if any(old_dims_in_c) and c not in old_dims:
                        c_t = X[c].isel(**{d : 0 for d in old_dims_in_c})
                        new_dims = [d for d in X[c].dims if d not in old_dims]
                        coords_t[c] = (new_dims, c_t.drop(old_dims_in_c))
                    elif c not in old_dims:
                        coords_t[c] = X[c]

        # string syntax
        else:
            # drop the reshaped dimension
            for c in X.coords:
                if self.reshapes in X[c].dims and c != self.reshapes:
                    c_t = X[c].isel(**{self.reshapes : 0})
                    new_dims = [d for d in X[c].dims if d != self.reshapes]
                    coords_t[c] = (new_dims, c_t.drop(self.reshapes))
                elif c != self.reshapes:
                    coords_t[c] = X[c]

        return coords_t

    def fit(self, X, y=None):
        """ A wrapper around the fitting function.

        Parameters
        ----------
        X : xarray.DataArray
            The training input samples.
        y : xarray.DataArray
            The target values (class labels in classification, real numbers in
            regression).

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
        X : xarray.DataArray
            The input samples.

        Returns
        -------
        y : xarray.DataArray
            Returns :math:`x^2` where :math:`x` is the first column of `X`.
        """

        if X.ndim < 3:
            check_array(X)

        if self.estimator is not None:
            if self.reshapes is not None:
                return xr.DataArray(
                    self.estimator.predict(X), coords=self._update_coords(X))
            else:
                return xr.DataArray(
                    self.estimator.predict(X), coords=X.coords)
        else:
            return X