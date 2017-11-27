"""
`sklearn_xarray.dataarray.wrappers`
"""

import numpy as np
import xarray as xr
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from utils import is_target, is_dataarray


def wrap(estimator, reshapes=None):
    """ Wrap an sklearn estimators by guessing its type.

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

    Returns
    -------
    A wrapped estimator.
    """

    if hasattr(estimator, '_estimator_type'):
        if estimator._estimator_type == 'classifier':
            return ClassifierWrapper(estimator, reshapes=reshapes)
        elif estimator._estimator_type == 'regressor':
            return RegressorWrapper(estimator, reshapes=reshapes)
        elif estimator._estimator_type == 'clusterer':
            raise NotImplementedError(
                'The wrapper for clustering estimators has not been '
                'implemented yet.')
        else:
            raise ValueError('Could not determine type')
    else:
        if hasattr(estimator, 'transform'):
            return TransformerWrapper(estimator, reshapes=reshapes)
        else:
            raise ValueError('Could not determine type')




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

    def __init__(self, estimator=None, reshapes=None, **kwargs):

        self.estimator = estimator
        self.reshapes = reshapes

    def _update_dims(self, X_in, X_out):
        """ Update the dimensions of a reshaped DataArray.

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
        """ Update the coordinates of a reshaped DataArray.

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

    def get_params(self, deep=True, compat=False):
        """ Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        compat : boolean, optional
            If True, will only return the wrapper's actual parameters and not
            those of the wrapped estimator. This might be necessary when the
            estimator defines parameters with the same name as the wrapper.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """

        if compat:
            return BaseEstimator.get_params(self, deep)

        else:
            if self.estimator is not None:
                params = self.estimator.get_params(deep)
            else:
                # TODO: check if this is necessary
                params = dict()

            for p in self._get_param_names():
                params[p] = getattr(self, p, None)

            return params

    def set_params(self, compat=False, **params):
        """ Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        compat : boolean, optional
            If True, will only set the wrapper's actual parameters and not
            those of the wrapped estimator. This might be necessary when the
            estimator defines parameters with the same name as the wrapper.

        Returns
        -------
        self
        """

        if compat:
            BaseEstimator.set_params(self, **params)

        else:
            for p in self._get_param_names():
                if p in params:
                    setattr(self, p, params.pop(p))

            self.estimator.set_params(**params)

        return self

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

        if self.estimator is None:
            raise ValueError('You must specify an estimator instance to wrap.')

        if not is_dataarray(X):
            if y is None:
                X = check_array(X)
            else:
                X, y = check_X_y(X, y, multi_output=True)

        if is_target(y):
            y = y(X)

        self.estimator_ = clone(self.estimator).fit(X, y, **fit_params)

        return self

class ImplementsPredictMixin(object):

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

        coords_new : dict
            The array's coordinates after prediction.

        dims_new : list
            The array's dimensions after prediction.
        """

        yp = self.estimator_.predict(np.array(X))
        coords_new = self._update_coords(X)
        dims_new = self._update_dims(X, yp)

        return yp, coords_new, dims_new

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

        check_is_fitted(self, ['estimator_'])

        if not is_dataarray(X):
            # TODO: check if we need to handle the case when this fails
            X = xr.DataArray(X)

        if self.reshapes is not None:
            data, coords, dims = self._predict(X)
            return xr.DataArray(data, coords=coords, dims=dims)
        else:
            return xr.DataArray(
                self.estimator_.predict(X), coords=X.coords, dims=X.dims)


class ImplementsScoreMixin(object):

    def score(self, X, y, sample_weight=None):
        """ Returns the score of the prediction.

        Parameters
        ----------
        X : xarray DataArray or Dataset
            The training set.

        y : xarray DataArray or Dataset
            The target values.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            Score of self.predict(X) wrt. y.
        """

        check_is_fitted(self, ['estimator_'])

        if is_target(y):
            y = y(X)

        return self.estimator_.score(X, y, sample_weight)


class TransformerWrapper(EstimatorWrapper, TransformerMixin):
    """ A wrapper around sklearn transformers compatible with xarray DataArrays.

    Parameters
    ----------
    estimator : sklearn estimator
        The estimator this instance wraps around.

    reshapes : str or dict, optional
        The dimension reshaped by this estimator.
    """

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

        coords_new : dict
            The array's coordinates after transformation.

        dims_new : list
            The array's dims after transformation.
        """

        Xt = self.estimator_.transform(np.array(X))
        coords_new = self._update_coords(X)
        dims_new = self._update_dims(X, Xt)

        return Xt, coords_new, dims_new

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

        check_is_fitted(self, ['estimator_'])

        if not is_dataarray(X):
            # TODO: check if we need to handle the case when this fails
            X = xr.DataArray(X)

        if self.reshapes is not None:
            data, coords, dims = self._transform(X)
            return xr.DataArray(data, coords=coords, dims=dims)
        else:
            return xr.DataArray(
                self.estimator_.transform(X), coords=X.coords, dims=X.dims)


class RegressorWrapper(
    EstimatorWrapper, ImplementsPredictMixin, ImplementsScoreMixin):
    """ A wrapper around sklearn regressors compatible with xarray DataArrays.

    Parameters
    ----------
    estimator : sklearn estimator
        The estimator this instance wraps around.

    reshapes : str or dict, optional
        The dimension reshaped by this estimator.

    groupby : str or list, optional
        Name of coordinate or list of coordinates by which the groups are
        determined.

    group_dim : str, optional
        Name of dimension along which the groups are indexed.
    """
    # TODO: _estimator_type = "regressor"?


class ClassifierWrapper(
    EstimatorWrapper, ImplementsPredictMixin, ImplementsScoreMixin):
    """ A wrapper around sklearn classifiers compatible with xarray DataArrays.

    Parameters
    ----------
    estimator : sklearn estimator
        The estimator this instance wraps around.

    reshapes : str or dict, optional
        The dimension reshaped by this estimator.

    groupby : str or list, optional
        Name of coordinate or list of coordinates by which the groups are
        determined.

    group_dim : str, optional
        Name of dimension along which the groups are indexed.
    """
    # TODO: _estimator_type = "classifier"?
