"""
``sklearn_xarray.dataaset.wrappers``
"""

import numpy as np
import xarray as xr
import six
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ..common.wrappers import _CommonEstimatorWrapper
from ..utils import is_target, is_dataset


def wrap(estimator, reshapes=None, sample_dim=None, compat=False):
    """ Wrap an sklearn estimator for xarray Datasets by guessing its type.

    Parameters
    ----------
    estimator : sklearn estimator
        The estimator instance this instance wraps around.

    reshapes : str or dict, optional
        The dimension(s) reshaped by this estimator. Any coordinates in the
        Dataset along these dimensions will be dropped. If the estimator drops
        this dimension (e.g. a binary classifier returning a 1D vector), the
        dimension itself will also be dropped.

        You can specify multiple dimensions mapping to multiple new dimensions
        with a dict whose items are lists of reshaped dimensions, e.g.
        ``{'new_feature': ['feature_1', 'feature_2'], ...}``

    sample_dim : str, optional
        The name of the dimension that represents the samples. By default,
        the wrapper will assume that this is the first dimension in the array.

    compat : bool, optional
        If True, ``set_params``/``get_params`` will only use the wrapper's
        actual parameters and not those of the wrapped estimator. This might
        be necessary when the estimator defines parameters with the same name
        as the wrapper.

    Returns
    -------
    A wrapped estimator.
    """

    if hasattr(estimator, '_estimator_type'):
        if estimator._estimator_type == 'classifier':
            return ClassifierWrapper(
                estimator, reshapes=reshapes, sample_dim=sample_dim,
                compat=compat)
        elif estimator._estimator_type == 'regressor':
            return RegressorWrapper(
                estimator, reshapes=reshapes, sample_dim=sample_dim,
                compat=compat)
        elif estimator._estimator_type == 'clusterer':
            raise NotImplementedError(
                'The wrapper for clustering estimators has not been '
                'implemented yet.')
        else:
            raise ValueError('Could not determine type')
    else:
        if hasattr(estimator, 'transform'):
            return TransformerWrapper(
                estimator, reshapes=reshapes, sample_dim=sample_dim,
                compat=compat)
        else:
            raise ValueError('Could not determine type')


class EstimatorWrapper(_CommonEstimatorWrapper):
    """ A wrapper around sklearn estimators compatible with xarray Datasets.

    Parameters
    ----------
    estimator : sklearn estimator
        The estimator instance this instance wraps around.

    reshapes : str or dict, optional
        The dimension(s) reshaped by this estimator. Any coordinates in the
        Dataset along these dimensions will be dropped. If the estimator drops
        this dimension (e.g. a binary classifier returning a 1D vector), the
        dimension itself will also be dropped.

        You can specify multiple dimensions mapping to multiple new dimensions
        with a dict whose items are lists of reshaped dimensions, e.g.
        ``{'new_feature': ['feature_1', 'feature_2'], ...}``

    sample_dim : str, optional
        The name of the dimension that represents the samples. By default,
        the wrapper will assume that this is the first dimension in the array.

    compat : bool, optional
        If True, ``set`_params``/``get_params`` will only use the wrapper's
        actual parameters and not those of the wrapped estimator. This might
        be necessary when the estimator defines parameters with the same name
        as the wrapper.
    """

    def __init__(self, estimator=None, reshapes=None, sample_dim=None,
                 compat=False, **kwargs):

        self.estimator = estimator
        self.reshapes = reshapes
        self.sample_dim = sample_dim
        self.compat = compat

    def fit(self, X, y=None, **fit_params):
        """ A wrapper around the fitting function.

        Parameters
        ----------
        X : xarray Dataset
            The training input samples.

        y : xarray Dataset
            The target values.

        Returns
        -------
        Returns self.
        """

        if self.estimator is None:
            raise ValueError('You must specify an estimator instance to wrap.')

        if not is_dataset(X):
            raise ValueError('X must be a Dataset.')

        if is_target(y):
            y = y(X)

        self.estimator_dict_ = {
            v: self._fit(X[v], y, **fit_params) for v in X.data_vars}

        return self


class _ImplementsPredictMixin(_CommonEstimatorWrapper):

    def predict(self, X):
        """ A wrapper around the prediction function.

        Parameters
        ----------
        X : xarray Dataset
            The input samples.

        Returns
        -------
        y : xarray Dataset
            The predicted output.
        """

        check_is_fitted(self, ['estimator_dict_'])

        if not is_dataset(X):
            # TODO: check if we need to handle the case when this fails
            X = xr.Dataset(X)

        if self.reshapes is not None:
            data_vars = dict()
            for v, e in six.iteritems(self.estimator_dict_):
                yp_v, dims = self._predict(e, X[v])
                data_vars[v] = (dims, yp_v)
            coords = self._update_coords(X)
            return xr.Dataset(data_vars, coords=coords)
        else:
            data_vars = {
                v: (X[v].dims, e.predict(X[v]))
                for v, e in six.iteritems(self.estimator_dict_)}
            return xr.Dataset(data_vars, coords=X.coords)


class _ImplementsTransformMixin(_CommonEstimatorWrapper, TransformerMixin):

    def transform(self, X):
        """ A wrapper around the transformation function.

        Parameters
        ----------
        X : xarray Dataset
            The input samples.

        Returns
        -------
        y : xarray Dataset
            The transformed output.
        """

        check_is_fitted(self, ['estimator_dict_'])

        if not is_dataset(X):
            # TODO: check if we need to handle the case when this fails
            X = xr.Dataset(X)

        if self.reshapes is not None:
            data_vars = dict()
            for v, e in six.iteritems(self.estimator_dict_):
                yp_v, dims = self._transform(e, X[v])
                data_vars[v] = (dims, yp_v)
            coords = self._update_coords(X)
            return xr.Dataset(data_vars, coords=coords)
        else:
            data_vars = {
                v: (X[v].dims, e.transform(X[v]))
                for v, e in six.iteritems(self.estimator_dict_)}
            return xr.Dataset(data_vars, coords=X.coords)

    def inverse_transform(self, X):
        """ A wrapper around the inverse transformation function.

        Parameters
        ----------
        X : xarray Dataset
            The input samples.

        Returns
        -------
        y : xarray Dataset
            The transformed output.
        """

        check_is_fitted(self, ['estimator_dict_'])

        if not is_dataset(X):
            # TODO: check if we need to handle the case when this fails
            X = xr.Dataset(X)

        if self.reshapes is not None:
            data_vars = dict()
            for v, e in six.iteritems(self.estimator_dict_):
                yp_v, dims = self._inverse_transform(e, X[v])
                data_vars[v] = (dims, yp_v)
            coords = self._update_coords(X)
            return xr.Dataset(data_vars, coords=coords)
        else:
            data_vars = {
                v: (X[v].dims, e.inverse_transform(X[v]))
                for v, e in six.iteritems(self.estimator_dict_)}
            return xr.Dataset(data_vars, coords=X.coords)


class _ImplementsScoreMixin(_CommonEstimatorWrapper):

    def score(self, X, y, sample_weight=None):
        """ Returns the score of the prediction.

        Parameters
        ----------
        X : xarray Dataset or Dataset
            The training set.

        y : xarray Dataset or Dataset
            The target values.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            Score of self.predict(X) wrt. y.
        """

        check_is_fitted(self, ['estimator_dict_'])

        if is_target(y):
            y = y(X)

        score_list = [
            e.score(X[v], y, sample_weight)
            for v, e in six.iteritems(self.estimator_dict_)
        ]

        return np.mean(score_list)


class TransformerWrapper(EstimatorWrapper, _ImplementsTransformMixin):
    """ A wrapper around sklearn transformers compatible with xarray Datasets.

    Parameters
    ----------
    estimator : sklearn estimator
        The estimator this instance wraps around.

    reshapes : str or dict, optional
        The dimension reshaped by this estimator.
    """


class RegressorWrapper(
    EstimatorWrapper, _ImplementsPredictMixin, _ImplementsScoreMixin):
    """ A wrapper around sklearn regressors compatible with xarray Datasets.

    Parameters
    ----------
    estimator : sklearn estimator
        The estimator this instance wraps around.

    reshapes : str or dict, optional
        The dimension reshaped by this estimator.
    """
    # TODO: _estimator_type = "regressor"?


class ClassifierWrapper(
    EstimatorWrapper, _ImplementsPredictMixin, _ImplementsScoreMixin):
    """ A wrapper around sklearn classifiers compatible with xarray Datasets.

    Parameters
    ----------
    estimator : sklearn estimator
        The estimator this instance wraps around.

    reshapes : str or dict, optional
        The dimension reshaped by this estimator.
    """
    # TODO: _estimator_type = "classifier"?
