""" ``sklearn_xarray.common.wrappers`` """

from sklearn.base import clone
from sklearn.utils.validation import check_X_y, check_array

from .base import (
    _CommonEstimatorWrapper, _ImplementsPredictMixin,
    _ImplementsScoreMixin, _ImplementsTransformMixin
)

from sklearn_xarray.utils import is_dataarray, is_dataset, is_target


def wrap(estimator, reshapes=None, sample_dim=None, compat=False):
    """ Wrap an sklearn estimator for xarray objects by guessing its type.

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
    """ A wrapper around sklearn estimators compatible with xarray objects.

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
                 compat=False, **fit_params):

        if isinstance(estimator, type):
            self.estimator = estimator(**fit_params)
        else:
            self.estimator = estimator

        self.reshapes = reshapes
        self.sample_dim = sample_dim
        self.compat = compat

    def fit(self, X, y=None, **fit_params):
        """ A wrapper around the fitting function.

        Parameters
        ----------
        X : xarray DataArray, Dataset other other array-like
            The training input samples.

        y : xarray DataArray, Dataset other other array-like
            The target values.

        Returns
        -------
        Returns self.
        """

        if self.estimator is None:
            raise ValueError('You must specify an estimator instance to wrap.')

        if is_target(y):
            y = y(X)

        if is_dataarray(X):

            self.type_ = 'DataArray'
            self.estimator_ = self._fit(X, y, **fit_params)

        elif is_dataset(X):

            self.type_ = 'Dataset'
            self.estimator_dict_ = {
                v: self._fit(X[v], y, **fit_params) for v in X.data_vars}

        else:

            self.type_ = 'other'
            if y is None:
                X = check_array(X)
            else:
                X, y = check_X_y(X, y)

            self.estimator_ = clone(self.estimator).fit(X, y, **fit_params)

            for v in vars(self.estimator_):
                if v.endswith('_') and not v.startswith('_'):
                    setattr(self, v, getattr(self.estimator_, v))

        return self


class TransformerWrapper(EstimatorWrapper, _ImplementsTransformMixin):
    """ A wrapper around sklearn transformers compatible with xarray objects.

    Parameters
    ----------
    estimator : sklearn estimator
        The estimator this instance wraps around.

    reshapes : str or dict, optional
        The dimension reshaped by this estimator.
    """


class RegressorWrapper(
    EstimatorWrapper, _ImplementsPredictMixin, _ImplementsScoreMixin):
    """ A wrapper around sklearn regressors compatible with xarray objects.

    Parameters
    ----------
    estimator : sklearn estimator
        The estimator this instance wraps around.

    reshapes : str or dict, optional
        The dimension reshaped by this estimator.
    """
    _estimator_type = "regressor"


class ClassifierWrapper(
    EstimatorWrapper, _ImplementsPredictMixin, _ImplementsScoreMixin):
    """ A wrapper around sklearn classifiers compatible with xarray objects.

    Parameters
    ----------
    estimator : sklearn estimator
        The estimator this instance wraps around.

    reshapes : str or dict, optional
        The dimension reshaped by this estimator.
    """
    _estimator_type = "classifier"
