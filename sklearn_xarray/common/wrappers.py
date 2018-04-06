""" ``sklearn_xarray.common.wrappers`` """

import types

from sklearn.base import clone
from sklearn.utils.validation import check_X_y, check_array

from .base import (
    _CommonEstimatorWrapper, _ImplementsPredictMixin,
    _ImplementsScoreMixin, _ImplementsTransformMixin,
    _ImplementsFitTransformMixin
)

from sklearn_xarray.utils import is_dataarray, is_dataset, is_target

# mapping from wrapped methods to wrapper methods
_method_map = {
    'predict':
        {'predict': _ImplementsPredictMixin.predict,
         '_predict': _ImplementsPredictMixin._predict},
    'transform':
        {'transform': _ImplementsTransformMixin.transform,
         '_transform': _ImplementsTransformMixin._transform},
    'inverse_transform':
        {'inverse_transform': _ImplementsTransformMixin.inverse_transform,
         '_inverse_transform': _ImplementsTransformMixin._inverse_transform},
    'fit_transform':
        {'fit_transform': _ImplementsFitTransformMixin.fit_transform,
         '_fit_transform': _ImplementsFitTransformMixin._fit_transform},
    'score':
        {'score': _ImplementsScoreMixin.score}
}


def wrap(estimator, reshapes=None, sample_dim=None, compat=False, **kwargs):
    """ Wrap an sklearn estimator for xarray objects.

    Parameters
    ----------
    estimator : sklearn estimator class or instance
        The estimator this instance wraps around.

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

    return EstimatorWrapper(estimator, reshapes=reshapes,
                            sample_dim=sample_dim, compat=compat, **kwargs)


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
                 compat=False, **kwargs):

        if isinstance(estimator, type):
            self.estimator = estimator(**kwargs)
        else:
            self.estimator = estimator

        self.reshapes = reshapes
        self.sample_dim = sample_dim
        self.compat = compat

        self._decorate()

    def __getstate__(self):

        state = self.__dict__.copy()

        for m_wrapped in _method_map:
            if hasattr(self.estimator, m_wrapped):
                for m_self in _method_map[m_wrapped]:
                    state.pop(m_self)

        return state

    def __setstate__(self, state):

        self.__dict__ = state
        self._decorate()

    def _decorate(self):
        """ Decorate this instance with wrapping methods for the estimator. """

        if hasattr(self.estimator, '_estimator_type'):
            setattr(self, '_estimator_type', self.estimator._estimator_type)

        for m_wrapped in _method_map:
            if hasattr(self.estimator, m_wrapped):
                for m_self in _method_map[m_wrapped]:
                    setattr(
                        self, m_self,
                        types.MethodType(_method_map[m_wrapped][m_self], self)
                    )

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


class TransformerWrapper(EstimatorWrapper, _ImplementsTransformMixin,
                         _ImplementsFitTransformMixin):
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
