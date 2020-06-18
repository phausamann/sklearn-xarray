""" ``sklearn_xarray.common.wrappers`` """

from types import MethodType

import warnings

import six
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_X_y, check_array

from .base import (
    partial_fit,
    predict,
    predict_proba,
    predict_log_proba,
    decision_function,
    transform,
    inverse_transform,
    fit_transform,
    score,
    _CommonEstimatorWrapper,
    _ImplementsPredictMixin,
    _ImplementsScoreMixin,
    _ImplementsTransformMixin,
    _ImplementsFitTransformMixin,
    _ImplementsInverseTransformMixin,
)

from sklearn_xarray.utils import is_dataarray, is_dataset, is_target


# mapping from wrapped methods to wrapper methods
_method_map = {
    "partial_fit": partial_fit,
    "predict": predict,
    "predict_proba": predict_proba,
    "predict_log_proba": predict_log_proba,
    "decision_function": decision_function,
    "transform": transform,
    "inverse_transform": inverse_transform,
    "fit_transform": fit_transform,
    "score": score,
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

    compat : bool, default False
        If True, the method will return a ``CompatEstimatorWrapper`` instead
        of an ``EstimatorWrapper``. This might be necessary when the
        estimator defines parameters with the same name as the wrapper.

    Returns
    -------
    A wrapped estimator.
    """

    if compat:
        return CompatEstimatorWrapper(
            estimator=estimator,
            reshapes=reshapes,
            sample_dim=sample_dim,
            **kwargs
        )
    else:
        return EstimatorWrapper(
            estimator=estimator,
            reshapes=reshapes,
            sample_dim=sample_dim,
            **kwargs
        )


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
    """

    def __init__(
        self, estimator=None, reshapes=None, sample_dim=None, **kwargs
    ):

        if "compat" in kwargs:
            kwargs.pop("compat")
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn(
                "The compat argument of EstimatorWrapper is deprecated and "
                "will be removed in a future version.",
                DeprecationWarning,
            )
            warnings.simplefilter("ignore", DeprecationWarning)

        if isinstance(estimator, type):
            self.estimator = estimator(**kwargs)
            params = self.estimator.get_params()
        else:
            self.estimator = estimator
            params = estimator.get_params()
            params.update(kwargs)

        self.reshapes = reshapes
        self.sample_dim = sample_dim

        for p in params:
            setattr(self, p, params[p])

        self._param_names = (
            self._get_param_names() + self.estimator._get_param_names()
        )

        self._decorate()

    def __getstate__(self):

        state = self.__dict__.copy()

        for m in _method_map:
            if hasattr(self.estimator, m):
                state.pop(m)

        return state

    def __setstate__(self, state):

        self.__dict__ = state
        self._decorate()

    def _decorate(self):
        """ Decorate this instance with wrapping methods for the estimator. """

        # TODO: check if this needs to be removed for compat wrappers
        if hasattr(self.estimator, "_estimator_type"):
            setattr(self, "_estimator_type", self.estimator._estimator_type)

        for m in _method_map:
            if hasattr(self.estimator, m):
                if six.PY2:
                    setattr(
                        self,
                        m,
                        MethodType(_method_map[m], self, EstimatorWrapper),
                    )
                else:
                    setattr(self, m, MethodType(_method_map[m], self))

    def _make_estimator(self):
        """ Return an instance of the wrapped estimator. """

        params = {
            p: getattr(self, p) for p in self.estimator._get_param_names()
        }

        return type(self.estimator)(**params)

    def _reset(self):
        """ Reset internal data-dependent state of the wrapper.

        __init__ parameters are not touched.
        """

        for v in vars(self).copy():
            if v.endswith("_") and not v.startswith("_"):
                delattr(self, v)

    def get_params(self, deep=True):
        """ Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """

        # TODO: check if this causes problems for wrapped nested estimators
        params = BaseEstimator.get_params(self, deep=False)
        params.update({p: getattr(self, p) for p in self._param_names})

        return params

    def set_params(self, **params):
        """ Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """

        for p in self._param_names:
            if p in params:
                setattr(self, p, params[p])

        return self

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
            raise ValueError("You must specify an estimator instance to wrap.")

        self._reset()

        if is_target(y):
            y = y(X)

        if is_dataarray(X):

            self.type_ = "DataArray"
            self.estimator_ = self._fit(X, y, **fit_params)

            # TODO: check if this needs to be removed for compat wrappers
            for v in vars(self.estimator_):
                if v.endswith("_") and not v.startswith("_"):
                    setattr(self, v, getattr(self.estimator_, v))

        elif is_dataset(X):

            self.type_ = "Dataset"
            self.estimator_dict_ = {
                v: self._fit(X[v], y, **fit_params) for v in X.data_vars
            }

            # TODO: check if this needs to be removed for compat wrappers
            for e_name, e in six.iteritems(self.estimator_dict_):
                for v in vars(e):
                    if v.endswith("_") and not v.startswith("_"):
                        if hasattr(self, v):
                            getattr(self, v).update({e_name: getattr(e, v)})
                        else:
                            setattr(self, v, {e_name: getattr(e, v)})

        else:

            self.type_ = "other"
            if y is None:
                X = check_array(X)
            else:
                X, y = check_X_y(X, y)

            self.estimator_ = self._make_estimator().fit(X, y, **fit_params)

            # TODO: check if this needs to be removed for compat wrappers
            for v in vars(self.estimator_):
                if v.endswith("_") and not v.startswith("_"):
                    setattr(self, v, getattr(self.estimator_, v))

        return self


class CompatEstimatorWrapper(EstimatorWrapper):
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
    """

    def __init__(
        self, estimator=None, reshapes=None, sample_dim=None, **kwargs
    ):

        if isinstance(estimator, type):
            self.estimator = estimator(**kwargs)
        else:
            self.estimator = estimator
            self.estimator.set_params(**kwargs)

        self.reshapes = reshapes
        self.sample_dim = sample_dim

        if "compat" in kwargs:
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn(
                "The compat argument of EstimatorWrapper is deprecated and "
                "will be removed in a future version.",
                DeprecationWarning,
            )
            warnings.simplefilter("ignore", DeprecationWarning)

        self._decorate()

    def _make_estimator(self):
        """ Return an instance of the wrapped estimator. """

        return clone(self.estimator)

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """

        return BaseEstimator.get_params(self, deep=deep)

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self
        """

        return BaseEstimator.set_params(self, **params)


class TransformerWrapper(
    EstimatorWrapper,
    _ImplementsTransformMixin,
    _ImplementsFitTransformMixin,
    _ImplementsInverseTransformMixin,
):
    """ A wrapper around sklearn transformers compatible with xarray objects.

    Parameters
    ----------
    estimator : sklearn estimator
        The estimator this instance wraps around.

    reshapes : str or dict, optional
        The dimension reshaped by this estimator.
    """


class RegressorWrapper(
    EstimatorWrapper, _ImplementsPredictMixin, _ImplementsScoreMixin
):
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
    EstimatorWrapper, _ImplementsPredictMixin, _ImplementsScoreMixin
):
    """ A wrapper around sklearn classifiers compatible with xarray objects.

    Parameters
    ----------
    estimator : sklearn estimator
        The estimator this instance wraps around.

    reshapes : str or dict, optional
        The dimension reshaped by this estimator.
    """

    _estimator_type = "classifier"
