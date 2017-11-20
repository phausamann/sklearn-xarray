import numpy as np
import xarray as xr
from xarray.testing import assert_equal, assert_allclose

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from sklearn_xarray import BaseEstimatorWrapper


class DummyEstimator(BaseEstimator):
    """ A dummy estimator that returns input as a numpy array.

    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, demo_param='demo_param'):
        self.demo_param = demo_param

    def fit(self, X, y):
        """A reference implementation of a fitting function

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        # X, y = check_X_y(X, y, multi_output=True)
        # Return the estimator
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            Returns :math:`x^2` where :math:`x` is the first column of `X`.
        """
        # X = check_array(X)
        return np.array(X)


class ReshapingEstimator(BaseEstimator):
    """ A dummy estimator that changes the number of features.

    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, new_shape=None):
        self.new_shape = new_shape

    def fit(self, X, y):
        """A reference implementation of a fitting function

        Parameters
        ----------
        X : array-like or sparse matrix
            The training input samples.
        y : array-like
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        # Return the estimator
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : array-like
            The input samples.

        Returns
        -------
        y : array-like
            The reshaped array.
        """

        I = [slice(None)]*X.ndim
        for i in range(len(self.new_shape)):
            if self.new_shape[i] >= 0:
                I[i] = range(self.new_shape[i])

        return X[tuple(I)]


def test_dummy_estimator():
    X = xr.DataArray(np.random.random((100, 10)))
    estimator = BaseEstimatorWrapper(DummyEstimator())
    estimator.fit(X, X)
    yp = estimator.predict(X)
    assert_equal(yp, X)


def test_ndim_dummy_estimator():
    X = xr.DataArray(np.random.random((100, 10, 10)))
    estimator = BaseEstimatorWrapper(DummyEstimator())
    estimator.fit(X, X)
    yp = estimator.predict(X)
    assert_equal(yp, X)


def test_reshaping_estimator():
    X = xr.DataArray(
        np.random.random((100, 10)), dims=['sample', 'feature'])
    estimator = BaseEstimatorWrapper(
        ReshapingEstimator(new_shape=(-1, 2)), reshapes='feature')
    estimator.fit(X, X)
    y = X[:, :2]
    yp = estimator.predict(X)
    assert_allclose(yp, y)


# def test_reshaping_estimator_singleton():
#     X = xr.DataArray(
#         np.random.random((100, 10)), dims=['sample', 'feature'])
#     estimator = BaseEstimatorWrapper(
#         ReshapingEstimator(new_shape=(-1, 1)), reshapes='feature')
#     estimator.fit(X, X)
#     y = X[:, 0]
#     yp = estimator.predict(X)
#     assert_allclose(yp, y)


def test_ndim_reshaping_estimator():
    X = xr.DataArray(
        np.random.random((100, 10, 5)),
        dims=['sample', 'feature_1', 'feature_2'])
    estimator = BaseEstimatorWrapper(
        ReshapingEstimator(new_shape=(-1, 5, 1)),
        reshapes={'feature': ['feature_1', 'feature_2']})
    estimator.fit(X, X)
    y = X[:, :5, :1]
    yp = estimator.predict(X)
    assert_allclose(yp, y)