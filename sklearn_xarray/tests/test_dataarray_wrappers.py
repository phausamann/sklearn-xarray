import numpy as np
import xarray as xr
from xarray.testing import assert_equal, assert_allclose
import numpy.testing as npt

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn_xarray import (
    wrap, TransformerWrapper, ClassifierWrapper, RegressorWrapper)


class DummyEstimator(BaseEstimator):
    """ A dummy estimator that returns the input as a numpy array."""

    def __init__(self, demo_param='demo_param'):

        self.demo_param = demo_param

    def fit(self, X, y=None):

        return self

    def predict(self, X):

        return np.array(X)


class DummyTransformer(BaseEstimator):
    """ A dummy estimator that returns the input as a numpy array."""

    def __init__(self, demo_param='demo_param'):

        self.demo_param = demo_param

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return np.array(X)


class ReshapingEstimator(BaseEstimator, TransformerMixin):
    """ A dummy estimator that changes the number of features."""
    def __init__(self, new_shape=None):
        self.new_shape = new_shape

    def fit(self, X, y=None):

        return self

    def predict(self, X):

        Xt = np.array(X)

        I = [slice(None)]*Xt.ndim
        for i in range(len(self.new_shape)):
            if self.new_shape[i] > 0:
                I[i] = range(self.new_shape[i])
            elif self.new_shape[i] == 0:
                I[i] = 0

        return Xt[tuple(I)]

    def transform(self, X):

        return self.predict(X)


def test_dummy_estimator():

    X = xr.DataArray(
        np.random.random((100, 10)),
        coords={'sample': range(100), 'feature': range(10)},
        dims=['sample', 'feature']
    )

    y = X

    estimator = RegressorWrapper(DummyEstimator())

    estimator.fit(X)
    estimator.fit(X, X)
    yp = estimator.predict(X)

    assert_equal(yp, X)


def test_dummy_transformer():

    X = xr.DataArray(
        np.random.random((100, 10)),
        coords={'sample': range(100), 'feature': range(10)},
        dims=['sample', 'feature']
    )

    estimator = TransformerWrapper(DummyTransformer())

    estimator.fit(X)
    yp = estimator.transform(X)

    assert_equal(yp, X)
    
    
def test_wrapped_transformer():

    from sklearn.preprocessing import StandardScaler

    X = xr.DataArray(
        np.random.random((100, 10)),
        coords={'sample': range(100), 'feature': range(10)},
        dims=['sample', 'feature']
    )

    estimator = TransformerWrapper(StandardScaler()).fit(X)

    assert_allclose(X, estimator.inverse_transform(estimator.transform(X)))


def test_ndim_dummy_estimator():

    X = xr.DataArray(
        np.random.random((100, 10, 10)),
        coords={'sample': range(100), 'feat_1': range(10), 'feat_2': range(10)},
        dims=['sample', 'feat_1', 'feat_2']
    )

    estimator = RegressorWrapper(DummyEstimator())

    estimator.fit(X, X)
    yp = estimator.predict(X)

    assert_equal(yp, X)


def test_reshaping_estimator():

    X = xr.DataArray(
        np.random.random((100, 10)),
        coords={'sample': range(100), 'feature': range(10),
                'dummy': (['sample', 'feature'], np.random.random((100, 10)))},
        dims=['sample', 'feature']
    )

    y = X[:, :2].drop('feature')
    y['dummy'] = y.dummy[:, 0]

    estimator = RegressorWrapper(
        ReshapingEstimator(new_shape=(-1, 2)),
        reshapes='feature'
    )

    estimator.fit(X, X)
    yp = estimator.predict(X)

    assert_allclose(yp, y)


def test_reshaping_transformer():

    X = xr.DataArray(
        np.random.random((100, 10)),
        coords={'sample': range(100), 'feature': range(10),
                'dummy': (['sample', 'feature'], np.random.random((100, 10)))},
        dims=['sample', 'feature']
    )

    y = X[:, :2].drop('feature')
    y['dummy'] = y.dummy[:, 0]

    estimator = TransformerWrapper(
        ReshapingEstimator(new_shape=(-1, 2)),
        reshapes='feature'
    )

    estimator.fit(X, X)
    yp = estimator.transform(X)

    assert_allclose(yp, y)


def test_reshaping_estimator_singleton():

    X = xr.DataArray(
        np.random.random((100, 10)),
        coords={'sample': range(100), 'feature': range(10),
                'dummy': (['sample', 'feature'], np.random.random((100, 10)))},
        dims=['sample', 'feature']
    )

    y = X[:, 0].drop('feature')

    estimator = ClassifierWrapper(
        ReshapingEstimator(new_shape=(-1, 0)),
        reshapes='feature'
    )

    estimator.fit(X, X)
    yp = estimator.predict(X)

    assert_allclose(yp, y)


def test_ndim_reshaping_estimator():

    X = xr.DataArray(
        np.random.random((100, 10, 10)),
        coords={'sample': range(100), 'feat_1': range(10), 'feat_2': range(10),
                'dummy': (['sample', 'feat_1'], np.random.random((100, 10)))},
        dims=['sample', 'feat_1', 'feat_2']
    )

    y = X[:, :5, 0].drop(['feat_1', 'feat_2']).rename({'feat_1': 'feature'})
    y['dummy'] = y.dummy[:, 0]

    estimator = RegressorWrapper(
        ReshapingEstimator(new_shape=(-1, 5, 0)),
        reshapes={'feature': ['feat_1', 'feat_2']}
    )

    estimator.fit(X, X)
    yp = estimator.predict(X)

    assert_allclose(yp, y)


def test_wrap():

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVC

    assert isinstance(wrap(StandardScaler()), TransformerWrapper)
    assert isinstance(wrap(SVC()), ClassifierWrapper)
    assert isinstance(wrap(LinearRegression()), RegressorWrapper)


def test_sample_dim():

    from sklearn.decomposition import PCA

    X = xr.DataArray(
        np.random.random((10, 100)),
        coords={'sample': range(100), 'feature': range(10)},
        dims=['feature', 'sample']
    )

    wrapper = wrap(PCA(n_components=5), reshapes='feature', sample_dim='sample')

    Xt = wrapper.fit_transform(X)
    Xr = wrapper.inverse_transform(Xt)

    npt.assert_equal(Xt.shape, (5, 100))
    npt.assert_equal(Xr.shape, (10, 100))


def test_score():

    from sklearn.linear_model import LinearRegression

    X = xr.DataArray(
        np.random.random((100, 10)),
        coords={'sample': range(100), 'feature': range(10)},
        dims=['sample', 'feature']
    )

    y = np.random.random(100)

    wrapper = wrap(LinearRegression, reshapes='feature').fit(X, y)

    wrapper.score(X, y)
