from unittest import TestCase

import numpy as np
import xarray as xr
from xarray.testing import assert_equal, assert_allclose
import numpy.testing as npt

from sklearn.utils.estimator_checks import check_estimator
from sklearn_xarray import wrap

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

from sklearn_xarray.tests.mocks import (
    DummyEstimator, DummyTransformer, ReshapingEstimator
)


class EstimatorWrapperTests(TestCase):

    def setUp(self):

        self.X = xr.Dataset(
            {'var_2d': (['sample', 'feat_1'],
                        np.random.random((100, 10))),
             'var_3d': (['sample', 'feat_1', 'feat_2'],
                        np.random.random((100, 10, 10)))},
            {'sample': range(100), 'feat_1': range(10), 'feat_2': range(10),
             'dummy': (['sample', 'feat_1'], np.random.random((100, 10)))})

    def test_update_restore_dims(self):

        estimator = wrap(
            ReshapingEstimator(new_shape=(-1, 0, 5)),
            reshapes={'feature': ['feat_1', 'feat_2']}
        )

        X = self.X.var_3d

        estimator.fit(X)

        X_out = estimator.estimator_.transform(X.values)
        dims_new = estimator._update_dims(X, X_out)
        Xt = xr.DataArray(X_out, dims=dims_new)

        assert dims_new == ['sample', 'feature']

        Xr_out = estimator.estimator_.inverse_transform(X_out)
        dims_old = estimator._restore_dims(Xt, Xr_out)

        assert dims_old == ['sample', 'feat_1', 'feat_2']

    def test_update_coords(self):

        pass


class PublicInterfaceTests(TestCase):

    def setUp(self):

        self.X = xr.Dataset(
            {'var_2d': (['sample', 'feat_1'],
                        np.random.random((100, 10))),
             'var_3d': (['sample', 'feat_1', 'feat_2'],
                        np.random.random((100, 10, 10)))},
            {'sample': range(100), 'feat_1': range(10), 'feat_2': range(10),
             'dummy': (['sample', 'feat_1'], np.random.random((100, 10)))})

    def test_dummy_estimator(self):

        estimator = wrap(DummyEstimator())

        # test DataArray
        X_da = self.X.var_2d

        estimator.fit(X_da)
        yp = estimator.predict(X_da)

        assert_equal(yp, X_da)

        # test Dataset
        X_ds = self.X

        estimator.fit(X_ds)
        yp = estimator.predict(X_ds)

        assert_equal(yp, X_ds)

    def test_dummy_transformer(self):

        estimator = wrap(DummyTransformer())

        # test DataArray
        X_da = self.X.var_2d

        estimator.fit(X_da)
        yp = estimator.transform(X_da)

        assert_equal(yp, X_da)

        # test Dataset
        X_ds = self.X

        estimator.fit(X_ds)
        yp = estimator.transform(X_ds)

        assert_equal(yp, X_ds)

    def test_wrapped_transformer(self):

        from sklearn.preprocessing import StandardScaler

        estimator = wrap(StandardScaler())

        # test DataArray
        X_da = self.X.var_2d

        estimator.partial_fit(X_da)

        assert_allclose(
            X_da, estimator.inverse_transform(estimator.transform(X_da)))

        # test Dataset
        X_ds = self.X.var_2d.to_dataset()

        estimator.fit(X_ds)

        assert_allclose(
            X_ds, estimator.inverse_transform(estimator.transform(X_ds)))

    def test_ndim_dummy_estimator(self):

        estimator = wrap(DummyEstimator())

        # test DataArray
        X_da = self.X.var_3d

        estimator.fit(X_da)
        yp = estimator.predict(X_da)

        assert_equal(yp, X_da)

        # test Dataset
        X_ds = self.X

        estimator.fit(X_ds)
        yp = estimator.predict(X_ds)

        assert_equal(yp, X_ds)

    def test_reshaping_estimator(self):

        estimator = wrap(
            ReshapingEstimator(new_shape=(-1, 2)),
            reshapes='feat_1'
        )

        # test DataArray
        X_da = self.X.var_2d

        y = X_da[:, :2].drop('feat_1')
        y['dummy'] = y.dummy[:, 0]

        estimator.fit(X_da)
        yp = estimator.predict(X_da)

        assert_allclose(yp, y)

        # test Dataset
        X_ds = self.X.var_2d.to_dataset()

        y = X_ds.var_2d[:, :2].drop('feat_1')
        y['dummy'] = y.dummy[:, 0]

        estimator.fit(X_ds)
        yp = estimator.predict(X_ds).var_2d

        assert_allclose(yp, y)

    def test_reshaping_transformer(self):

        estimator = wrap(
            ReshapingEstimator(new_shape=(-1, 2)),
            reshapes='feat_1'
        )

        # test DataArray
        X_da = self.X.var_3d

        y = X_da[:, :2].drop('feat_1')
        y['dummy'] = y.dummy[:, 0]

        estimator.fit(X_da)
        yp = estimator.transform(X_da)

        assert_allclose(yp, y)

        # test Dataset
        X_ds = self.X.var_2d.to_dataset()

        y = X_ds.var_2d[:, :2].drop('feat_1')
        y['dummy'] = y.dummy[:, 0]

        estimator.fit(X_ds)
        yp = estimator.transform(X_ds).var_2d

        assert_allclose(yp, y)

    def test_reshaping_estimator_singleton(self):

        estimator = wrap(
            ReshapingEstimator(new_shape=(-1, 0)),
            reshapes='feat_1'
        )

        # test DataArray
        X_da = self.X.var_2d

        y = X_da[:, 0].drop('feat_1')
        estimator.fit(X_da)
        yp = estimator.predict(X_da)

        assert_allclose(yp, y)

        # test Dataset
        X_ds = self.X

        y = X_ds.var_2d[:, 0].drop('feat_1')

        estimator.fit(X_ds)
        yp = estimator.predict(X_ds).var_2d

        assert_allclose(yp, y)

    def test_ndim_reshaping_estimator(self):

        estimator = wrap(
            ReshapingEstimator(new_shape=(-1, 5, 0)),
            reshapes={'feature': ['feat_1', 'feat_2']}
        )

        # test DataArray
        X_da = self.X.var_3d

        Xt = X_da[:, :5, 0].drop(
            ['feat_1', 'feat_2']).rename({'feat_1': 'feature'})
        Xt['dummy'] = Xt.dummy[:, 0]

        estimator.fit(X_da)
        Xt_da = estimator.transform(X_da)
        Xr_da = estimator.inverse_transform(Xt_da)

        assert_allclose(Xt_da, Xt)

        # test Dataset
        X_ds = self.X.var_3d.to_dataset()

        y = X_ds.var_3d[:, :5, 0].drop(['feat_1', 'feat_2'])
        y = y.rename({'feat_1': 'feature'})
        y['dummy'] = y.dummy[:, 0]

        estimator.fit(X_ds)
        yp = estimator.predict(X_ds).var_3d

        assert_allclose(yp, y)

    def test_sample_dim(self):

        from sklearn.decomposition import PCA

        estimator = wrap(PCA(n_components=5), reshapes='feat_1',
                       sample_dim='sample')

        # test DataArray
        X_da = self.X.var_2d

        Xt_da = estimator.fit_transform(X_da)
        Xr_da = estimator.inverse_transform(Xt_da)

        npt.assert_equal(Xt_da.shape, (100, 5))
        npt.assert_equal(Xr_da.shape, (100, 10))

        # test Dataset
        X_ds = self.X.var_2d.to_dataset()

        Xt = estimator.fit_transform(X_ds)

        npt.assert_equal(Xt.var_2d.shape, (100, 5))

    def test_score(self):

        from sklearn.linear_model import LinearRegression

        estimator = wrap(LinearRegression, reshapes='feat_1')

        # test DataArray
        X_da = self.X.var_2d

        y = np.random.random(100)

        estimator.fit(X_da, y)

        estimator.score(X_da, y)

        # test Dataset
        X_ds = self.X.var_2d.to_dataset()

        wrapper = estimator.fit(X_ds, y)

        wrapper.score(X_ds, y)


def test_classifier():

    svc = wrap(SVC)
    check_estimator(svc)
    assert hasattr(svc, 'predict')
    assert hasattr(svc, 'decision_function')

    svc_proba = wrap(SVC(probability=True))
    check_estimator(svc_proba)
    assert hasattr(svc_proba, 'predict_proba')
    assert hasattr(svc_proba, 'predict_log_proba')


def test_regressor():

    lr = wrap(LinearRegression)
    check_estimator(lr)
    assert hasattr(lr, 'predict')
    assert hasattr(lr, 'score')


def test_transformer():

    ss = wrap(StandardScaler)
    check_estimator(ss)
    assert hasattr(ss, 'partial_fit')
    assert hasattr(ss, 'transform')
    assert hasattr(ss, 'inverse_transform')
    assert hasattr(ss, 'fit_transform')
