from unittest import TestCase

import numpy as np
import xarray as xr
from xarray.testing import assert_equal, assert_allclose
import numpy.testing as npt

from sklearn_xarray import wrap

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, KernelCenterer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC

from tests.mocks import (
    DummyEstimator,
    DummyTransformer,
    ReshapingEstimator,
)


class EstimatorWrapperTests(TestCase):
    def setUp(self):

        self.X = xr.Dataset(
            {
                "var_2d": (["sample", "feat_1"], np.random.random((100, 10))),
                "var_3d": (
                    ["sample", "feat_1", "feat_2"],
                    np.random.random((100, 10, 10)),
                ),
            },
            {
                "sample": range(100),
                "feat_1": range(10),
                "feat_2": range(10),
                "dummy": (["sample", "feat_1"], np.random.random((100, 10))),
            },
        )

    def test_update_restore_dims(self):

        estimator = wrap(
            ReshapingEstimator(new_shape=(-1, 0, 5)),
            reshapes={"feature": ["feat_1", "feat_2"]},
        )

        X = self.X.var_3d

        estimator.fit(X)

        X_out = estimator.estimator_.transform(X.values)
        dims_new = estimator._update_dims(X, X_out)
        Xt = xr.DataArray(X_out, dims=dims_new)

        assert dims_new == ["sample", "feature"]

        Xr_out = estimator.estimator_.inverse_transform(X_out)
        dims_old = estimator._restore_dims(Xt, Xr_out)

        assert dims_old == ["sample", "feat_1", "feat_2"]

    def test_update_coords(self):

        pass

    def test_params(self):

        estimator = StandardScaler(with_mean=False)
        params = estimator.get_params()
        params.update(
            {"estimator": estimator, "reshapes": None, "sample_dim": None}
        )

        # check params set in constructor
        wrapper = wrap(estimator)
        self.assertEqual(wrapper.get_params(), params)
        self.assertEqual(wrapper.with_mean, False)

        # check params set by attribute
        wrapper.with_std = False
        params.update({"with_std": False})
        self.assertEqual(wrapper.get_params(), params)

        # check params set with set_params
        wrapper.set_params(copy=False)
        params.update({"copy": False})
        self.assertEqual(wrapper.get_params(), params)

    def test_attributes(self):

        estimator = wrap(StandardScaler())

        # check pass-through wrapper
        estimator.fit(self.X.var_2d.values)
        npt.assert_allclose(estimator.mean_, estimator.estimator_.mean_)

        # check DataArray wrapper
        estimator.fit(self.X.var_2d)
        npt.assert_allclose(estimator.mean_, estimator.estimator_.mean_)

        # check Dataset wrapper
        estimator.fit(self.X.var_2d.to_dataset())
        npt.assert_allclose(
            estimator.mean_["var_2d"],
            estimator.estimator_dict_["var_2d"].mean_,
        )


class PublicInterfaceTests(TestCase):
    def setUp(self):

        self.X = xr.Dataset(
            {
                "var_2d": (["sample", "feat_1"], np.random.random((100, 10))),
                "var_3d": (
                    ["sample", "feat_1", "feat_2"],
                    np.random.random((100, 10, 10)),
                ),
            },
            {
                "sample": range(100),
                "feat_1": range(10),
                "feat_2": range(10),
                "dummy": (["sample", "feat_1"], np.random.random((100, 10))),
            },
        )

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

        estimator = wrap(StandardScaler())

        # test DataArray
        X_da = self.X.var_2d

        estimator.partial_fit(X_da)

        assert_allclose(
            X_da, estimator.inverse_transform(estimator.transform(X_da))
        )

        # test Dataset
        X_ds = self.X.var_2d.to_dataset()

        estimator.fit(X_ds)

        assert_allclose(
            X_ds, estimator.inverse_transform(estimator.transform(X_ds))
        )

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
            ReshapingEstimator(new_shape=(-1, 2)), reshapes="feat_1"
        )

        # test DataArray
        X_da = self.X.var_2d

        y = X_da[:, :2].drop("feat_1")
        y["dummy"] = y.dummy[:, 0]

        estimator.fit(X_da)
        yp = estimator.predict(X_da)

        assert_allclose(yp, y)

        # test Dataset
        X_ds = self.X.var_2d.to_dataset()

        y = X_ds.var_2d[:, :2].drop("feat_1")
        y["dummy"] = y.dummy[:, 0]

        estimator.fit(X_ds)
        yp = estimator.predict(X_ds).var_2d

        assert_allclose(yp, y)

    def test_reshaping_transformer(self):

        estimator = wrap(
            ReshapingEstimator(new_shape=(-1, 2)), reshapes="feat_1"
        )

        # test DataArray
        X_da = self.X.var_3d

        y = X_da[:, :2].drop("feat_1")
        y["dummy"] = y.dummy[:, 0]

        estimator.fit(X_da)
        yp = estimator.transform(X_da)

        assert_allclose(yp, y)

        # test Dataset
        X_ds = self.X.var_2d.to_dataset()

        y = X_ds.var_2d[:, :2].drop("feat_1")
        y["dummy"] = y.dummy[:, 0]

        estimator.fit(X_ds)
        yp = estimator.transform(X_ds).var_2d

        assert_allclose(yp, y)

    def test_reshaping_estimator_singleton(self):

        estimator = wrap(
            ReshapingEstimator(new_shape=(-1, 0)), reshapes="feat_1"
        )

        # test DataArray
        X_da = self.X.var_2d

        y = X_da[:, 0].drop("feat_1")
        estimator.fit(X_da)
        yp = estimator.predict(X_da)

        assert_allclose(yp, y)

        # test Dataset
        X_ds = self.X

        y = X_ds.var_2d[:, 0].drop("feat_1")

        estimator.fit(X_ds)
        yp = estimator.predict(X_ds).var_2d

        assert_allclose(yp, y)

    def test_ndim_reshaping_estimator(self):

        estimator = wrap(
            ReshapingEstimator(new_shape=(-1, 5, 0)),
            reshapes={"feature": ["feat_1", "feat_2"]},
        )

        # test DataArray
        X_da = self.X.var_3d

        Xt = (
            X_da[:, :5, 0]
            .drop(["feat_1", "feat_2"])
            .rename({"feat_1": "feature"})
        )
        Xt["dummy"] = Xt.dummy[:, 0]

        estimator.fit(X_da)
        Xt_da = estimator.transform(X_da)
        estimator.inverse_transform(Xt_da)

        assert_allclose(Xt_da, Xt)

        # test Dataset
        X_ds = self.X.var_3d.to_dataset()

        y = X_ds.var_3d[:, :5, 0].drop(["feat_1", "feat_2"])
        y = y.rename({"feat_1": "feature"})
        y["dummy"] = y.dummy[:, 0]

        estimator.fit(X_ds)
        yp = estimator.predict(X_ds).var_3d

        assert_allclose(yp, y)

    def test_sample_dim(self):

        from sklearn.decomposition import PCA

        estimator = wrap(
            PCA(n_components=5), reshapes="feat_1", sample_dim="sample"
        )

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

        estimator = wrap(LinearRegression, reshapes="feat_1")

        # test DataArray
        X_da = self.X.var_2d

        y = np.random.random(100)

        estimator.fit(X_da, y)

        estimator.score(X_da, y)

        # test Dataset
        X_ds = self.X.var_2d.to_dataset()

        wrapper = estimator.fit(X_ds, y)

        wrapper.score(X_ds, y)

    def test_partial_fit(self):

        estimator = wrap(StandardScaler())

        # check pass-through wrapper
        estimator.partial_fit(self.X.var_2d.values)
        assert hasattr(estimator, "mean_")

        with self.assertRaises(ValueError):
            estimator.partial_fit(self.X.var_2d)
        with self.assertRaises(ValueError):
            estimator.partial_fit(self.X)

        # check DataArray wrapper
        estimator = clone(estimator)
        estimator.partial_fit(self.X.var_2d)

        with self.assertRaises(ValueError):
            estimator.partial_fit(self.X.var_2d.values)
        with self.assertRaises(ValueError):
            estimator.partial_fit(self.X)
        assert hasattr(estimator, "mean_")

        # check Dataset wrapper
        estimator = clone(estimator)
        estimator.partial_fit(self.X.var_2d.to_dataset())

        with self.assertRaises(ValueError):
            estimator.partial_fit(self.X.var_2d.values)
        with self.assertRaises(ValueError):
            estimator.partial_fit(self.X.var_2d)
        assert hasattr(estimator, "mean_")


def test_classifier():

    lr = wrap(LogisticRegression)
    # wrappers don't pass check_estimator anymore because estimators
    # "should not set any attribute apart from parameters during init"
    assert hasattr(lr, "predict")
    assert hasattr(lr, "decision_function")

    lr = wrap(LogisticRegression)
    assert hasattr(lr, "C")

    svc_proba = wrap(SVC(probability=True))
    # check_estimator(svc_proba) fails because the wrapper is not excluded
    # from tests that are known to fail for SVC...
    assert hasattr(svc_proba, "predict_proba")
    assert hasattr(svc_proba, "predict_log_proba")


def test_regressor():

    lr = wrap(LinearRegression, compat=True)
    assert hasattr(lr, "predict")
    assert hasattr(lr, "score")

    lr = wrap(LinearRegression)
    assert hasattr(lr, "normalize")


def test_transformer():

    wrap(KernelCenterer, compat=True)

    tr = wrap(KernelCenterer)
    assert hasattr(tr, "transform")

    ss = wrap(StandardScaler)
    # check_estimator(ss) fails because the wrapper is not excluded
    # from tests that are known to fail for StandardScaler...
    assert hasattr(ss, "partial_fit")
    assert hasattr(ss, "inverse_transform")
    assert hasattr(ss, "fit_transform")
