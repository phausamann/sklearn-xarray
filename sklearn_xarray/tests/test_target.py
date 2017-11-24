import numpy as np
import xarray as xr
import xarray.testing as xrt
import numpy.testing as npt

from sklearn_xarray import Target
from sklearn.preprocessing import LabelBinarizer


def test_constructor():

    coord_1 = ['a']*51 + ['b']*49
    coord_2 = list(range(10))*10

    X_ds = xr.Dataset(
        {'var_1': (['sample', 'feature'], np.random.random((100, 10)))},
        coords={'sample': range(100), 'feature': range(10),
                'coord_1': (['sample'], coord_1),
                'coord_2': (['sample'], coord_2)}
    )

    target = Target()
    target.assign_to(X_ds)

    npt.assert_equal(target.values, np.array(X_ds.var_1))

    target = Target(coord='coord_1', transformer=LabelBinarizer())
    target.assign_to(X_ds)

    npt.assert_equal(
        target.values, LabelBinarizer().fit_transform(coord_1))


def test_array():

    coord_1 = ['a']*51 + ['b']*49
    coord_2 = list(range(10))*10

    X_ds = xr.Dataset(
        {'var_1': (['sample', 'feature'], np.random.random((100, 10)))},
        coords={'sample': range(100), 'feature': range(10),
                'coord_1': (['sample'], coord_1),
                'coord_2': (['sample'], coord_2)}
    )

    target = Target(coord='coord_1', transformer=LabelBinarizer(), lazy=True)

    target.assign_to(X_ds)

    npt.assert_equal(np.array(target), LabelBinarizer().fit_transform(coord_1))


def test_getitem():

    coord_1 = ['a']*51 + ['b']*49
    coord_2 = list(range(10))*10

    X_ds = xr.Dataset(
        {'var_1': (['sample', 'feature'], np.random.random((100, 10)))},
        coords={'sample': range(100), 'feature': range(10),
                'coord_1': (['sample'], coord_1),
                'coord_2': (['sample'], coord_2)}
    )

    target = Target(coord='coord_1', transformer=LabelBinarizer())

    target.assign_to(X_ds)

    y_test = target[-1]

    assert y_test == LabelBinarizer().fit_transform(coord_1)[-1]


def test_shape_and_ndim():

    coord_1 = ['a']*51 + ['b']*49
    coord_2 = list(range(10))*10

    X_ds = xr.Dataset(
        {'var_1': (['sample', 'feature'], np.random.random((100, 10)))},
        coords={'sample': range(100), 'feature': range(10),
                'coord_1': (['sample'], coord_1),
                'coord_2': (['sample'], coord_2)}
    )

    target = Target(coord='coord_1', transformer=LabelBinarizer())

    target.assign_to(X_ds)

    npt.assert_equal(
        target.shape, LabelBinarizer().fit_transform(coord_1).shape)

    npt.assert_equal(
        target.ndim, LabelBinarizer().fit_transform(coord_1).ndim)
