import numpy as np
import xarray as xr
import xarray.testing as xrt
import numpy.testing as npt

from sklearn_xarray import Target
from sklearn.preprocessing import LabelBinarizer


def test_constructor():

    from sklearn_xarray.utils import convert_to_ndarray

    coord_1 = ['a']*51 + ['b']*49
    coord_2 = list(range(10))*10

    X_ds = xr.Dataset(
        {'var_1': (['sample', 'feature'], np.random.random((100, 10)))},
        coords={'sample': range(100), 'feature': range(10),
                'coord_1': (['sample'], coord_1),
                'coord_2': (['sample'], coord_2)}
    )

    target = Target(transform_func=convert_to_ndarray)
    target.assign_to(X_ds)

    npt.assert_equal(target.values, np.array(X_ds.var_1))

    target = Target(coord='coord_1', transformer=LabelBinarizer())(X_ds)

    npt.assert_equal(
        target.values, LabelBinarizer().fit_transform(coord_1))


def test_str():

    assert str(Target()).startswith(
        'Unassigned sklearn_xarray.Target without coordinate')

    assert str(Target(coord='test')).startswith(
        'Unassigned sklearn_xarray.Target with coordinate "test"')

    assert str(Target()(np.ones(10))).startswith(
        'sklearn_xarray.Target with data:')


def test_array():

    coord_1 = ['a']*51 + ['b']*49
    coord_2 = list(range(10))*10

    X_ds = xr.Dataset(
        {'var_1': (['sample', 'feature'], np.random.random((100, 10)))},
        coords={'sample': range(100), 'feature': range(10),
                'coord_1': (['sample'], coord_1),
                'coord_2': (['sample'], coord_2)}
    )

    target = Target(
        coord='coord_1', transform_func=LabelBinarizer().fit_transform,
        lazy=True)(X_ds)

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

    target = Target(
        coord='coord_1', transform_func=LabelBinarizer().fit_transform)(X_ds)

    y_test = target[-1]

    assert y_test == LabelBinarizer().fit_transform(coord_1)[-1]

    # test lazy eval
    target = Target(
        coord='coord_1', transform_func=LabelBinarizer().fit_transform,
        lazy=True)(X_ds)

    y_test = target[-1]

    assert y_test == LabelBinarizer().fit_transform(coord_1)[-1]
    assert not y_test.lazy


def test_shape_and_ndim():

    coord_1 = ['a']*51 + ['b']*49
    coord_2 = list(range(10))*10

    X_ds = xr.Dataset(
        {'var_1': (['sample', 'feature'], np.random.random((100, 10)))},
        coords={'sample': range(100), 'feature': range(10),
                'coord_1': (['sample'], coord_1),
                'coord_2': (['sample'], coord_2)}
    )

    target = Target(
        coord='coord_1', transform_func=LabelBinarizer().fit_transform)(X_ds)

    npt.assert_equal(
        target.shape, LabelBinarizer().fit_transform(coord_1).shape)

    npt.assert_equal(
        target.ndim, LabelBinarizer().fit_transform(coord_1).ndim)


def test_multidim_coord():

    coord_1 = np.tile(['a']*51 + ['b']*49, (10, 1)).T
    coord_2 = np.random.random((100, 10, 10))

    X_ds = xr.Dataset(
        {'var_1': (['sample', 'feat_1', 'feat_2'],
                   np.random.random((100, 10, 10)))},
        coords={'sample': range(100), 'feature': range(10),
                'coord_1': (['sample', 'feat_1'], coord_1),
                'coord_2': (['sample', 'feat_1', 'feat_2'], coord_2)}
    )

    target_1 = Target(
        coord='coord_1', transform_func=LabelBinarizer().fit_transform,
        dim='sample')(X_ds)
    target_2 = Target(
        coord='coord_2', dim=['sample', 'feat_1'], reduce_func=np.mean)(X_ds)

    npt.assert_equal(target_1, LabelBinarizer().fit_transform(coord_1[:, 0]))
    npt.assert_equal(target_2, np.mean(coord_2, 2))
