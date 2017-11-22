import numpy as np
import xarray as xr
from xarray.testing import assert_equal, assert_allclose

from sklearn_xarray.preprocessing import (
    preprocess, transpose, split, segment, resample, concatenate, featurize,
    sanitize, reduce
)


def test_preprocess():

    from sklearn.preprocessing import scale

    X_da = xr.DataArray(
        np.random.random((100, 10)),
        coords={'sample': range(100), 'feature': range(10)}
    )

    Xt_da_gt = X_da
    Xt_da_gt.data = scale(X_da)

    Xt_da = preprocess(X_da, scale)

    assert_allclose(Xt_da, Xt_da_gt)

    X_ds = xr.Dataset(
        {'var' : (['sample', 'feature'], np.random.random((100, 10)))},
        coords={'sample': range(100), 'feature': range(10)}
    )

    Xt_ds = preprocess(X_ds, scale)

    assert_allclose(Xt_ds, X_ds.apply(scale))


def test_groupwise():

    from sklearn.preprocessing import scale

    coord_1 = ['a']*50 + ['b']*50
    coord_2 = list(range(10))*10

    X_ds = xr.Dataset(
        {'var' : (['sample', 'feature'], np.random.random((100, 10)))},
        coords={'sample': range(100), 'feature': range(10),
                'coord_1': (['sample'], coord_1),
                'coord_2': (['sample'], coord_2)}
    )

    Xt_ds = preprocess(X_ds, scale, groupby='coord_1')

    Xt_ds2 = transpose(X_ds, order=['feature', 'sample'], groupby='coord_1')

    # TODO: check result


def test_transpose():

    X_da = xr.DataArray(
        np.random.random((100, 10)),
        coords={'sample': range(100), 'feature': range(10)}
    )

    Xt_da = transpose(X_da, order=['feature', 'sample'])

    assert_allclose(Xt_da, X_da.transpose())

    X_ds = xr.Dataset(
        {'var' : (['sample', 'feature'], np.random.random((100, 10)))},
        coords={'sample': range(100), 'feature': range(10)}
    )

    Xt_ds = transpose(X_ds, order=['feature', 'sample'])

    assert_allclose(Xt_ds, X_ds.transpose())


def test_split():

    X_da = xr.DataArray(
        np.random.random((100, 10)),
        coords={'sample': range(100), 'feature': range(10)}
    )

    Xt_da = split(
        X_da, new_dim='split_sample', new_len=10, reduce_index='subsample')

    X_ds = xr.Dataset(
        {'var': (['sample', 'feature'], np.random.random((100, 10)))},
        coords={'sample': range(100), 'feature': range(10)}
    )

    Xt_ds = split(
        X_ds, new_dim='split_sample', new_len=10, reduce_index='head')

    #TODO: check result


def test_segment():

    X_da = xr.DataArray(
        np.random.random((100, 10)),
        coords={'sample': range(100), 'feature': range(10)}
    )

    Xt_da = segment(
        X_da, new_dim='split_sample', new_len=10, step=5,
        reduce_index='subsample')

    X_ds = xr.Dataset(
        {'var': (['sample', 'feature'], np.random.random((100, 10)))},
        coords={'sample': range(100), 'feature': range(10)}
    )

    Xt_ds = segment(
        X_ds, new_dim='split_sample', new_len=10, step=5, reduce_index='head')

    #TODO: check result


def test_resample():

    import pandas as pd

    X_da = xr.DataArray(
        np.random.random((100, 10)),
        coords={'sample': pd.timedelta_range(0, periods=100, freq='10ms'),
                'feature': range(10)}
    )

    Xt_da = resample(X_da, freq='20ms')

    X_ds = xr.Dataset(
        {'var': (['sample', 'feature'], np.random.random((100, 10)))},
        coords={'sample': pd.timedelta_range(0, periods=100, freq='10ms'),
                'feature': range(10)}
    )

    Xt_ds = resample(X_ds, freq='20ms')

    #TODO: check result


def test_concatenate():

    X_ds = xr.Dataset(
        {'var_1': (['sample', 'feature'], np.random.random((100, 10))),
         'var_2': (['sample', 'feature'], np.random.random((100, 10)))},
        coords={'sample': range(100), 'feature': range(10)}
    )

    Xt_ds = concatenate(X_ds)

    #TODO: check result


def test_featurize():

    X_da = xr.DataArray(
        np.random.random((100, 10, 10)),
        coords={'sample': range(100), 'feat_1': range(10), 'feat_2': range(10)}
    )

    Xt_da = featurize(X_da)

    X_ds = xr.Dataset(
        {'var_1': (['sample', 'feat_1', 'feat_2'],
                   np.random.random((100, 10, 10))),
         'var_2': (['sample', 'feat_1', 'feat_2'],
                   np.random.random((100, 10, 10)))},
        coords={'sample': range(100), 'feat_1': range(10), 'feat_2': range(10)}
    )

    Xt_ds = featurize(X_ds)

    #TODO: check result


def test_sanitize():

    X_da = xr.DataArray(
        np.random.random((100, 10)),
        coords={'sample': range(100), 'feature': range(10)},
    )

    X_da[0, 0] = np.nan

    Xt_da = sanitize(X_da)

    X_ds = xr.Dataset(
        {'var': (['sample', 'feature'], np.random.random((100, 10)))},
        coords={'sample': range(100), 'feature': range(10)}
    )

    X_ds['var'][0, 0] = np.nan

    Xt_ds = sanitize(X_ds)

    #TODO: check result


def test_reduce():

    X_da = xr.DataArray(
        np.random.random((100, 10)),
        coords={'sample': range(100), 'feature': range(10)}
    )

    Xt_da = reduce(X_da)

    assert_allclose(Xt_da, X_da.reduce(np.linalg.norm, dim='feature'))
