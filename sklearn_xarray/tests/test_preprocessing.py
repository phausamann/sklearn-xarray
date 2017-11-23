import numpy as np
import xarray as xr
import xarray.testing as xrt
import numpy.testing as npt

from sklearn_xarray.preprocessing import (
    preprocess, transpose, split, segment, resample, concatenate, featurize,
    sanitize, reduce, Transposer, Splitter
)


def test_preprocess():

    from sklearn.preprocessing import scale

    X_da = xr.DataArray(
        np.random.random((100, 10)),
        coords={'sample': range(100), 'feature': range(10)},
        dims=('sample', 'feature')
    )

    Xt_da_gt = X_da
    Xt_da_gt.data = scale(X_da)

    Xt_da = preprocess(X_da, scale)

    xrt.assert_allclose(Xt_da, Xt_da_gt)

    X_ds = xr.Dataset(
        {'var_1' : (['sample', 'feature'], np.random.random((100, 10)))},
        coords={'sample': range(100), 'feature': range(10)}
    )

    Xt_ds = preprocess(X_ds, scale)

    xrt.assert_allclose(Xt_ds, X_ds.apply(scale))


def test_groupwise():

    from sklearn.preprocessing import scale

    coord_1 = ['a']*51 + ['b']*49
    coord_2 = list(range(10))*10

    X_ds = xr.Dataset(
        {'var_1' : (['sample', 'feature'], np.random.random((100, 10)))},
        coords={'sample': range(100), 'feature': range(10),
                'coord_1': (['sample'], coord_1),
                'coord_2': (['sample'], coord_2)}
    )

    # test wrapped sklearn estimator
    Xt_ds = preprocess(X_ds, scale, groupby='coord_1')

    # test newly defined estimator
    Xt_ds2 = split(X_ds, new_dim='split_sample', new_len=5, groupby='coord_1')

    assert Xt_ds2.var_1.shape == (19, 10, 5)


def test_transpose():

    # test on DataArray
    X_da = xr.DataArray(
        np.random.random((100, 10)),
        coords={'sample': range(100), 'feature': range(10)},
        dims=('sample', 'feature')
    )

    Xt_da, estimator = transpose(
        X_da, order=['feature', 'sample'], return_estimator=True)

    xrt.assert_allclose(Xt_da, X_da.transpose())

    Xt_da = estimator.inverse_transform(Xt_da)

    xrt.assert_allclose(Xt_da, X_da)

    # test on Dataset
    X_ds = xr.Dataset(
        {'var_1': (['sample', 'feature'], np.random.random((100, 10)))},
        coords={'sample': range(100), 'feature': range(10)}
    )

    Xt_ds = transpose(X_ds, order=['feature', 'sample'])

    xrt.assert_allclose(Xt_ds, X_ds.transpose())


def test_split():

    # test on DataArray with number of samples multiple of new length
    X_da = xr.DataArray(
        np.random.random((100, 10)),
        coords={'sample': range(100), 'feature': range(10)},
        dims=('sample', 'feature')
    )

    estimator = Splitter(
        new_dim='split_sample', new_len=5, reduce_index='subsample',
        keep_coords_as='sample_coord'
    )

    Xt_da = estimator.fit_transform(X_da)

    assert Xt_da.shape == (20, 10, 5)

    Xt_da = estimator.inverse_transform(Xt_da)

    xrt.assert_allclose(X_da, Xt_da)

    # test on Dataset with number of samples NOT multiple of new length
    X_ds = xr.Dataset(
        {'var_1': (['sample', 'feature'], np.random.random((100, 10)))},
        coords={'sample': range(100), 'feature': range(10)}
    )

    Xt_ds = split(
        X_ds, new_dim='split_sample', new_len=7, reduce_index='head',
        new_index_func=None
    )

    assert Xt_ds['var_1'].shape == (14, 10, 7)


def test_segment():

    X_da = xr.DataArray(
        np.random.random((100, 10)),
        coords={'sample': range(100), 'feature': range(10)},
        dims=('sample', 'feature')
    )

    Xt_da = segment(
        X_da, new_dim='split_sample', new_len=10, step=5,
        reduce_index='subsample')

    X_ds = xr.Dataset(
        {'var_1': (['sample', 'feature'], np.random.random((100, 10)))},
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
                'feature': range(10)},
        dims=('sample', 'feature')
    )

    Xt_da = resample(X_da, freq='20ms')

    X_ds = xr.Dataset(
        {'var_1': (['sample', 'feature'], np.random.random((100, 10)))},
        coords={'sample': pd.timedelta_range(0, periods=100, freq='10ms'),
                'feature': range(10)}
    )

    Xt_ds = resample(X_ds, freq='20ms')

    #TODO: check result


def test_concatenate():

    X_ds = xr.Dataset(
        {'var_1': (['sample', 'feature'], np.random.random((100, 10))),
         'var_2': (['sample', 'feature'], np.random.random((100, 10))),
         'var_3': (['sample', 'feature'], np.random.random((100, 10)))},
        coords={'sample': range(100), 'feature': range(10)}
    )

    Xt_ds = concatenate(X_ds)

    assert Xt_ds.Feature.shape == (100, 30)

    Xt_ds2 = concatenate(
        X_ds, variables=['var_1', 'var_2'], new_index_func=np.arange)

    #TODO: check result


def test_featurize():

    X_da = xr.DataArray(
        np.random.random((100, 10, 10)),
        coords={'sample': range(100), 'feat_1': range(10), 'feat_2': range(10)},
        dims=('sample', 'feat_1', 'feat_2')
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
        dims=('sample', 'feature')
    )

    X_da[0, 0] = np.nan

    Xt_da = sanitize(X_da)

    X_ds = xr.Dataset(
        {'var_1': (['sample', 'feature'], np.random.random((100, 10)))},
        coords={'sample': range(100), 'feature': range(10)}
    )

    X_ds['var_1'][0, 0] = np.nan

    Xt_ds = sanitize(X_ds)

    #TODO: check result


def test_reduce():

    X_da = xr.DataArray(
        np.random.random((100, 10)),
        coords={'sample': range(100), 'feature': range(10)},
        dims=('sample', 'feature')
    )

    Xt_da = reduce(X_da)

    xrt.assert_allclose(Xt_da, X_da.reduce(np.linalg.norm, dim='feature'))
