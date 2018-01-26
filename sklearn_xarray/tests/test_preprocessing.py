import numpy as np
import xarray as xr
import xarray.testing as xrt
import numpy.testing as npt

from sklearn_xarray.preprocessing import (
    preprocess, transpose, split, segment, resample, concatenate, featurize,
    sanitize, reduce, Splitter
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
        {'var_1': (['sample', 'feature'], np.random.random((100, 10)))},
        coords={'sample': range(100), 'feature': range(10),
                'coord_1': (['sample'], coord_1),
                'coord_2': (['sample'], coord_2)}
    )

    # test wrapped sklearn estimator
    Xt_ds = preprocess(X_ds, scale, groupby='coord_1')

    # test newly defined estimator
    Xt_ds2, estimator = split(
        X_ds, new_dim='split_sample', new_len=5, groupby='coord_1',
        keep_coords_as='initial_sample', return_estimator=True
    )

    assert Xt_ds2.var_1.shape == (19, 10, 5)

    Xt_ds2 = estimator.inverse_transform(Xt_ds2)

    assert Xt_ds2.var_1.shape == (95, 10)


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

    # test on Dataset with subset of dimensions
    X_ds = xr.Dataset(
        {'var_1': (['sample', 'feat_1', 'feat_2'],
                   np.random.random((100, 10, 5))),
         'var_2': (['feat_2', 'sample'],
                   np.random.random((5, 100)))},
        coords={'sample': range(100), 'feat_1': range(10), 'feat_2': range(5)}
    )

    Xt_ds, estimator = transpose(
        X_ds, order=['sample', 'feat_2'], return_estimator=True)

    xrt.assert_allclose(Xt_ds, X_ds.transpose('sample', 'feat_1', 'feat_2'))

    Xt_ds = estimator.inverse_transform(Xt_ds)

    xrt.assert_allclose(Xt_ds, X_ds)


def test_split():

    # test on DataArray with number of samples multiple of new length
    X_da = xr.DataArray(
        np.random.random((100, 10)),
        coords={'sample': range(100), 'feature': range(10),
                'coord_1': (['sample', 'feature'],
                            np.tile('Test', (100, 10)))},
        dims=('sample', 'feature')
    )

    estimator = Splitter(
        new_dim='split_sample', new_len=5, reduce_index='subsample', axis=1,
        keep_coords_as='sample_coord'
    )

    Xt_da = estimator.fit_transform(X_da)

    assert Xt_da.shape == (20, 5, 10)
    npt.assert_allclose(Xt_da[0, :, 0], X_da[:5, 0])

    Xit_da = estimator.inverse_transform(Xt_da)

    xrt.assert_allclose(X_da, Xit_da)

    # test on Dataset with number of samples NOT multiple of new length
    X_ds = xr.Dataset(
        {'var_1': (['sample', 'feature'], np.random.random((100, 10)))},
        coords={'sample': range(100), 'feature': range(10)}
    )

    Xt_ds = split(
        X_ds, new_dim='split_sample', new_len=7, reduce_index='head',
        axis=1, new_index_func=None
    )

    assert Xt_ds['var_1'].shape == (14, 7, 10)
    npt.assert_allclose(Xt_ds.var_1[0, :, 0], X_ds.var_1[:7, 0])


def test_segment():

    X_da = xr.DataArray(
        np.tile(np.arange(10), (100, 1)),
        coords={'sample': range(100), 'feature': range(10),
                'coord_1': (['sample', 'feature'],
                            np.tile('Test', (100, 10)))},
        dims=('sample', 'feature')
    )

    Xt_da, estimator = segment(
        X_da, new_dim='split_sample', new_len=20, step=5, axis=0,
        reduce_index='subsample', keep_coords_as='backup', return_estimator=True
    )

    assert Xt_da.coord_1.shape == (20, 17, 10)
    npt.assert_allclose(Xt_da[:, 0, 0], X_da[:20, 0])

    Xit_da = estimator.inverse_transform(Xt_da)

    xrt.assert_allclose(Xit_da, X_da)

    X_ds = xr.Dataset({
        'var_1': (['sample', 'feat_1', 'feat_2'],
                  np.tile(np.arange(10), (100, 10, 1))),
        'var_2': (['feat_2'], np.random.random((10,)))},
        coords={'sample': range(100), 'feat_1': range(10), 'feat_2': range(10),
                'coord_1': (['sample', 'feat_1'], np.tile('Test', (100, 10)))}
    )

    Xt_ds, estimator = segment(
        X_ds, new_dim='split_sample', new_len=20, step=5, reduce_index='head',
        keep_coords_as='backup', return_estimator=True
    )

    assert Xt_ds.var_1.shape == (17, 10, 10, 20)
    npt.assert_allclose(Xt_ds.var_1[0, 0, 0, :], X_ds.var_1[:20, 0, 0])

    xrt.assert_allclose(estimator.inverse_transform(Xt_ds), X_ds)


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

    assert Xt_ds2.Feature.shape == (100, 20)
    npt.assert_equal(Xt_ds2.feature.values, np.arange(20))


def test_featurize():

    X_da = xr.DataArray(
        np.random.random((100, 10, 10)),
        coords={'sample': range(100), 'feat_1': range(10), 'feat_2': range(10)},
        dims=('sample', 'feat_1', 'feat_2')
    )

    Xt_da = featurize(X_da)

    assert Xt_da.shape == (100, 100)

    X_ds = xr.Dataset(
        {'var_1': (['sample', 'feat_1', 'feat_2'],
                   np.random.random((100, 10, 10))),
         'var_2': (['sample', 'feat_1'],
                   np.random.random((100, 10)))},
        coords={'sample': range(100), 'feat_1': range(10), 'feat_2': range(10)}
    )

    Xt_ds = featurize(X_ds, return_array=True)

    assert Xt_ds.shape == (100, 110)


def test_sanitize():

    X_da = xr.DataArray(
        np.random.random((100, 10)),
        coords={'sample': range(100), 'feature': range(10)},
        dims=('sample', 'feature')
    )

    X_da[0, 0] = np.nan

    Xt_da = sanitize(X_da)

    xrt.assert_allclose(X_da[1:], Xt_da)

    X_ds = xr.Dataset(
        {'var_1': (['sample', 'feature'], np.random.random((100, 10)))},
        coords={'sample': range(100), 'feature': range(10)}
    )

    X_ds['var_1'][0, 0] = np.nan

    Xt_ds = sanitize(X_ds)

    xrt.assert_allclose(X_ds.isel(sample=range(1, 100)), Xt_ds)


def test_reduce():

    X_da = xr.DataArray(
        np.random.random((100, 10)),
        coords={'sample': range(100), 'feature': range(10)},
        dims=('sample', 'feature')
    )

    Xt_da = reduce(X_da)

    xrt.assert_allclose(Xt_da, X_da.reduce(np.linalg.norm, dim='feature'))
