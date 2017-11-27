import numpy as np
import xarray as xr

from sklearn_xarray.model_selection import CrossValidatorWrapper
from sklearn.model_selection import KFold, GroupKFold

def test_cross_validator():

    X_da = xr.DataArray(
        np.random.random((100, 10)),
        coords={'sample': range(100), 'feature': range(10)},
        dims=['sample', 'feature']
    )

    X_ds = xr.Dataset(
        {'var_1': (['sample', 'feature'], np.random.random((100, 10)))},
        coords={'sample': range(100), 'feature': range(10)}
    )

    cv = CrossValidatorWrapper(KFold(n_splits=3))

    assert cv.get_n_splits() == 3

    cv_list = list(cv.split(X_da))
    assert cv_list[0][0].shape[0] + cv_list[0][1].shape[0] == 100

    cv_list = list(cv.split(X_ds))
    assert cv_list[0][0].shape[0] + cv_list[0][1].shape[0] == 100


def test_cross_validator_groupwise():

    coord_1 = ['a']*51 + ['b']*49
    coord_2 = list(range(10))*10

    X_da = xr.DataArray(
        np.random.random((100, 10)),
        coords={'sample': range(100), 'feature': range(10),
                'coord_1': (['sample'], coord_1),
                'coord_2': (['sample'], coord_2)},
        dims=['sample', 'feature']
    )

    cv = CrossValidatorWrapper(GroupKFold(n_splits=2), groupby='coord_1')

    cv_list = list(cv.split(X_da))

    assert np.any([c.size == 51 for c in cv_list[0]])
