import numpy as np
import xarray as xr

from numpy.testing import assert_equal

from sklearn_xarray.utils import (get_group_indices)


def test_get_group_indices():

    import itertools

    coord_1 = ['a']*50 + ['b']*50
    coord_2 = list(range(10))*10

    X = xr.DataArray(
        np.random.random((100, 10)),
        coords={'sample': range(100), 'feature': range(10),
                'coord_1': (['sample'], coord_1),
                'coord_2': (['sample'], coord_2)},
        dims=['sample', 'feature']
    )

    g1 = get_group_indices(X, 'coord_1')
    for i, gg in enumerate(g1):
        idx = np.array(coord_1) == np.unique(coord_1)[i]
        assert_equal(gg, idx)

    g2 = get_group_indices(X, ['coord_1', 'coord_2'])
    combinations = list(
        itertools.product(np.unique(coord_1), np.unique(coord_2)))
    for i, gg in enumerate(g2):
        idx = (np.array(coord_1) == combinations[i][0]) \
            & (np.array(coord_2) == combinations[i][1])
        assert_equal(gg, idx)
