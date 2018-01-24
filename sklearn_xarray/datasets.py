""" ``sklearn_xarray.datasets`` """

import numpy as np
import pandas as pd
import xarray as xr


def load_dummy_dataarray():
    """ Load a DataArray for demonstration purposes. """

    return xr.DataArray(
        np.random.random((100, 10)),
        coords={'sample': range(100), 'feature': range(10)},
        dims=('sample', 'feature')
    )


def load_dummy_dataset():
    """ Load a Dataset for demonstration purposes. """

    return xr.Dataset(
        {'var_1' : (['sample', 'feature'], np.random.random((100, 10)))},
        coords={'sample': range(100), 'feature': range(10)}
    )


def load_digits_dataarray(load_images=False, nan_probability=0):
    """ Load a the 'digits' dataset from sklearn as a DataArray.

    Parameters
    ----------
    load_images : bool, optional
        If true, the DataArray will contain the two-dimensional images as
        data instead of the vectorized samples.

    nan_probability : float between 0 and 1
        The probability with which a sample is injected with NaN values. For
        demonstration purposes only.
    """

    from sklearn.datasets import load_digits

    if load_images:

        bunch = load_digits()
        data = bunch.images

        if nan_probability > 0:
            for i in range(data.shape[0]):
                if np.random.rand(1) < nan_probability:
                    data[i, 0, 0] = np.nan

        return xr.DataArray(
            data,
            coords={'sample': range(data.shape[0]),
                    'row': range(data.shape[1]),
                    'col': range(data.shape[2]),
                    'digit': (['sample'], bunch.target)},
            dims=('sample', 'row', 'col')
        )

    else:

        data, target = load_digits(return_X_y=True)

        if nan_probability > 0:
            for i in range(data.shape[0]):
                if np.random.rand(1) < nan_probability:
                    data[i, 0] = np.nan

        return xr.DataArray(
            data,
            coords={'sample': range(data.shape[0]),
                    'feature': range(data.shape[1]),
                    'digit': (['sample'], target)},
            dims=('sample', 'feature')
        )


def load_wisdm_dataarray(url='http://www.cis.fordham.edu/wisdm/includes/'
                             'datasets/latest/WISDM_ar_latest.tar.gz',
                         file='WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt',
                         folder='data/', tmp_file='widsm.tar.gz'):
    """ Load the WISDM activity recognition dataset.

    Parameters
    ----------
    url : str, optional
        The URL of the dataset.

    file : str, optional
        The file containing the data.

    folder : str, optional
        The folder where the data will be downloaded and extracted to.

    tmp_file : str, optional
        The name of the temporary .tar file in the folder.

    Returns
    -------
    X: xarray DataArray
        The loaded dataset.

    """

    import os
    import tarfile
    import six.moves.urllib.request as ul

    if not os.path.isfile(os.path.join(folder, file)):
        ul.urlretrieve(url, tmp_file)
        tar = tarfile.open(tmp_file)
        tar.extractall(folder)
        tar.close()
        os.remove(tmp_file)

    column_names = ['subject', 'activity', 'timestamp', 'x', 'y', 'z']
    df = pd.read_csv(os.path.join(folder, file), header=None,
                     names=column_names, comment=';')

    time = pd.TimedeltaIndex(start=0, periods=df.shape[0], freq='50ms')

    coords = {'subject': ('sample', df.subject),
              'activity': ('sample', df.activity),
              'sample': time,
              'axis': ['x', 'y', 'z']}

    X = xr.DataArray(
        df.iloc[:, 3:6], coords=coords, dims=('sample', 'axis'))

    return X
