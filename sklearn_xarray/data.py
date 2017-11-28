""" ``sklearn_xarray.data`` """

import os
import tarfile

try:
    import urllib.request as ul
except ImportError:
    import urllib as ul

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


def load_wisdm(url='http://www.cis.fordham.edu/wisdm/includes/'
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
