from sklearn_xarray.datasets import (
    load_dummy_dataarray, load_dummy_dataset, load_digits_dataarray,
    load_wisdm_dataarray
)

import os
from sklearn_xarray import ROOT_DIR


def test_load_dummy_dataarray():

    load_dummy_dataarray()


def test_load_dummy_dataset():

    load_dummy_dataset()


def test_load_digits_dataarray():

    load_digits_dataarray(nan_probability=0.1)

    load_digits_dataarray(load_images=True, nan_probability=0.1)


def test_load_wisdm_dataarray():

    load_wisdm_dataarray(folder=os.path.join(ROOT_DIR, '../data'))
