from sklearn_xarray.data import (
    load_dummy_dataarray, load_dummy_dataset, load_digits_dataarray,
    load_wisdm_dataarray
)

import os
from sklearn_xarray import ROOT_DIR

def test_load_dummy_dataarray():

    X = load_dummy_dataarray()


def test_load_dummy_dataset():

    X = load_dummy_dataset()


def test_load_digits_dataarray():

    X = load_digits_dataarray()

    X = load_digits_dataarray(load_images=True)


def test_load_wisdm_dataarray():

    X = load_wisdm_dataarray(folder=os.path.join(ROOT_DIR, '../data'))
