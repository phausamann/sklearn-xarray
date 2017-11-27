import os
from sklearn_xarray.datasets import (
    load_dummy_dataarray, load_dummy_dataset, load_wisdm)


def test_load_dummy_datarray():

    X = load_dummy_dataarray()


def test_load_dummy_dataset():

    X = load_dummy_dataset()


def test_load_wisdm():

    X = load_wisdm()
