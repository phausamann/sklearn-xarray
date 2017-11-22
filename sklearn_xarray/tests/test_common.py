from sklearn.utils.estimator_checks import check_estimator
from sklearn_xarray.dataarray import (
    EstimatorWrapper, TransformerWrapper)


def test_estimator():
    return check_estimator(EstimatorWrapper)


def test_transformer():
    return check_estimator(TransformerWrapper)
