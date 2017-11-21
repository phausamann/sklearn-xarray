from sklearn.utils.estimator_checks import (
    check_estimator, check_transformer_general)
from sklearn_xarray import (EstimatorWrapper, TransformerWrapper)


def test_estimator():
    return check_estimator(EstimatorWrapper)


def test_transformer():
    return check_estimator(TransformerWrapper)